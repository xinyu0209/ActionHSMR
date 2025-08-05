from lib.kits.basic import *

from concurrent.futures import ThreadPoolExecutor
from lightning_fabric.utilities.rank_zero import _get_rank

from lib.data.augmentation.skel import rot_skel_on_plane
from lib.utils.data import to_tensor, to_numpy
from lib.utils.camera import perspective_projection, estimate_camera_trans
from lib.utils.vis import Wis3D
from lib.modeling.optim.skelify.skelify import SKELify
from lib.body_models.common import make_SKEL

DEBUG = False
DEBUG_ROUND = False

#在训练过程中通过SKELify进行伪标签优化
class SKELifySPIN(pl.Callback):
    '''
    Call SKELify to optimize the prediction results.
    Here we have several concepts of data: gt, opgt, pd, (res), bpgt.

    1. `gt`: Static Ground Truth: they are loaded from static training datasets,
       they might be real ground truth (like 2D keypoints), or pseudo ground truth (like SKEL
       parameters). They will be gradually replaced by the better pseudo ground truth through
       iterations in anticipation.
    2. `opgt`: Old Pseudo-Ground Truth: they are the better ground truth among static datasets
       (those called gt), and the dynamic datasets (maintained in the callbacks), and will serve
       as the labels for training the network.
    3. `pd`: Predicted Results: they are from the network outputs and will be optimized later.
       After being optimized, they will be called as `res`(Results from optimization).
    4. `bpgt`: Better Pseudo Ground Truth: they are the optimized results stored in extra file
       and in the memory. These data are the highest quality data picked among the static
       ground truth, or cached better pseudo ground truth, or the predicted & optimized data.
    '''

    # TODO: Now I only consider to use kp2d to evaluate the performance. (Because not all data provide kp3d.)
    # TODO: But we need to consider the kp3d in the future, which is if we have, than use it.

    def __init__(
        self,
        cfg     : DictConfig,
        skelify : DictConfig,
        **kwargs,
    ):
        super().__init__()
        self.interval = cfg.interval
        self.B = cfg.batch_size
        self.kb_pr = cfg.get('max_batches_per_round', None)  # latest k batches per round are SPINed
        self.better_pgt_fn = Path(cfg.better_pgt_fn)  # load it before training
        self.skip_warm_up_steps = cfg.skip_warm_up_steps
        self.update_better_pgt = cfg.update_better_pgt
        self.skelify_cfg = skelify

        # The threshold to determine if the result is valid. (In case some data
        # don't have parameters at first but was updated to a bad parameters.)
        self.valid_betas_threshold = cfg.valid_betas_threshold

        self._init_pd_dict()

        self.better_pgt = None


    def on_train_batch_start(self, trainer, pl_module, raw_batch, batch_idx):
        # Lazy initialization for better pgt.
        if self.better_pgt is None:
            self._init_better_pgt()

        # GPU_monitor.snapshot('GPU-Mem-Before-Train-Before-SPIN-Update')
        device = pl_module.device
        batch = raw_batch['img_ds']

        if not self.update_better_pgt:
            return

        # 1. Compose the data from batches.
        seq_key_list = batch['__key__']
        batch_do_flip_list = batch['augm_args']['do_flip']
        sample_uid_list = [
                f'{seq_key}_flip' if do_flip else f'{seq_key}_orig'
                for seq_key, do_flip in zip(seq_key_list, batch_do_flip_list)
            ]

        # 2. Update the labels from better_pgt.
        for i, sample_uid in enumerate(sample_uid_list):
            if sample_uid in self.better_pgt['poses'].keys():
                batch['raw_skel_params']['poses'][i] = to_tensor(self.better_pgt['poses'][sample_uid], device=device)  # (46,)
                batch['raw_skel_params']['betas'][i] = to_tensor(self.better_pgt['betas'][sample_uid], device=device)  # (10,)
                batch['has_skel_params']['poses'][i] = self.better_pgt['has_poses'][sample_uid]  # 0 or 1
                batch['has_skel_params']['betas'][i] = self.better_pgt['has_betas'][sample_uid]  # 0 or 1
                batch['updated_by_spin'][i] = True  # add information for inspection
                # get_logger().trace(f'Update the pseudo-gt for {sample_uid}.')

        # GPU_monitor.snapshot('GPU-Mem-Before-Train-After-SPIN-Update')


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # GPU_monitor.snapshot('GPU-Mem-After-Train-Before-SPIN-Update')

        # Since the prediction from network might be far from the ground truth before well-trained,
        # we can skip some steps to avoid meaningless optimization.
        if trainer.global_step > self.skip_warm_up_steps or DEBUG_ROUND:
            # Collect the prediction results.
            self._save_pd(batch['img_ds'], outputs)

            if self.interval > 0 and trainer.global_step % self.interval == 0 or DEBUG_ROUND:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                with PM.time_monitor('SPIN'):
                    self._spin(trainer.logger, pl_module.device)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # GPU_monitor.snapshot('GPU-Mem-After-Train-After-SPIN-Update')
        # GPU_monitor.report_latest(k=4)


    def _init_pd_dict(self):
        ''' Memory clean up for each SPIN. '''
        ''' Use numpy to store the value to save GPU memory. '''
        self.cache = {
                # Things to identify one sample.
                'seq_key_list' : [],
                # Things for comparison.
                'opgt_poses_list'     : [],
                'opgt_betas_list'     : [],
                'opgt_has_poses_list' : [],
                'opgt_has_betas_list' : [],
                # Things for optimization and self-iteration.
                'gt_kp2d_list'  : [],
                'pd_poses_list' : [],
                'pd_betas_list' : [],
                'pd_cam_t_list' : [],
                'do_flip_list'  : [],
                'rot_deg_list'  : [],
                'do_extreme_crop_list': [],  # if the extreme crop is applied, we don't update the pseudo-gt
                # Things for visualization.
                'img_patch': [],
                # No gt_cam_t_list.
            }


    def _format_pd(self):
        ''' Format the cache to numpy. '''
        if self.kb_pr is None:
            last_k = len(self.cache['seq_key_list'])
        else:
            last_k = self.kb_pr * self.B  # the latest k samples to be optimized.
        self.cache['seq_key_list']         = to_numpy(self.cache['seq_key_list'])[-last_k:]
        self.cache['opgt_poses_list']      = to_numpy(self.cache['opgt_poses_list'])[-last_k:]
        self.cache['opgt_betas_list']      = to_numpy(self.cache['opgt_betas_list'])[-last_k:]
        self.cache['opgt_has_poses_list']  = to_numpy(self.cache['opgt_has_poses_list'])[-last_k:]
        self.cache['opgt_has_betas_list']  = to_numpy(self.cache['opgt_has_betas_list'])[-last_k:]
        self.cache['gt_kp2d_list']         = to_numpy(self.cache['gt_kp2d_list'])[-last_k:]
        self.cache['pd_poses_list']        = to_numpy(self.cache['pd_poses_list'])[-last_k:]
        self.cache['pd_betas_list']        = to_numpy(self.cache['pd_betas_list'])[-last_k:]
        self.cache['pd_cam_t_list']        = to_numpy(self.cache['pd_cam_t_list'])[-last_k:]
        self.cache['do_flip_list']         = to_numpy(self.cache['do_flip_list'])[-last_k:]
        self.cache['rot_deg_list']         = to_numpy(self.cache['rot_deg_list'])[-last_k:]
        self.cache['do_extreme_crop_list'] = to_numpy(self.cache['do_extreme_crop_list'])[-last_k:]

        if DEBUG:
            self.cache['img_patch'] = to_numpy(self.cache['img_patch'])[-last_k:]


    def _save_pd(self, batch, outputs):
        ''' Save all the prediction results and related labels from the outputs. '''
        B = len(batch['__key__'])

        self.cache['seq_key_list'].extend(batch['__key__'])  # (NS,)

        self.cache['opgt_poses_list'].extend(to_numpy(batch['raw_skel_params']['poses']))  # (NS, 46)
        self.cache['opgt_betas_list'].extend(to_numpy(batch['raw_skel_params']['betas']))  # (NS, 10)
        self.cache['opgt_has_poses_list'].extend(to_numpy(batch['has_skel_params']['poses']))  # (NS,) 0 or 1
        self.cache['opgt_has_betas_list'].extend(to_numpy(batch['has_skel_params']['betas']))  # (NS,) 0 or 1
        self.cache['gt_kp2d_list'].extend(to_numpy(batch['kp2d']))  # (NS, 44, 3)

        self.cache['pd_poses_list'].extend(to_numpy(outputs['pd_params']['poses']))
        self.cache['pd_betas_list'].extend(to_numpy(outputs['pd_params']['betas']))
        self.cache['pd_cam_t_list'].extend(to_numpy(outputs['pd_cam_t']))
        self.cache['do_flip_list'].extend(to_numpy(batch['augm_args']['do_flip']))
        self.cache['rot_deg_list'].extend(to_numpy(batch['augm_args']['rot_deg']))
        self.cache['do_extreme_crop_list'].extend(to_numpy(batch['augm_args']['do_extreme_crop']))

        if DEBUG:
            img_patch = batch['img_patch'].clone().permute(0, 2, 3, 1)  # (NS, 256, 256, 3)
            mean = torch.tensor([0.485, 0.456, 0.406], device=img_patch.device).reshape(1, 1, 1, 3)
            std = torch.tensor([0.229, 0.224, 0.225], device=img_patch.device).reshape(1, 1, 1, 3)
            img_patch = 255 * (img_patch * std + mean)
            self.cache['img_patch'].extend(to_numpy(img_patch).astype(np.uint8))  # (NS, 256, 256, 3)


    def _init_better_pgt(self):
        ''' DDP adaptable initialization. '''
        self.rank = _get_rank()
        get_logger().info(f'Initializing better pgt cache @ rank {self.rank}')

        if self.rank is not None:
            self.better_pgt_fn = Path(f'{self.better_pgt_fn}_r{self.rank}')
            get_logger().info(f'Redirecting better pgt cache to {self.better_pgt_fn}')

        if self.better_pgt_fn.exists():
            better_pgt_z = np.load(self.better_pgt_fn, allow_pickle=True)
            self.better_pgt = {k: better_pgt_z[k].item() for k in better_pgt_z.files}
        else:
            self.better_pgt = {'poses': {}, 'betas': {}, 'has_poses': {}, 'has_betas': {}}


    def _spin(self, tb_logger, device):
        skelify : SKELify = instantiate(self.skelify_cfg, tb_logger=tb_logger, device=device, _recursive_=False)
        skel_model = skelify.skel_model

        self._format_pd()

        # 1. Make up the cache to run SKELify.
        with PM.time_monitor('preparation'):
            sample_uid_list = [
                    f'{seq_key}_flip' if do_flip else f'{seq_key}_orig'
                    for seq_key, do_flip in zip(self.cache['seq_key_list'], self.cache['do_flip_list'])
                ]

            all_gt_kp2d    = self.cache['gt_kp2d_list']   # (NS, 44, 2)
            all_init_poses = self.cache['pd_poses_list']  # (NS, 46)
            all_init_betas = self.cache['pd_betas_list']  # (NS, 10)
            all_init_cam_t = self.cache['pd_cam_t_list']  # (NS, 3)
            all_do_extreme_crop = self.cache['do_extreme_crop_list']  # (NS,)
            all_res_poses = []
            all_res_betas = []
            all_res_cam_t = []
            all_res_kp2d_err = []  # the evaluation of the keypoints 2D error

        # 2. Run SKELify optimization here to get better results.
        with PM.time_monitor('SKELify') as tm:
            get_logger().info(f'Start to run SKELify optimization. GPU-Mem: {torch.cuda.memory_allocated() / 1e9:.2f}G.')
            n_samples = len(self.cache['seq_key_list'])
            n_round = (n_samples - 1) // self.B + 1
            get_logger().info(f'Running SKELify optimization for {n_samples} samples in {n_round} rounds.')
            for rid in range(n_round):
                sid = rid * self.B
                eid = min(sid + self.B, n_samples)

                gt_kp2d_with_conf = to_tensor(all_gt_kp2d[sid:eid], device=device)
                init_poses = to_tensor(all_init_poses[sid:eid], device=device)
                init_betas = to_tensor(all_init_betas[sid:eid], device=device)
                init_cam_t = to_tensor(all_init_cam_t[sid:eid], device=device)

                # Run the SKELify optimization.
                outputs = skelify(
                        gt_kp2d    = gt_kp2d_with_conf,
                        init_poses = init_poses,
                        init_betas = init_betas,
                        init_cam_t = init_cam_t,
                        img_patch  = self.cache['img_patch'][sid:eid] if DEBUG else None,
                    )

                # Store the results.
                all_res_poses.extend(to_numpy(outputs['poses']))  # (~NS, 46)
                all_res_betas.extend(to_numpy(outputs['betas']))  # (~NS, 10)
                all_res_cam_t.extend(to_numpy(outputs['cam_t']))  # (~NS, 3)
                all_res_kp2d_err.extend(to_numpy(outputs['kp2d_err']))  # (~NS,)

                tm.tick(f'SKELify round {rid} finished.')

        # 3. Initialize the uninitialized better pseudo-gt with old ground truth.
        with PM.time_monitor('init_bpgt'):
            get_logger().info(f'Initializing bgbt. GPU-Mem: {torch.cuda.memory_allocated() / 1e9:.2f}G.')
            for i in range(n_samples):
                sample_uid = sample_uid_list[i]
                if sample_uid not in self.better_pgt.keys():
                    self.better_pgt['poses'][sample_uid] = self.cache['opgt_poses_list'][i]
                    self.better_pgt['betas'][sample_uid] = self.cache['opgt_betas_list'][i]
                    self.better_pgt['has_poses'][sample_uid] = self.cache['opgt_has_poses_list'][i]
                    self.better_pgt['has_betas'][sample_uid] = self.cache['opgt_has_betas_list'][i]

        # 4. Update the results.
        with PM.time_monitor('upd_bpgt'):
            upd_cnt = 0  # Count the number of updated samples.
            get_logger().info(f'Update the results. GPU-Mem: {torch.cuda.memory_allocated() / 1e9:.2f}G.')
            for rid in range(n_round):
                torch.cuda.empty_cache()
                sid = rid * self.B
                eid = min(sid + self.B, n_samples)

                focal_length = np.ones(2) * 5000 / 256  # TODO: These data should be loaded from configuration files.
                focal_length = focal_length.reshape(1, 2).repeat(eid - sid, 1)  # (B, 2)
                gt_kp2d_with_conf = to_tensor(all_gt_kp2d[sid:eid], device=device)  # (B, 44, 3)
                rot_deg = to_tensor(self.cache['rot_deg_list'][sid:eid], device=device)

                # 4.1. Prepare the better pseudo-gt and the results.
                res_betas  = to_tensor(all_res_betas[sid:eid], device=device)  # (B, 10)
                res_poses_after_augm  = to_tensor(all_res_poses[sid:eid], device=device)  # (B, 46)
                res_poses_before_augm = rot_skel_on_plane(res_poses_after_augm, -rot_deg)  # recover the augmentation rotation
                res_kp2d_err = to_tensor(all_res_kp2d_err[sid:eid], device=device)  # (B,)
                cur_do_extreme_crop = all_do_extreme_crop[sid:eid]

                # 4.2. Evaluate the quality of the existing better pseudo-gt.
                uids = sample_uid_list[sid:eid]  # [sid ~ eid] -> sample_uids
                bpgt_betas = to_tensor([self.better_pgt['betas'][uid] for uid in uids], device=device)
                bpgt_poses_before_augm = to_tensor([self.better_pgt['poses'][uid] for uid in uids], device=device)
                bpgt_poses_after_augm = rot_skel_on_plane(bpgt_poses_before_augm.clone(), rot_deg)  # recover the augmentation rotation

                skel_outputs = skel_model(poses=bpgt_poses_after_augm, betas=bpgt_betas, skelmesh=False)
                bpgt_kp3d = skel_outputs.joints.detach()  # (B, 44, 3)
                bpgt_est_cam_t = estimate_camera_trans(
                        S            = bpgt_kp3d,
                        joints_2d    = gt_kp2d_with_conf.clone(),
                        focal_length = 5000,
                        img_size     = 256,
                    )  # estimate camera translation from inference 3D keypoints and GT 2D keypoints
                bpgt_reproj_kp2d = perspective_projection(
                        points       = to_tensor(bpgt_kp3d, device=device),
                        translation  = to_tensor(bpgt_est_cam_t, device=device),
                        focal_length = to_tensor(focal_length, device=device),
                    )
                bpgt_kp2d_err = SKELify.eval_kp2d_err(gt_kp2d_with_conf, bpgt_reproj_kp2d)  # (B, 44)

                valid_betas_mask = res_betas.abs().max(dim=-1)[0] < self.valid_betas_threshold  # (B,)
                better_mask = res_kp2d_err < bpgt_kp2d_err  # (B,)
                upd_mask = torch.logical_and(valid_betas_mask, better_mask)  # (B,)
                upd_ids = torch.arange(eid-sid, device=device)[upd_mask]  # uids -> ids

                # Update one by one.
                for upd_id in upd_ids:
                    # `uid` for dynamic dataset unique id, `id` for in-round batch data.
                    # Notes: id starts from zeros, it should be applied to [sid ~ eid] directly.
                    #        Either `all_res_poses[upd_id]` or `res_poses[upd_id - sid]` is wrong.
                    if cur_do_extreme_crop[upd_id]:
                        # Skip the extreme crop data.
                        continue
                    sample_uid = uids[upd_id]
                    self.better_pgt['poses'][sample_uid] = to_numpy(res_poses_before_augm[upd_id])
                    self.better_pgt['betas'][sample_uid] = to_numpy(res_betas[upd_id])
                    self.better_pgt['has_poses'][sample_uid] = 1.  # If updated, then must have.
                    self.better_pgt['has_betas'][sample_uid] = 1.  # If updated, then must have.
                    upd_cnt += 1

            get_logger().info(f'Update {upd_cnt} samples among all {n_samples} samples.')

        # 5. [Async] Save the results.
        with PM.time_monitor('async_dumping'):
            # TODO: Use lock and other techniques to achieve a better submission system.
            # TODO: We need to design a better way to solve the synchronization problem.
            if hasattr(self, 'dump_thread'):
                self.dump_thread.result()  # Wait for the previous dump to finish.
            with ThreadPoolExecutor() as executor:
                self.dump_thread = executor.submit(lambda: np.savez(self.better_pgt_fn, **self.better_pgt))

        # 5. Clean up the memory.
        del skelify, skel_model
        self._init_pd_dict()