import time
from lib.kits.basic import *
from lib.utils.vis import Wis3D
from lib.utils.vis.py_renderer import render_mesh_overlay_img
from lib.utils.data import to_tensor
from lib.utils.media import draw_kp2d_on_img, annotate_img, splice_img
from lib.utils.camera import perspective_projection
from lib.body_models.abstract_skeletons import Skeleton_OpenPose25
from lib.modeling.losses import *
from lib.modeling.networks.discriminators import HSMRDiscriminator
from lib.platform.config_utils import get_PM_info_dict
from lib.modeling.networks.backbones import mamba2

def build_inference_pipeline( 
    model_root: Union[Path, str],
    ckpt_fn   : Optional[Union[Path, str]] = None,
    tuned_bcb : bool = True,
    device    : str = 'cuda',
):
    # 1.1. Load the config file.
    if isinstance(model_root, str):
        model_root = Path(model_root)
    cfg_path = model_root / '.hydra' / 'config.yaml'   
    cfg = OmegaConf.load(cfg_path)
    
    # 1.2. Override PM info dict. 
    PM_overrides = get_PM_info_dict()._pm_
    cfg._pm_ = PM_overrides
    get_logger(brief=True).info(f'Building inference pipeline of {cfg.exp_name}')

    # 2.1. Instantiate the pipeline.  
    init_bcb = not tuned_bcb
    pipeline = instantiate(cfg.pipeline, init_backbone=init_bcb, _recursive_=False)
    pipeline = pipeline.to(device) 
    pipeline.set_data_adaption(data_module_name='IMG_PATCHES')
    print(pipeline)
    print(cfg.pipeline)

    # 2.2. Load the checkpoint. 
    if ckpt_fn is None:
        ckpt_fn = model_root / 'checkpoints' / ''  # to modify
    pipeline.load_state_dict(torch.load(ckpt_fn, map_location=device)['state_dict'])
    get_logger(brief=True).info(f'Load checkpoint from {ckpt_fn}.')

    pipeline.train()  #train&eval
    return pipeline.to(device)

class MambaPipeline(pl.LightningModule):

    def __init__(self, cfg:DictConfig, name:str, init_backbone=True): 
        super(MambaPipeline, self).__init__()
        self.name = name
        #self.skel_model = instantiate(cfg.SKEL)
        self.skel_model = instantiate(cfg.SKEL)
        self.backbone = instantiate(cfg.backbone)
        self.head = instantiate(cfg.head)
        self.cfg = cfg

        if init_backbone:
            # For inference mode with tuned backbone checkpoints, we don't need to initialize the backbone here.
            self._init_backbone()
        
        # Loss layers.
        self.kp_3d_loss  = Keypoint3DLoss(loss_type='l1')
        self.kp_2d_loss  = Keypoint2DLoss(loss_type='l1')
        self.params_loss = ParameterLoss()

        # Discriminator. 
        self.enable_disc = self.cfg.loss_weights.get('adversarial', 0) > 0
        if self.enable_disc:
            self.discriminator = HSMRDiscriminator()
            get_logger().warning(f'Discriminator enabled, the global_steps will be doubled. Use the checkpoints carefully.')
        else:
            self.discriminator = None
            self.cfg.loss_weights.pop('adversarial', None)  # pop the adversarial term if not enabled

        # Manually control the optimization since we have an adversarial process.
        self.automatic_optimization = False
        self.set_data_adaption()

        # For visualization debug.
        if False:
            self.wis3d = Wis3D(seq_name=PM.cfg.exp_name)
        else:
            self.wis3d = None

    def set_data_adaption(self, data_module_name:Optional[str]=None):  # Adapt image patches and skeleton data
        if data_module_name is None:
            # get_logger().warning('Data adapter schema is not defined. The input will be regarded as image patches.')
            self.adapt_batch = self._adapt_img_inference
        elif data_module_name == 'IMG_PATCHES':
            self.adapt_batch = self._adapt_img_inference
        elif data_module_name.startswith('SKEL_HSMR_V1'):
            self.adapt_batch = self._adapt_actionhsmr_v1
        else:
            raise ValueError(f'Unknown data module: {data_module_name}')
        print(f"adapt_batch 指向: {self.adapt_batch.__name__}")


    def print_summary(self, max_depth=1): 
        from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary
        print(ModelSummary(self, max_depth=max_depth))


    # return optimizers
    def configure_optimizers(self): 
        optimizers = []
    
        params_main_all = list(self._params_main())
        params_main = [p for p in params_main_all if p.requires_grad]

        optimizer_main = instantiate(self.cfg.optimizer, params=params_main)
        optimizers.append(optimizer_main)

        params_disc_all = list(self._params_disc())
        if len(params_disc_all) > 0:
            params_disc = [p for p in params_disc_all if p.requires_grad]  
            optimizer_disc = instantiate(self.cfg.optimizer, params=params_disc)
            optimizers.append(optimizer_disc)
        return optimizers

    def training_step(self, raw_batch, batch_idx):
        with PM.time_monitor('training_step'):
            return self._training_step(raw_batch, batch_idx)

    def _training_step(self, raw_batch, batch_idx):
        # GPU_monitor = GPUMonitor()
        # GPU_monitor.snapshot('HSMR training start')
        t0 = time.time()
        
        t1 = time.time()
        batch = self.adapt_batch(raw_batch['img_ds'])
        t2 = time.time()
        # GPU_monitor.snapshot('HSMR adapt batch')

        optimizers = self.optimizers(use_pl_optimizer=True)
        if isinstance(optimizers, List):
            optimizer_main, optimizer_disc = optimizers
        else:
            optimizer_main = optimizers
        t3 = time.time()
        # GPU_monitor.snapshot('HSMR get optimizer')

        with PM.time_monitor('forward_step'):
            img_patch = to_tensor(batch['img_patch'], self.device)  # (B, T, C, H, W)
            B = len(img_patch)
            outputs = self.forward_step(img_patch)  # {...}
            # GPU_monitor.snapshot('HSMR forward')
            pd_skel_params = MambaPipeline._adapt_skel_params(outputs['pd_params'])
            t4 = time.time()
            # GPU_monitor.snapshot('HSMR adapt SKEL params')

        # 2. [Optional] Discriminator forward pass in main training step.
        if self.enable_disc:
            with PM.time_monitor('disc_forward'):
                pd_poses_mat, _ = self.skel_model.pose_params_to_rot(pd_skel_params['poses'])  # (B, J=24, 3, 3)
                pd_poses_body_mat = pd_poses_mat[:, 1:, :, :]  # (B, J=23, 3, 3)
                pd_betas = pd_skel_params['betas']  # (B, 10)
                disc_out = self.discriminator(
                        poses_body = pd_poses_body_mat,   # (B, J=23, 3, 3)
                        betas      = pd_betas,            # (B, 10)
                    )
        else:
            disc_out = None
        t5 = time.time()

       # 3. Prepare the secondary products
        with PM.time_monitor('Secondary Products Preparation'):
            # 3.1. Body model outputs.
            with PM.time_monitor('SKEL Forward'):
                skel_outputs = self.skel_model(**pd_skel_params, skelmesh=False)
                pd_kp3d = skel_outputs.joints      # (B, Q=44, 3)
                pd_skin = skel_outputs.skin_verts  # (B, V=6890, 3)
            # 3.2. Reproject the 3D joints to 2D plain.
            with PM.time_monitor('Reprojection'): 
                pd_kp2d = perspective_projection(
                        points       = pd_kp3d,  # (B, K=Q=44, 3)
                        translation  = outputs['pd_cam_t'],  # (B, 3)
                        focal_length = outputs['focal_length'] / self.cfg.policy.img_patch_size,  # (B, 2)
                    )  # (B, 44, 2)
            # 3.3. Extract G.T. from inputs.
            gt_kp2d_with_conf = batch['kp2d'].clone()  # (B, 44, 3)
            gt_kp3d_with_conf = batch['kp3d'].clone()  # (B, 44, 4)
            # 3.4. Extract G.T. skin mesh only for visualization.
            gt_skel_params = MambaPipeline._adapt_skel_params(batch['gt_params'])  # {poses, betas}
            gt_skel_params = {k: v[:self.cfg.logger.samples_per_record] for k, v in gt_skel_params.items()}
            skel_outputs = self.skel_model(**gt_skel_params, skelmesh=False)
            gt_skin = skel_outputs.skin_verts  # (B', V=6890, 3)
            gt_valid_body = batch['has_gt_params']['poses_body'][:self.cfg.logger.samples_per_record]  # {poses_orient, poses_body, betas}
            gt_valid_orient = batch['has_gt_params']['poses_orient'][:self.cfg.logger.samples_per_record]  # {poses_orient, poses_body, betas}
            gt_valid_betas = batch['has_gt_params']['betas'][:self.cfg.logger.samples_per_record]  # {poses_orient, poses_body, betas}
            gt_valid = torch.logical_and(torch.logical_and(gt_valid_body, gt_valid_orient), gt_valid_betas)
        # GPU_monitor.snapshot('HSMR secondary products')

        t6 = time.time()

        # 4. Compute losses.
        with PM.time_monitor('Compute Loss'):
            
            loss_main, losses_main = self._compute_losses_main(
                    self.cfg.loss_weights,
                    pd_kp3d,  # (B, 44, 3)
                    gt_kp3d_with_conf,  # (B, 44, 4)
                    pd_kp2d,  # (B, 44, 2)
                    gt_kp2d_with_conf,  # (B, 44, 3)
                    outputs['pd_params'],  # {'poses_orient':..., 'poses_body':..., 'betas':...}
                    batch['gt_params'],  # {'poses_orient':..., 'poses_body':..., 'betas':...}
                    batch['has_gt_params'],
                    disc_out,
                )
        # GPU_monitor.snapshot('HSMR compute losses')
        t7 = time.time()
        if torch.isnan(loss_main):
            get_logger().error(f'NaN detected in loss computation. Losses: {losses}')

        # 5. Main parts backward pass.
        with PM.time_monitor('Backward Step'):
            optimizer_main.zero_grad()
            self.manual_backward(loss_main)
            optimizer_main.step()
            t8 = time.time()
        # GPU_monitor.snapshot('HSMR backwards')

        # 6. [Optional] Discriminator training part.
        if self.enable_disc:
            with PM.time_monitor('Train Discriminator'):
                losses_disc = self._train_discriminator(
                        mocap_batch       = raw_batch['mocap_ds'],
                        pd_poses_body_mat = pd_poses_body_mat,
                        pd_betas          = pd_betas,
                        optimizer         = optimizer_disc,
                    )
        else:
            losses_disc = {}
        t9 = time.time()

        if batch_idx == 0:
            print("img_patch shape:", img_patch.shape)
            print("outputs keys:", outputs.keys())
            if self.enable_disc:
                print("disc_out shape:", disc_out.shape)
                print("pd_kp2d shape:", pd_kp2d.shape)
                print("gt_kp2d_with_conf shape:", gt_kp2d_with_conf.shape)



        mid_frame = 1
        img_key_frame = img_patch[:self.cfg.logger.samples_per_record, mid_frame] 
        # 7. Logging.
        with PM.time_monitor('Tensorboard Logging'):
            vis_data = {
                    'img_patch'         : to_numpy(img_key_frame).transpose((0, 2, 3, 1)).copy(),
                    'pd_kp2d'           : pd_kp2d[:self.cfg.logger.samples_per_record].clone(),
                    'pd_kp3d'           : pd_kp3d[:self.cfg.logger.samples_per_record].clone(),
                    'gt_kp2d_with_conf' : gt_kp2d_with_conf[:self.cfg.logger.samples_per_record].clone(),
                    'gt_kp3d_with_conf' : gt_kp3d_with_conf[:self.cfg.logger.samples_per_record].clone(),
                    'pd_skin'           : pd_skin[:self.cfg.logger.samples_per_record].clone(),
                    'gt_skin'           : gt_skin.clone(),
                    'gt_skin_valid'     : gt_valid,
                    'cam_t'             : outputs['pd_cam_t'][:self.cfg.logger.samples_per_record].clone(),
                    'img_key'           : batch['__key__'][:self.cfg.logger.samples_per_record],
                }
            import numpy as np
            for k, v in vis_data.items():
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                if isinstance(v, np.ndarray):
                    if np.isinf(v).any():
                        print(f"[VisData] {k} contains inf at", np.where(np.isinf(v)))
                    if np.isnan(v).any():
                        print(f"[VisData] {k} contains nan at", np.where(np.isnan(v)))

            self._tb_log(losses_main=losses_main, losses_disc=losses_disc, vis_data=vis_data)
        # GPU_monitor.snapshot('HSMR logging')
        self.log('_/loss_main', losses_main['weighted_sum'], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
        t10 = time.time()

        if batch_idx % 10 == 0: 
            with open("training_times.log", "a") as f:
                f.write(f"\nBatch {batch_idx} Time Points (seconds):\n")
                f.write(f"  Total: {t10-t0:.4f}\n")
                f.write(f"  1. Data adapt: {t2-t1:.4f}\n")
                f.write(f"  2. Optim setup: {t3-t2:.4f}\n")
                f.write(f"  3. Forward: {t4-t3:.4f}\n")
                f.write(f"  4. Disc forward: {t5-t4:.4f}\n")
                f.write(f"  5. Secondary prep: {t6-t5:.4f}\n")
                f.write(f"  6. Loss comp: {t7-t6:.4f}\n")
                f.write(f"  7. Backward: {t8-t7:.4f}\n")
                f.write(f"  8. Disc train: {t9-t8:.4f}\n")
                f.write(f"  9. Logging: {t10-t9:.4f}\n")
                f.write("="*50 + "\n")
                
        # GPU_monitor.report_all()
        return outputs

    def forward(self, batch): 
        '''
        ### Returns
        - outputs: Dict
            - pd_kp3d: torch.Tensor, shape (B, Q=44, 3)
            - pd_kp2d: torch.Tensor, shape (B, Q=44, 2)
            - pred_keypoints_2d: torch.Tensor, shape (B, Q=44, 2)
            - pred_keypoints_3d: torch.Tensor, shape (B, Q=44, 3)
            - pd_params: Dict
                - poses: torch.Tensor, shape (B, 46)
                - betas: torch.Tensor, shape (B, 10)
            - pd_cam: torch.Tensor, shape (B, 3)
            - pd_cam_t: torch.Tensor, shape (B, 3)
            - focal_length: torch.Tensor, shape (B, 2)
        '''
        batch = self.adapt_batch(batch)

        # 1. Main parts forward pass.
        img_patch = to_tensor(batch['img_patch'], self.device)  # (B, C, H, W)
        outputs = self.forward_step(img_patch)  # {...}

        # 2. Prepare the secondary products
        # 2.1. Body model outputs.
        pd_skel_params = MambaPipeline._adapt_skel_params(outputs['pd_params'])
        pd_skel_params = {k: v.to(self.device) for k, v in pd_skel_params.items()}
        #skel_outputs = self.skel_model(**pd_skel_params, skelmesh=False)
        skel_outputs = self.skel_model(**pd_skel_params, skelmesh=False)  
        pd_kp3d = skel_outputs.joints  # (B, Q=44, 3)
        pd_skin_verts = skel_outputs.skin_verts.detach().cpu().clone()  # (B, V=6890, 3)
        # 2.2. Reproject the 3D joints to 2D plain.
        pd_kp2d = perspective_projection(  
                points       = to_tensor(pd_kp3d, device=self.device),  # (B, K=Q=44, 3)
                translation  = to_tensor(outputs['pd_cam_t'], device=self.device),  # (B, 3)
                focal_length = to_tensor(outputs['focal_length'], device=self.device) / self.cfg.policy.img_patch_size,  # (B, 2)
            )

        outputs['pd_kp3d'] = pd_kp3d
        outputs['pd_kp2d'] = pd_kp2d
        outputs['pred_keypoints_2d'] = pd_kp2d  # adapt HMR2.0's script
        outputs['pred_keypoints_3d'] = pd_kp3d  # adapt HMR2.0's script
        outputs['pd_params'] = pd_skel_params
        outputs['pd_skin_verts'] = pd_skin_verts.cpu() 

        return outputs

    def forward_step(self, x: torch.Tensor):
        """
        Args
        ----
        x : (B, T, C, H, W)   # actually(B, 3, 3, 256, 256)
        
        Returns
        -------
        dict consists of 'pd_cam', 'pd_params', 'focal_length'
        """
        B, T, C, H, W = x.shape
        assert T == 3 and C == 3, f"Expect (B,3,3,H,W), got {x.shape}"
        x = x.to(self.device, non_blocking=True)
        
    # 1. 256 → 192
        x = x[..., :, 32:-32]  # (B,3,3,256,192)
        B, T, C, H, W = x.shape    # H=256, W=192
        #print("after crop", x.shape)   # (B,3,3,256,192)

    # 2. backbone（mamba2）
        feat = self.backbone(x) 

    # 3. Run the head to predict the body model parameters.
        with PM.time_monitor('Predict Head Forward'):
            pd_params, pd_cam = self.head(feat)
        # GPU_monitor.snapshot('HSMR forward head')

   # 4. Transform the camera parameters to camera translation.
        focal_length = self.cfg.policy.focal_length * torch.ones(B, 2, device=self.device, dtype=pd_cam.dtype)  # (B, 2)
        pd_cam_t = torch.stack([
                    pd_cam[:, 1],
                    pd_cam[:, 2],
                    2 * focal_length[:, 0] / (self.cfg.policy.img_patch_size * pd_cam[:, 0] + 1e-9)
                ], dim=-1)  # (B, 3)

    # 5. Store the results.
        outputs = {
                'pd_cam'       : pd_cam,
                'pd_cam_t'     : pd_cam_t,
                'pd_params'    : pd_params,
                # 'pd_params'    : {k: v.clone() for k, v in pd_params.items()},
                'focal_length' : focal_length,  # (B, 2)
            }
        # GPU_monitor.report_all()
        return outputs


    # ========== Internal Functions ==========

    def _params_main(self): 
        return list(self.head.parameters()) + list(self.backbone.parameters())

    def _params_disc(self):   
        if self.discriminator is None:
            return []
        else:
            return list(self.discriminator.parameters())

    @staticmethod
    def _adapt_skel_params(params:Dict):  
        ''' Change the parameters formed like [pose_orient, pose_body, betas, trans] to [poses, betas, trans]. '''
        adapted_params = {}

        if 'poses' in params.keys():
            adapted_params['poses'] = params['poses']
        elif 'poses_orient' in params.keys() and 'poses_body' in params.keys():
            poses_orient = params['poses_orient']  # (B, 3)
            poses_body = params['poses_body']  # (B, 43)
            adapted_params['poses'] = torch.cat([poses_orient, poses_body], dim=1)  # (B, 46)
        else:
            raise ValueError(f'Cannot find the poses parameters among {list(params.keys())}.')

        if 'betas' in params.keys():
            adapted_params['betas'] = params['betas']  # (B, 10)
        else:
            raise ValueError(f'Cannot find the betas parameters among {list(params.keys())}.')

        return adapted_params

    def _init_backbone(self):
        if self.cfg.backbone_ckpt:
            get_logger().info(f'Loading backbone weights from {self.cfg.backbone_ckpt}')
            state_dict = torch.load(self.cfg.backbone_ckpt, map_location='cpu')['state_dict']
            
            if 'backbone.cls_token' in state_dict.keys():
                state_dict = {k: v for k, v in state_dict.items() if 'backbone' in k and 'cls_token' not in k}
                state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            
            if len(missing) > 0:
                get_logger().warning(f'Missing keys in backbone: {missing}')
            if len(unexpected) > 0:
                get_logger().warning(f'Unexpected keys in backbone: {unexpected}')
        else:
            get_logger().info('Mamba backbone: No weights provided, using random initialization')
        
        if self.cfg.get('freeze_backbone', False):
            self.backbone.eval()
            self.backbone.requires_grad_(False)


    def _compute_losses_main(  
        self,
        loss_weights : Dict,
        pd_kp3d      : torch.Tensor,
        gt_kp3d      : torch.Tensor,
        pd_kp2d      : torch.Tensor,
        gt_kp2d      : torch.Tensor,
        pd_params    : Dict,
        gt_params    : Dict,
        has_params   : Dict,
        disc_out     : Optional[torch.Tensor]=None,
        *args, **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        ''' Compute the weighted losses according to the config file. '''

        # 1. Preparation.
        with PM.time_monitor('Preparation'):
            B = len(pd_kp3d)
            gt_skel_params = MambaPipeline._adapt_skel_params(gt_params)  # {poses, betas}
            pd_skel_params = MambaPipeline._adapt_skel_params(pd_params)  # {poses, betas}

            gt_betas = gt_skel_params['betas'].reshape(-1, 10)
            pd_betas = pd_skel_params['betas'].reshape(-1, 10)
            gt_poses = gt_skel_params['poses'].reshape(-1, 46)
            pd_poses = pd_skel_params['poses'].reshape(-1, 46)

        # 2. Keypoints losses.
        with PM.time_monitor('kp2d & kp3d Loss'):
            kp2d_loss = self.kp_2d_loss(pd_kp2d, gt_kp2d) / B
            kp3d_loss = self.kp_3d_loss(pd_kp3d, gt_kp3d) / B

        # 3. Prior losses.
        with PM.time_monitor('Prior Loss'):
            prior_loss = compute_poses_angle_prior_loss(pd_poses).mean()  # (,)

        # 4. Parameters losses.
        if self.cfg.sp_poses_repr == 'rotation_matrix':
            with PM.time_monitor('q2mat'):
                gt_poses_mat, _ = self.skel_model.pose_params_to_rot(gt_poses)  # (B, J=24, 3, 3)
                pd_poses_mat, _ = self.skel_model.pose_params_to_rot(pd_poses)  # (B, J=24, 3, 3)

                gt_poses = gt_poses_mat.reshape(-1, 24*3*3)  # (B, 24*3*3)
                pd_poses = pd_poses_mat.reshape(-1, 24*3*3)  # (B, 24*3*3)

        with PM.time_monitor('Parameters Loss'):
            poses_orient_loss = self.params_loss(pd_poses[:, :9], gt_poses[:, :9], has_params['poses_orient']) / B
            poses_body_loss   = self.params_loss(pd_poses[:, 9:], gt_poses[:, 9:], has_params['poses_body']) / B
            betas_loss        = self.params_loss(pd_betas, gt_betas, has_params['betas']) / B

        # 5. Collect main losses.
        with PM.time_monitor('Accumulate'):
            losses = {
                    'kp3d'         : kp3d_loss,          # (,)
                    'kp2d'         : kp2d_loss,          # (,)
                    'prior'        : prior_loss,         # (,)
                    'poses_orient' : poses_orient_loss,  # (,)
                    'poses_body'   : poses_body_loss,    # (,)
                    'betas'        : betas_loss,         # (,)
                }

        # 6. Consider adversarial loss.
        if disc_out is not None:
            with PM.time_monitor('Adversarial Loss'):
                adversarial_loss = ((disc_out - 1.0) ** 2).sum() / B  # (,)
                losses['adversarial'] = adversarial_loss

        with PM.time_monitor('Accumulate'):
            loss = torch.tensor(0., device=self.device)
            for k, v in losses.items():
                loss += v * loss_weights[k]
            losses = {k: v.item() for k, v in losses.items()}
            losses['weighted_sum'] = loss.item()
        return loss, losses

    def _train_discriminator(self, mocap_batch, pd_poses_body_mat, pd_betas, optimizer): 
        '''
        Train the discriminator using the regressed body model parameters and the realistic MoCap data.

        ### Args
        - mocap_batch: Dict
            - 'poses_body': torch.Tensor, shape (B, 43)
            - 'betas': torch.Tensor, shape (B, 10)
        - pd_poses_body_mat: torch.Tensor, shape (B, J=23, 3, 3)
        - pd_betas: torch.Tensor, shape (B, 10)
        - optimizer: torch.optim.Optimizer

        ### Returns
        - losses: Dict
            - 'pd_disc': float
            - 'mc_disc': float
        '''
        pd_B = len(pd_poses_body_mat)
        mc_B = len(mocap_batch['poses_body'])
        get_logger().warning(f'pd_B: {pd_B} != mc_B: {mc_B}')

        # 1. Extract the realistic 3D MoCap label.
        mc_poses_body = mocap_batch['poses_body']  # (B, 43)
        padding_zeros = mc_poses_body.new_zeros(mc_B, 3)  # (B, 3)
        mc_poses = torch.cat([padding_zeros, mc_poses_body], dim=1)  # (B, 46)
        mc_poses_mat, _ = self.skel_model.pose_params_to_rot(mc_poses)  # (B, J=24, 3, 3)
        mc_poses_body_mat = mc_poses_mat[:, 1:, :, :]  # (B, J=23, 3, 3)
        mc_betas = mocap_batch['betas']  # (B, 10)

        # 2. Forward pass.
        # Discriminator forward pass for the predicted data.
        pd_disc_out = self.discriminator(pd_poses_body_mat.detach(), pd_betas.detach())
        pd_disc_loss = ((pd_disc_out - 0.0) ** 2).sum() / pd_B  # (,)
        # Discriminator forward pass for the realistic MoCap data.
        mc_disc_out = self.discriminator(mc_poses_body_mat, mc_betas)
        mc_disc_loss = ((mc_disc_out - 1.0) ** 2).sum() / pd_B  # (,)  TODO: This 'pd_B' is from HMR2, not sure if it's a bug.

        # 3. Backward pass.
        disc_loss = self.cfg.loss_weights.adversarial * (pd_disc_loss + mc_disc_loss)
        optimizer.zero_grad()
        self.manual_backward(disc_loss)
        optimizer.step()

        return {
                'pd_disc': pd_disc_loss.item(),
                'mc_disc': mc_disc_loss.item(),
            }

    import numpy as np

    def check_array(arr, name, img_shape=None):
        arr = np.asarray(arr)
        if np.isnan(arr).any():
            print(f"[Check] {name} contains NaN at {np.where(np.isnan(arr))}, value: {arr[np.where(np.isnan(arr))]}")
        if np.isinf(arr).any():
            print(f"[Check] {name} contains Inf at {np.where(np.isinf(arr))}, value: {arr[np.where(np.isinf(arr))]}")
        if img_shape is not None and arr.shape[-1] >= 2:
            h, w = img_shape[:2]
        # 检查越界
            out_x = np.where((arr[...,0] < 0) | (arr[...,0] >= w))
            out_y = np.where((arr[...,1] < 0) | (arr[...,1] >= h))
            if len(out_x[0]) > 0 or len(out_y[0]) > 0:
                print(f"[Check] {name} out-of-bounds:")
                print(f"  x: idx={out_x}, value={arr[...,0][out_x]}")
                print(f"  y: idx={out_y}, value={arr[...,1][out_y]}")
        if arr.dtype not in [np.float32, np.float64, np.int32, np.int64, np.uint8]:
            print(f"[Check] {name} dtype: {arr.dtype}")

    @rank_zero_only
    def _tb_log(self, losses_main:Dict, losses_disc:Dict, vis_data:Dict, mode:str='train'):
        ''' Write the logging information to the TensorBoard. '''
        if self.logger is None:
            return

        if self.global_step != 1 and self.global_step % self.cfg.logger.interval != 0:
            return

        # 1. Losses.
        summary_writer = self.logger.experiment
        for loss_name, loss_val in losses_main.items():
            summary_writer.add_scalar(f'{mode}/losses_main/{loss_name}', loss_val, self.global_step)
        for loss_name, loss_val in losses_disc.items():
            summary_writer.add_scalar(f'{mode}/losses_disc/{loss_name}', loss_val, self.global_step)

        # 2. Visualization.
        try:
            pelvis_id = 39
            # 2.1. Visualize 3D information.
            self.wis3d.add_motion_mesh(
                verts = vis_data['pd_skin'] - vis_data['pd_kp3d'][:, pelvis_id:pelvis_id+1],  # center the mesh
                faces = self.skel_model.skin_f,
                name  = 'pd_skin',
            )
            self.wis3d.add_motion_mesh3d.add_motion_mesh(
                verts = vis_data['gt_skin'] - vis_data['gt_kp3d_with_conf'][:, pelvis_id:pelvis_id+1, :3],  # center the mesh
                faces = self.skel_model.skin_f,
                name  = 'gt_skin',
            )
            self.wis3d.add_motion_skel(
                joints = vis_data['pd_kp3d'] - vis_data['pd_kp3d'][:, pelvis_id:pelvis_id+1],
                bones  = Skeleton_OpenPose25.bones,
                colors = Skeleton_OpenPose25.bone_colors,
                name   = 'pd_kp3d',
            )

            aligned_gt_kp3d = vis_data['gt_kp3d_with_conf']
            aligned_gt_kp3d[..., :3] -= vis_data['gt_kp3d_with_conf'][:, pelvis_id:pelvis_id+1, :3]
            self.wis3d.add_motion_skel(
                joints = aligned_gt_kp3d,
                bones  = Skeleton_OpenPose25.bones,
                colors = Skeleton_OpenPose25.bone_colors,
                name   = 'gt_kp3d',
            )
        except Exception as e:
            if self.wis3d is not None:
                get_logger().error(f'Failed to visualize the current performance on wis3d: {e}')

        try:
            # 2.2. Visualize 2D information.
            if vis_data['img_patch'] is not None:
                # Overlay the skin mesh of the results on the original image.
                imgs_spliced = []
                for i, img_patch in enumerate(vis_data['img_patch']):
                    # TODO: make this more elegant.
                    img_mean = to_numpy(OmegaConf.to_container(self.cfg.policy.img_mean))[None, None]  # (1, 1, 3)
                    img_std = to_numpy(OmegaConf.to_container(self.cfg.policy.img_std))[None, None]  # (1, 1, 3)
                    img_patch = ((img_mean + img_patch * img_std) * 255).astype(np.uint8)

                    h, w = img_patch.shape[:2] 

                    gt_kp2d_with_conf = vis_data['gt_kp2d_with_conf'][i].clone() 
                    gt_kp2d_with_conf = gt_kp2d_with_conf.cpu().numpy()
                    print(f"\n[check] gt_kp2d_with_conf[{i}] 处理前 shape:{gt_kp2d_with_conf.shape} dtype:{gt_kp2d_with_conf.dtype}")
                    print("nan:", np.isnan(gt_kp2d_with_conf).any(), "inf:", np.isinf(gt_kp2d_with_conf).any(),
                          "min:", np.nanmin(gt_kp2d_with_conf), "max:", np.nanmax(gt_kp2d_with_conf))
                    
                    gt_kp2d_with_conf[:, :2] = (gt_kp2d_with_conf[:, :2] + 0.5) * self.cfg.policy.img_patch_size
                    print(f"[check] gt_kp2d_with_conf[{i}] 转换后 shape:{gt_kp2d_with_conf.shape} dtype:{gt_kp2d_with_conf.dtype}")
                    print("nan:", np.isnan(gt_kp2d_with_conf).any(), "inf:", np.isinf(gt_kp2d_with_conf).any(),
                          "min:", np.nanmin(gt_kp2d_with_conf), "max:", np.nanmax(gt_kp2d_with_conf))
        
                    gt_kp2d_with_conf[:, 0] = np.clip(gt_kp2d_with_conf[:, 0], 0, w-1)
                    gt_kp2d_with_conf[:, 1] = np.clip(gt_kp2d_with_conf[:, 1], 0, h-1)
                    print(f"[check] gt_kp2d_with_conf[{i}] clip后 shape:{gt_kp2d_with_conf.shape} dtype:{gt_kp2d_with_conf.dtype}")
                    print("nan:", np.isnan(gt_kp2d_with_conf).any(), "inf:", np.isinf(gt_kp2d_with_conf).any(),
                          "min:", np.nanmin(gt_kp2d_with_conf), "max:", np.nanmax(gt_kp2d_with_conf))
                    if np.any(gt_kp2d_with_conf[:, 0] < 0) or np.any(gt_kp2d_with_conf[:, 0] >= w) or np.any(gt_kp2d_with_conf[:, 1] < 0) or np.any(gt_kp2d_with_conf[:, 1] >= h):
                        print("[warning] gt_kp2d_with_conf 越界!")

                    pd_kp2d_vis = vis_data['pd_kp2d'][i].clone().detach().cpu().numpy() 
                    print(f"\n[check] pd_kp2d[{i}] 处理前 shape:{pd_kp2d_vis.shape} dtype:{pd_kp2d_vis.dtype}")
                    print("nan:", np.isnan(pd_kp2d_vis).any(), "inf:", np.isinf(pd_kp2d_vis).any(),
                          "min:", np.nanmin(pd_kp2d_vis), "max:", np.nanmax(pd_kp2d_vis))
        
                    pd_kp2d_vis = (pd_kp2d_vis + 0.5) * self.cfg.policy.img_patch_size
                    print(f"[check] pd_kp2d[{i}] 转换后 shape:{pd_kp2d_vis.shape} dtype:{pd_kp2d_vis.dtype}")
                    print("nan:", np.isnan(pd_kp2d_vis).any(), "inf:", np.isinf(pd_kp2d_vis).any(),
                          "min:", np.nanmin(pd_kp2d_vis), "max:", np.nanmax(pd_kp2d_vis))

                    pd_kp2d_vis[:, 0] = np.clip(pd_kp2d_vis[:, 0], 0, w-1)
                    pd_kp2d_vis[:, 1] = np.clip(pd_kp2d_vis[:, 1], 0, h-1)
                    print(f"[check] pd_kp2d[{i}] clip后 shape:{pd_kp2d_vis.shape} dtype:{pd_kp2d_vis.dtype}")
                    print("nan:", np.isnan(pd_kp2d_vis).any(), "inf:", np.isinf(pd_kp2d_vis).any(),
                          "min:", np.nanmin(pd_kp2d_vis), "max:", np.nanmax(pd_kp2d_vis))
                    if np.any(pd_kp2d_vis[:, 0] < 0) or np.any(pd_kp2d_vis[:, 0] >= w) or np.any(pd_kp2d_vis[:, 1] < 0) or np.any(pd_kp2d_vis[:, 1] >= h):
                        print("[warning] pd_kp2d 越界!")

                    img_patch_raw = annotate_img(img_patch, 'raw')
                    img_with_mesh = render_mesh_overlay_img(
                            faces      = self.skel_model.skin_f,
                            verts      = vis_data['pd_skin'][i].float(),
                            K4         = [self.cfg.policy.focal_length, self.cfg.policy.focal_length, 128, 128],
                            img        = img_patch,
                            Rt         = [torch.eye(3).float(), vis_data['cam_t'][i].float()],
                            mesh_color = 'pink',
                        )
                    img_with_mesh = annotate_img(img_with_mesh, 'pd_mesh')

                    img_with_gt_mesh = render_mesh_overlay_img(
                            faces      = self.skel_model.skin_f,
                            verts      = vis_data['gt_skin'][i].float(),
                            K4         = [self.cfg.policy.focal_length, self.cfg.policy.focal_length, 128, 128],
                            img        = img_patch,
                            Rt         = [torch.eye(3).float(), vis_data['cam_t'][i].float()],
                            mesh_color = 'pink',
                        )
                    valid = 'valid' if vis_data['gt_skin_valid'][i] else 'invalid'
                    img_with_gt_mesh = annotate_img(img_with_gt_mesh, f'gt_mesh_{valid}')

                    img_with_gt = annotate_img(img_patch, 'gt_kp2d')
                    gt_kp2d_with_conf = vis_data['gt_kp2d_with_conf'][i]
                    gt_kp2d_with_conf[:, :2] = (gt_kp2d_with_conf[:, :2] + 0.5) * self.cfg.policy.img_patch_size

                    img_with_gt = draw_kp2d_on_img(
                            img_with_gt,
                            gt_kp2d_with_conf,
                            Skeleton_OpenPose25.bones,
                            Skeleton_OpenPose25.bone_colors,
                        )

                    img_with_pd = annotate_img(img_patch, 'pd_kp2d')
                    pd_kp2d_vis = vis_data['pd_kp2d'][i]
                    pd_kp2d_vis = (pd_kp2d_vis + 0.5) * self.cfg.policy.img_patch_size
                    img_with_pd = draw_kp2d_on_img(
                            img_with_pd,
                            (vis_data['pd_kp2d'][i] + 0.5) * self.cfg.policy.img_patch_size,
                            Skeleton_OpenPose25.bones,
                            Skeleton_OpenPose25.bone_colors,
                        )

                    img_spliced = splice_img([img_patch_raw, img_with_gt, img_with_pd, img_with_mesh, img_with_gt_mesh], grid_ids=[[0, 1, 2, 3, 4]])
                    img_spliced = annotate_img(img_spliced, vis_data['img_key'][i], pos='tl')
                    imgs_spliced.append(img_spliced)

                    try:
                        self.wis3d.set_scene_id(i)
                        self.wis3d.add_image(
                            image = img_spliced,
                            name = 'image',
                        )
                    except Exception as e:
                        if self.wis3d is not None:
                            get_logger().error(f'Failed to visualize the current performance on wis3d: {e}')

                img_final = splice_img(imgs_spliced, grid_ids=[[i] for i in range(len(vis_data['img_patch']))])

                img_final = to_tensor(img_final, device=None).permute(2, 0, 1)
                summary_writer.add_image(f'{mode}/visualization', img_final, self.global_step)

        except Exception as e:
            get_logger().error(f'Failed to visualize the current performance: {e}')


    def _adapt_actionhsmr_v1(self, batch): 
        from lib.data.augmentation.skel import rot_skel_on_plane
        rot_deg = batch['augm_args']['rot_deg']  # (B,)

        skel_params = rot_skel_on_plane(batch['raw_skel_params'], rot_deg)
        batch['gt_params'] = {}
        batch['gt_params']['poses_orient'] = skel_params['poses'][:, :3]
        batch['gt_params']['poses_body'] = skel_params['poses'][:, 3:]
        batch['gt_params']['betas'] = skel_params['betas']

        has_skel_params = batch['has_skel_params']
        batch['has_gt_params'] = {}
        batch['has_gt_params']['poses_orient'] = has_skel_params['poses']
        batch['has_gt_params']['poses_body'] = has_skel_params['poses']
        batch['has_gt_params']['betas'] = has_skel_params['betas']
        return batch

    def _adapt_img_inference(self, batch):
        if 'img_sequence' in batch:
            img_seq = batch['img_sequence']
        elif 'img_patch' in batch:
            img_seq = batch['img_patch']
        else:
            raise KeyError("There are no img_sequence or img_patch fields in the batch!")
            
        # turn to tensor
        if not torch.is_tensor(img_seq):
            img_seq = torch.tensor(img_seq)

        # Handle the batch dimension
        if img_seq.ndim == 3:  # [C, H, W]
            img_seq = img_seq.unsqueeze(0)  # [1, C, H, W]
        if img_seq.ndim == 4:  # [B, C, H, W]
            img_seq = img_seq.unsqueeze(1)  # [B, 1, C, H, W]

        if img_seq.shape[1] == 1:
            img_seq = img_seq.repeat(1, 3, 1, 1, 1)  # [B, 3, C, H, W]
        elif img_seq.shape[1] != 3:
            raise ValueError(f"Unsupported frame rates：img_sequence.shape={img_seq.shape}")

        return {
            'img_patch': img_seq,
            'kp2d': batch.get('kp2d'),
            'kp3d': batch.get('kp3d'),
            'raw_skel_params': batch.get('raw_skel_params'),
            'has_skel_params': batch.get('has_skel_params'),
            '__key__': batch.get('__key__', [''] * len(img_seq))
        }