from lib.kits.basic import *
import re
from collections import defaultdict, deque
import itertools
from typing import Iterator, Dict, Any
import math
import webdataset as wds

from .utils import (
    get_augm_args,
    expand_to_aspect_ratio,
    generate_image_patch_cv2,
    flip_lr_keypoints,
    extreme_cropping_aggressive,
)

def apply_corrupt_filter(dataset:wds.WebDataset):
    AIC_TRAIN_CORRUPT_KEYS = {
        '0a047f0124ae48f8eee15a9506ce1449ee1ba669', '1a703aa174450c02fbc9cfbf578a5435ef403689',
        '0394e6dc4df78042929b891dbc24f0fd7ffb6b6d', '5c032b9626e410441544c7669123ecc4ae077058',
        'ca018a7b4c5f53494006ebeeff9b4c0917a55f07', '4a77adb695bef75a5d34c04d589baf646fe2ba35',
        'a0689017b1065c664daef4ae2d14ea03d543217e', '39596a45cbd21bed4a5f9c2342505532f8ec5cbb',
        '3d33283b40610d87db660b62982f797d50a7366b',
    }

    CORRUPT_KEYS = {
        *{f'aic-train/{k}' for k in AIC_TRAIN_CORRUPT_KEYS},
        *{f'aic-train-vitpose/{k}' for k in AIC_TRAIN_CORRUPT_KEYS},
    }

    dataset = dataset.select(lambda sample: (sample['__key__'] not in CORRUPT_KEYS))
    return dataset


def apply_multi_ppl_splitter(dataset:wds.WebDataset):
    '''
    Each item in the raw dataset contains multiple people, we need to split them into individual samples.
    Meanwhile, we also need to note down the person id (pid) for each sample.
    '''
    def multi_ppl_splitter(source):
        for item in source:
            data_multi_ppl = item['data.pyd']  # list of data for multiple people
            for pid, data in enumerate(data_multi_ppl):
                data['pid'] = pid

                if 'detection.npz' in item:
                    det_idx = data['extra_info']['detection_npz_idx']
                    mask = item['detection.npz']['masks'][det_idx]
                else:
                    mask = np.ones_like(item['jpg'][:, :, 0], dtype=bool)
                yield {
                    '__key__'  : item['__key__'] + f'_{pid}',
                    'img_name' : item['__key__'],
                    'img'      : item['jpg'],
                    'data'     : data,
                    'mask'     : mask,
                }

    return dataset.compose(multi_ppl_splitter)


def apply_keys_adapter(dataset:wds.WebDataset):
    ''' Adapt the keys of the items, so we can adapt different version of dataset. '''
    def keys_adapter(item):
        data = item['data']
        data['kp2d'] = data.pop('keypoints_2d')
        data['kp3d'] = data.pop('keypoints_3d')
        return item
    return dataset.map(keys_adapter)


def apply_bad_pgt_params_nan_suppressor(dataset:wds.WebDataset):
    ''' If the poses or betas contain NaN, we regard it as bad pseudo-GT and zero them out. '''
    def bad_pgt_params_suppressor(item):
        for side in ['orig', 'flip']:
            poses = item['data'][f'{side}_poses']  # (J, 3)
            betas = item['data'][f'{side}_betas']  # (10,)
            poses_has_nan = np.isnan(poses).any()
            betas_has_nan = np.isnan(betas).any()
            if poses_has_nan or betas_has_nan:
                item['data'][f'{side}_has_poses'] = False
                item['data'][f'{side}_has_betas'] = False
                if poses_has_nan:
                    item['data'][f'{side}_poses'][:] = 0
                if betas_has_nan:
                    item['data'][f'{side}_betas'][:] = 0
        return item
    dataset = dataset.map(bad_pgt_params_suppressor)
    return dataset


def apply_bad_pgt_params_kp2d_err_suppressor(dataset:wds.WebDataset, thresh:float=0.1):
    ''' If the 2D keypoints error of one single person is higher than the threshold, we regard it as bad pseudo-GT. '''
    if thresh > 0:
        def bad_pgt_params_suppressor(item):
            for side in ['orig', 'flip']:
                if thresh > 0:
                    kp2d_err = item['data'][f'{side}_kp2d_err']
                    is_valid_pgt = kp2d_err < thresh
                    item['data'][f'{side}_has_poses'] = is_valid_pgt
                    item['data'][f'{side}_has_betas'] = is_valid_pgt
            return item
        dataset = dataset.map(bad_pgt_params_suppressor)
    return dataset


def apply_bad_pgt_params_pve_max_suppressor(dataset:wds.WebDataset, thresh:float=0.1):
    ''' If the PVE-Max of one single person is higher than the threshold, we regard it as bad pseudo-GT. '''
    if thresh > 0:
        def bad_pgt_params_suppressor(item):
            for side in ['orig', 'flip']:
                if thresh > 0:
                    pve_max = item['data'][f'{side}_pve_max']
                    is_valid_pose = not math.isnan(pve_max)
                    is_valid_pgt = pve_max < thresh and is_valid_pose
                    item['data'][f'{side}_has_poses'] = is_valid_pgt
                    item['data'][f'{side}_has_betas'] = is_valid_pgt
            return item
        dataset = dataset.map(bad_pgt_params_suppressor)
    return dataset


def apply_bad_kp_suppressor(dataset:wds.WebDataset, thresh:float=0.0):
    ''' If the confidence of a keypoint is lower than the threshold, we reset it to 0. '''
    eps = 1e-6
    if thresh > eps:
        def bad_kp_suppressor(item):
            if thresh > 0:
                kp2d = item['data']['kp2d']
                kp2d_conf = np.where(kp2d[:, 2] < thresh, 0.0, kp2d[:, 2])  # suppress bad keypoints
                item['data']['kp2d'] = np.concatenate([kp2d[:, :2], kp2d_conf[:, None]], axis=1)
            return item
        dataset = dataset.map(bad_kp_suppressor)
    return dataset


def apply_bad_betas_suppressor(dataset:wds.WebDataset, thresh:float=3):
    ''' If the absolute value of betas is higher than the threshold, we regard it as bad betas. '''
    eps = 1e-6
    if thresh > eps:
        def bad_betas_suppressor(item):
                for side in ['orig', 'flip']:
                    has_betas = item['data'][f'{side}_has_betas']  # use this condition to save time
                    if thresh > 0 and has_betas:
                        betas_abs = np.abs(item['data'][f'{side}_betas'])
                        if (betas_abs > thresh).any():
                            item['data'][f'{side}_has_betas'] = False
                return item
        dataset = dataset.map(bad_betas_suppressor)
    return dataset


def apply_params_synchronizer(dataset:wds.WebDataset, poses_betas_simultaneous:bool=False):
    ''' Only when both poses and betas are valid, we regard them as valid. '''
    if poses_betas_simultaneous:
        def params_synchronizer(item):
            for side in ['orig', 'flip']:
                has_betas = item['data'][f'{side}_has_betas']
                has_poses = item['data'][f'{side}_has_poses']
                has_both = np.array(float((has_poses > 0) and (has_betas > 0)))
                item['data'][f'{side}_has_betas'] = has_both
                item['data'][f'{side}_has_poses'] = has_both
            return item
        dataset = dataset.map(params_synchronizer)
    return dataset


def apply_insuff_kp_filter(dataset:wds.WebDataset, cnt_thresh:int=4, conf_thresh:float=0.0):
    '''
    Counting the number of keypoints with confidence higher than the threshold.
    If the number is less than the threshold, we regard it has insufficient valid 2D keypoints.
    '''
    if cnt_thresh > 0:
        def insuff_kp_filter(item):
            kp_conf = item['data']['kp2d'][:, 2]
            return (kp_conf > conf_thresh).sum() > cnt_thresh
        dataset = dataset.select(insuff_kp_filter)
    return dataset


def apply_bbox_size_filter(dataset:wds.WebDataset, bbox_size_thresh:Optional[float]=None):
    if bbox_size_thresh:
        def bbox_size_filter(item):
            bbox_size = item['data']['scale'] * 200
            return bbox_size.min() > bbox_size_thresh  # ensure the lower bound is large enough
        dataset = dataset.select(bbox_size_filter)
    return dataset


def apply_reproj_err_filter(dataset:wds.WebDataset, thresh:float=0.0):
    ''' If the re-projection error is higher than the threshold, we regard it as bad sample. '''
    if thresh > 0:
        def reproj_err_filter(item):
            losses = item['data'].get('extra_info', {}).get('fitting_loss', np.array({})).item()
            reproj_loss = losses.get('reprojection_loss', None)
            return reproj_loss is None or reproj_loss < thresh
        dataset = dataset.select(reproj_err_filter)
    return dataset


def apply_invalid_betas_regularizer(dataset:wds.WebDataset, reg_betas:bool=False):
    ''' For those items with invalid betas, set them to zero. '''
    if reg_betas:
        def betas_regularizer(item):
            # Always have betas set to zero, and all valid.
            for side in ['orig', 'flip']:
                has_betas = item['data'][f'{side}_has_betas']
                betas = item['data'][f'{side}_betas']

                if not (has_betas > 0):
                    item['data'][f'{side}_has_betas'] = np.array(float((True)))
                    item['data'][f'{side}_betas'] = betas * 0
            return item
        dataset = dataset.map(betas_regularizer)
    return dataset

# 添加三帧序列处理函数
def apply_three_frame_sequence(dataset: wds.WebDataset):
    """
    顺序地每 3 条样本拼成一组：
        prev_frame, key_frame, next_frame
    如果剩余不足 3 条就丢弃（或者你也可以做 padding）。
    """
    def _make_sequence(source: Iterator[Dict[str, Any]]):
        # itertools.batched 需要 python 3.12+
        # 低版本用 grouper 实现即可
        try:
            from itertools import batched
        except ImportError:
            # 低版本兼容
            def batched(iterable, n):
                it = iter(iterable)
                while True:
                    chunk = list(itertools.islice(it, n))
                    if len(chunk) < n:
                        break   # 丢弃尾部不足 3 条的数据
                    yield chunk

        for triplet in batched(source, 3):
            prev, center, nxt = triplet
            yield {
                "__key__": f"{center['__key__']}_seq",
                "prev_frame": prev,
                "key_frame": center,
                "next_frame": nxt,
                "center_frame_name": center["img_name"],
            }

    return dataset.compose(_make_sequence)

def apply_example_formatter(dataset:wds.WebDataset, cfg:DictConfig):
    ''' Format each item to the wanted format for training, 三帧序列版本，轻量内存友好. '''

    def get_fmt_data(raw_item:Dict, augm_args:Dict, cfg:DictConfig):
        '''
        On the one hand, we will perform the augmentation to the image, on the other hand, we need to
        crop the image to the patch according to the bbox. Both processes would influence the position
        of related keypoints.
        After that, we need to align the 2D & 3D keypoints to the augmented image.
        '''
        # 1. Prepare the raw data that will be used in the following steps.
        img_rgb = raw_item['img']  # (H, W, 3)
        img_a = raw_item['mask'].astype(np.uint8)[:, :, None] * 255  # (H, W, 1) mask is 0/1 valued
        img_rgba = np.concatenate([img_rgb, img_a], axis=2)  # (H, W, 4)
        H, W, C = img_rgb.shape
        cx, cy = raw_item['data']['center']
        # bbox_size = (raw_item['data']['scale'] * 200).max()
        bbox_size = expand_to_aspect_ratio(
                raw_item['data']['scale'] * 200,
                target_aspect_ratio = cfg.policy.bbox_shape,
            ).max()

        kp2d_with_conf = raw_item['data']['kp2d'].astype('float32')  # (J, 3)
        kp3d_with_conf = raw_item['data']['kp3d'].astype('float32')  # (J, 4)

        # 2. [img][Augmentation] Extreme cropping according to the 2D keypoints.
        if augm_args['do_extreme_crop']:
            cx_, cy_, bbox_size_ = extreme_cropping_aggressive(cx, cy, bbox_size, bbox_size, kp2d_with_conf)
            # Only apply the crop if the results is large enough.
            THRESH = 4
            if bbox_size_ > THRESH:
                cx, cy = cx_, cy_
                bbox_size = bbox_size_

        # 3. [img][Augmentation] Shift the center of the image.
        cx += augm_args['tx_ratio'] * bbox_size
        cy += augm_args['ty_ratio'] * bbox_size

        # 4. [img][Format] Crop the image to the patch.
        img_patch_cv2, transform_2d = generate_image_patch_cv2(
                img          = img_rgba,
                c_x          = cx,
                c_y          = cy,
                bb_width     = bbox_size,
                bb_height    = bbox_size,
                patch_width  = cfg.policy.img_patch_size,
                patch_height = cfg.policy.img_patch_size,
                do_flip      = augm_args['do_flip'],
                scale        = augm_args['bbox_scale'],
                rot          = augm_args['rot_deg'],
            )  # (H, W, 4), (2, 3)

        img_patch_hwc = img_patch_cv2.copy()[:, :, :3]  # (H, W, C)
        img_patch_chw = img_patch_hwc.transpose(2, 0, 1).astype(np.float32)

        # 5. [img][Augmentation] Scale the color
        for cid in range(min(C, 3)):
            img_patch_chw[cid] = np.clip(
                    a     = img_patch_chw[cid] * augm_args['color_scale'][cid],
                    a_min = 0,
                    a_max = 255,
                )

        # 6. [img][Format] Normalize the color.
        img_mean = [255. * x for x in cfg.policy.img_mean]
        img_std  = [255. * x for x in cfg.policy.img_std]
        for cid in range(min(C, 3)):
            img_patch_chw[cid] = (img_patch_chw[cid] - img_mean[cid]) / img_std[cid]

        # 7. [kp2d][Alignment] Align the 2D keypoints.
        # 7.1. Flip.
        if augm_args['do_flip']:
            kp2d_with_conf = flip_lr_keypoints(kp2d_with_conf, W)
        # 7.2. Others. Transform the 2D keypoints according to the same transformation of image.
        J = len(kp2d_with_conf)
        kp2d_homo = np.concatenate([kp2d_with_conf[:, :2], np.ones((J, 1))], axis=1)  # (J, 3)
        kp2d = np.einsum('ph, jh -> jp', transform_2d, kp2d_homo)  # (J, 2)
        kp2d_with_conf[:, :2] = kp2d  # (J, 3)

        # 8. [kp2d][Format] Normalize the 2D keypoints position to [-0.5, 0.5]-visible space.
        kp2d_with_conf[:, :2] = kp2d_with_conf[:, :2] / cfg.policy.img_patch_size - 0.5

        # 9. [kp3d][Alignment] Align the 3D keypoints.
        # 9.1. Flip.
        if augm_args['do_flip']:
            kp3d_with_conf = flip_lr_keypoints(kp3d_with_conf, W)
        # 9.2. In-plane rotation.
        rot_mat = np.eye(3)
        # TODO: maybe this part can be packed to a single function.
        if not augm_args['rot_deg'] == 0:
            rot_rad = -augm_args['rot_deg'] * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn,  cs]
        kp3d_with_conf[:, :3] = np.einsum('ij, kj -> ki', rot_mat, kp3d_with_conf[:, :3])


        return img_patch_chw, kp2d_with_conf, kp3d_with_conf

    def sequence_formatter(seq_item):
        augm_args = get_augm_args(cfg.image_augmentation)
        frames = []
    # 只处理关键帧的 gt 数据，其它帧只处理 img_patch
        for frame_type in ['prev_frame', 'key_frame', 'next_frame']:
            frame_item = seq_item[frame_type]
            img_patch, kp2d, kp3d = get_fmt_data(frame_item, augm_args, cfg)
            frames.append(img_patch)
        # 只在关键帧时保留 gt
            if frame_type == 'key_frame':
                key_kp2d = kp2d
                key_kp3d = kp3d

        img_sequence = np.stack(frames, axis=0)  # [3, C, H, W]

        key_data = seq_item['key_frame']['data']
        params_side = 'flip' if augm_args['do_flip'] else 'orig'

        return {
            'pid': key_data['pid'],
            'img_name': seq_item['key_frame']['img_name'],
            'img_patch': img_sequence.astype(np.float32),
            'kp2d': key_kp2d,      # 只输出关键帧的 gt
            'kp3d': key_kp3d,
            'augm_args': augm_args,
            'raw_skel_params': {
                'poses': key_data[f'{params_side}_poses'],
                'betas': key_data[f'{params_side}_betas'],
            },
            'has_skel_params': {
                'poses': key_data[f'{params_side}_has_poses'],
                'betas': key_data[f'{params_side}_has_betas'],
            },
            'updated_by_spin': False
        }

    return dataset.map(sequence_formatter)
