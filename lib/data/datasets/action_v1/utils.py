from lib.kits.basic import *

import os
import cv2
import braceexpand
from typing import List, Union

from .crop import *


def expand_urls(urls: Union[str, List[str]]):

    def expand_url(s):
        return os.path.expanduser(os.path.expandvars(s))

    if isinstance(urls, str):
        urls = [urls]
    urls = [u for url in urls for u in braceexpand.braceexpand(expand_url(url))]
    return urls


def get_augm_args(img_augm_cfg:Optional[DictConfig]):
    '''
    Perform some random augmentation to the image and patch it. Here we perform generate augmentation arguments
    according to the configuration and random seed.

    Briefly speaking, things done here are: size scale, color scale, rotate, flip, extreme crop, translate.
    '''
    sample_args = {
            'bbox_scale'      : 1.0,
            'color_scale'     : [1.0, 1.0, 1.0],
            'rot_deg'         : 0.0,
            'do_flip'         : False,
            'do_extreme_crop' : False,
            'tx_ratio'        : 0.0,
            'ty_ratio'        : 0.0,
        }

    if img_augm_cfg is not None:
        sample_args['tx_ratio']   += np.clip(np.random.randn(), -1.0, 1.0) * img_augm_cfg.trans_factor
        sample_args['ty_ratio']   += np.clip(np.random.randn(), -1.0, 1.0) * img_augm_cfg.trans_factor
        sample_args['bbox_scale'] += np.clip(np.random.randn(), -1.0, 1.0) * img_augm_cfg.bbox_scale_factor

        if np.random.random() <= img_augm_cfg.rot_aug_rate:
            sample_args['rot_deg'] += np.clip(np.random.randn(), -2.0, 2.0) * img_augm_cfg.rot_factor
        if np.random.random() <= img_augm_cfg.flip_aug_rate:
            sample_args['do_flip'] = True
        if np.random.random() <= img_augm_cfg.extreme_crop_aug_rate:
            sample_args['do_extreme_crop'] = True

        c_up  = 1.0 + img_augm_cfg.half_color_scale
        c_low = 1.0 - img_augm_cfg.half_color_scale
        sample_args['color_scale'] = [
                np.random.uniform(c_low, c_up),
                np.random.uniform(c_low, c_up),
                np.random.uniform(c_low, c_up),
            ]
    return sample_args


def rotate_2d(pt_2d: np.ndarray, rot_rad: float) -> np.ndarray:
    '''
    Rotate a 2D point on the x-y plane.
    Copied from: https://github.com/shubham-goel/4D-Humans/blob/6ec79656a23c33237c724742ca2a0ec00b398b53/hmr2/datasets/utils.py#L90-L104

    ### Args
        - pt_2d: np.ndarray
            - Input 2D point with shape (2,).
        - rot_rad: float
            - Rotation angle.

    ### Returns
        - np.ndarray
            - Rotated 2D point.
    '''
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def extreme_cropping_aggressive(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray) -> Tuple:
    """
    Perform aggressive extreme cropping.
    Copied from: https://github.com/shubham-goel/4D-Humans/blob/6ec79656a23c33237c724742ca2a0ec00b398b53/hmr2/datasets/utils.py#L978-L1025

    ### Args
        - center_x: float
            - x coordinate of bounding box center.
        - center_y: float
            - y coordinate of bounding box center.
        - width: float
            - Bounding box width.
        - height: float
            - Bounding box height.
        - keypoints_2d: np.ndarray
            - Array of shape (N, 3) containing 2D keypoint locations.
        - rescale: float
            - Scale factor to rescale bounding boxes computed from the keypoints.

    ### Returns
        - center_x: float
            - x coordinate of bounding box center.
        - center_y: float
            - y coordinate of bounding box center.
        - bbox_size: float
            - Bounding box size.
    """
    p = torch.rand(1).item()
    if full_body(keypoints_2d):
        if p < 0.2:
            center_x, center_y, width, height = crop_to_hips(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.3:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.4:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.5:
            center_x, center_y, width, height = crop_torso_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.6:
            center_x, center_y, width, height = crop_rightarm_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.7:
            center_x, center_y, width, height = crop_leftarm_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.8:
            center_x, center_y, width, height = crop_legs_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.9:
            center_x, center_y, width, height = crop_rightleg_only(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_leftleg_only(center_x, center_y, width, height, keypoints_2d)
    elif upper_body(keypoints_2d):
        if p < 0.2:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.4:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.6:
            center_x, center_y, width, height = crop_torso_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.8:
            center_x, center_y, width, height = crop_rightarm_only(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_leftarm_only(center_x, center_y, width, height, keypoints_2d)
    return center_x, center_y, max(width, height)


def gen_trans_from_patch_cv(
    c_x        : float,
    c_y        : float,
    src_width  : float,
    src_height : float,
    dst_width  : float,
    dst_height : float,
    scale      : float,
    rot        : float
) -> np.ndarray:
    '''
    Create transformation matrix for the bounding box crop.
    Copied from: https://github.com/shubham-goel/4D-Humans/blob/6ec79656a23c33237c724742ca2a0ec00b398b53/hmr2/datasets/utils.py#L107-L154

    ### Args
        - c_x: float
            - Bounding box center x coordinate in the original image.
        - c_y: float
            - Bounding box center y coordinate in the original image.
        - src_width: float
            - Bounding box width.
        - src_height: float
            - Bounding box height.
        - dst_width: float
            - Output box width.
        - dst_height: float
            - Output box height.
        - scale: float
            - Rescaling factor for the bounding box (augmentation).
        - rot: float
            - Random rotation applied to the box.

    ### Returns
        - trans: np.ndarray
            - Target geometric transformation.
    '''
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))  # (2, 3)  # type: ignore

    return trans


def generate_image_patch_cv2(
    img          : np.ndarray,
    c_x          : float,
    c_y          : float,
    bb_width     : float,
    bb_height    : float,
    patch_width  : float,
    patch_height : float,
    do_flip      : bool,
    scale        : float,
    rot          : float,
    border_mode  = cv2.BORDER_CONSTANT,
    border_value = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Crop the input image and return the crop and the corresponding transformation matrix.
    Copied from: https://github.com/shubham-goel/4D-Humans/blob/6ec79656a23c33237c724742ca2a0ec00b398b53/hmr2/datasets/utils.py#L343-L386

    ### Args
        - img: np.ndarray, shape = (H, W, 3)
        - c_x: float
            - Bounding box center x coordinate in the original image.
        - c_y: float
            - Bounding box center y coordinate in the original image.
        - bb_width: float
            - Bounding box width.
        - bb_height: float
            - Bounding box height.
        - patch_width: float
            - Output box width.
        - patch_height: float
            - Output box height.
        - do_flip: bool
            - Whether to flip image or not.
        - scale: float
            - Rescaling factor for the bounding box (augmentation).
        - rot: float
            - Random rotation applied to the box.
    ### Returns
        - img_patch: np.ndarray
            - Cropped image patch of shape (patch_height, patch_height, 3)
        - trans: np.ndarray
            - Transformation matrix.
    '''
    img_height, img_width, img_channels = img.shape
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot)  # (2, 3)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)),
                        flags=cv2.INTER_LINEAR,
                        borderMode=border_mode,
                        borderValue=border_value,
                )  # type: ignore
    # Force borderValue=cv2.BORDER_CONSTANT for alpha channel
    if (img.shape[2] == 4) and (border_mode != cv2.BORDER_CONSTANT):
        img_patch[:,:,3] = cv2.warpAffine(img[:,:,3], trans, (int(patch_width), int(patch_height)),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT,
                            )

    return img_patch, trans


def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    '''
    Increase the size of the bounding box to match the target shape.
    Copied from https://github.com/shubham-goel/4D-Humans/blob/6ec79656a23c33237c724742ca2a0ec00b398b53/hmr2/datasets/utils.py#L14-L33
    '''
    if target_aspect_ratio is None:
        return input_shape

    try:
        w , h = input_shape
    except (ValueError, TypeError):
        return input_shape

    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    if h_new < h or w_new < w:
        breakpoint()
    return np.array([w_new, h_new])


body_permutation    = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
extra_permutation   = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
FLIP_KP_PERMUTATION = body_permutation + [25 + i for i in extra_permutation]

def flip_lr_keypoints(joints: np.ndarray, width: float) -> np.ndarray:
    """
    Flip 2D or 3D keypoints.
    Modified from: https://github.com/shubham-goel/4D-Humans/blob/6ec79656a23c33237c724742ca2a0ec00b398b53/hmr2/datasets/utils.py#L448-L462

    ### Args
        - joints: np.ndarray
            - Array of shape (N, 3) or (N, 4) containing 2D or 3D keypoint locations and confidence.
        - flip_permutation: list
            - Permutation to apply after flipping.
    ### Returns
        - np.ndarray
            - Flipped 2D or 3D keypoints with shape (N, 3) or (N, 4) respectively.
    """
    joints = joints.copy()
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1
    joints = joints[FLIP_KP_PERMUTATION]

    return joints