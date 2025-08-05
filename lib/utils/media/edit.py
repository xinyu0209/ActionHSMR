import cv2
import imageio
import numpy as np

from typing import Union, Tuple, List
from pathlib import Path


def flex_resize_img(
    img    : np.ndarray,
    tgt_wh : Union[Tuple[int, int], None] = None,
    ratio  : Union[float, None] = None,
    kp_mod : int = 1,
):
    '''
    Resize the image to the target width and height. Set one of width and height to -1 to keep the aspect ratio.
    Only one of `tgt_wh` and `ratio` can be set, if both are set, `tgt_wh` will be used.

    ### Args
    - img: np.ndarray, (H, W, 3)
    - tgt_wh: Tuple[int, int], default=None
        - The target width and height, set one of them to -1 to keep the aspect ratio.
    - ratio: float, default=None
        - The ratio to resize the frames. It will be used if `tgt_wh` is not set.
    - kp_mod: int, default 1
        - Keep the width and height as multiples of `kp_mod`.
        - For example, if `kp_mod=16`, the width and height will be rounded to the nearest multiple of 16.

    ### Returns
    - np.ndarray, (H', W', 3)
        - The resized iamges.
    '''
    assert len(img.shape) == 3, 'img must have 3 dimensions.'
    return flex_resize_video(img[None], tgt_wh, ratio, kp_mod)[0]


def flex_resize_video(
    frames : np.ndarray,
    tgt_wh : Union[Tuple[int, int], None] = None,
    ratio  : Union[float, None] = None,
    kp_mod : int = 1,
):
    '''
    Resize the frames to the target width and height. Set one of width and height to -1 to keep the aspect ratio.
    Only one of `tgt_wh` and `ratio` can be set, if both are set, `tgt_wh` will be used.

    ### Args
    - frames: np.ndarray, (L, H, W, 3)
    - tgt_wh: Tuple[int, int], default=None
        - The target width and height, set one of them to -1 to keep the aspect ratio.
    - ratio: float, default=None
        - The ratio to resize the frames. It will be used if `tgt_wh` is not set.
    - kp_mod: int, default 1
        - Keep the width and height as multiples of `kp_mod`.
        - For example, if `kp_mod=16`, the width and height will be rounded to the nearest multiple of 16.

    ### Returns
    - np.ndarray, (L, H', W', 3)
        - The resized frames.
    '''
    assert tgt_wh is not None or ratio is not None, 'At least one of tgt_wh and ratio must be set.'
    if tgt_wh is not None:
        assert len(tgt_wh) == 2, 'tgt_wh must be a tuple of 2 elements.'
        assert tgt_wh[0] > 0 or tgt_wh[1] > 0, 'At least one of width and height must be positive.'
    if ratio is not None:
        assert ratio > 0, 'ratio must be positive.'
    assert len(frames.shape) == 4, 'frames must have 3 or 4 dimensions.'

    def align_size(val:float):
        ''' It will round the value to the nearest multiple of `kp_mod`. '''
        return int(round(val / kp_mod) * kp_mod)

    # Calculate the target width and height.
    orig_h, orig_w = frames.shape[1], frames.shape[2]
    tgt_wh = (int(orig_w * ratio), int(orig_h * ratio)) if tgt_wh is None else tgt_wh  # Get wh from ratio if not given. # type: ignore
    tgt_w, tgt_h = tgt_wh
    tgt_w = align_size(orig_w * tgt_h / orig_h) if tgt_w == -1 else align_size(tgt_w)
    tgt_h = align_size(orig_h * tgt_w / orig_w) if tgt_h == -1 else align_size(tgt_h)
    # Resize the frames.
    resized_frames = np.stack([cv2.resize(frame, (tgt_w, tgt_h)) for frame in frames])

    return resized_frames


def splice_img(
    img_grids : Union[List[np.ndarray], np.ndarray],
    grid_ids  : Union[List[int], np.ndarray],
):
    '''
    Splice the images with the same size, according to the grid index.
    For example, you have 3 images [i1, i2, i3], and a `grid_ids` matrix:
    [[ 0,  1],                              |i1|i2|
     [ 2, -1],  , then the results will be  |i3|ib|  , where ib means a black place holder.
     [-1, -1]]                              |ib|ib|

    ### Args
    - img_grids: List[np.ndarray] or np.ndarray, (K, H, W, 3)
        - The source images to splice. It indicates that all the images have the same size.
    - grid_ids: List[int] or np.ndarray, (Y, X)
        - The grid index of each image. It should be a 2D matrix with integers as the type of elements.
        - The value in this matrix indexed the image in the `video_grids`, so it ranges from 0 to K-1.
        - Specially, set the grid index to -1 to use a black place holder.

    ### Returns
    - np.ndarray, (H*Y, W*X, 3)
        - The spliced images.
    '''
    if isinstance(img_grids, List):
        img_grids = np.stack(img_grids)
    if isinstance(grid_ids, List):
        grid_ids = np.array(grid_ids)

    assert len(img_grids.shape) == 4, 'img_grids must be in shape (K, H, W, 3).'
    return splice_video(img_grids[:, None], grid_ids)[0]


def splice_video(
    video_grids : Union[List[np.ndarray], np.ndarray],
    grid_ids    : Union[List[int], np.ndarray],
):
    '''
    Splice the videos with the same size, according to the grid index.
    For example, you have 3 videos [v1, v2, v3], and a `grid_ids` matrix:
    [[ 0,  1],                              |v1|v2|
     [ 2, -1],  , then the results will be  |v3|vb|  , wher vb means a black place holder.
     [-1, -1]]                              |vb|vb|

    ### Args
    - video_grids: List[np.ndarray] or np.ndarray, (K, L, H, W, C)
        - The source videos to splice. It indicates that all the videos have the same size.
    - grid_ids: List[int] or np.ndarray, (Y, X)
        - The grid index of each video. It should be a 2D matrix with integers as the type of elements.
        - The value in this matrix indexed the video in the `video_grids`, so it ranges from 0 to K-1.
        - Specially, set the grid index to -1 to use a black place holder.

    ### Returns
    - np.ndarray, (L, H*Y, W*X, C)
        - The spliced video.
    '''
    if isinstance(video_grids, List):
        video_grids = np.stack(video_grids)
    if isinstance(grid_ids, List):
        grid_ids = np.array(grid_ids)

    assert len(video_grids.shape) == 5, 'video_grids must be in shape (K, L, H, W, 3).'
    assert len(grid_ids.shape) == 2, 'grid_ids must be a 2D matrix.'
    assert isinstance(grid_ids[0, 0].item(), int), f'grid_ids must be an integer matrix, but got {grid_ids.dtype}.'

    # Splice the videos.
    K, L, H, W, C = video_grids.shape
    Y, X = grid_ids.shape

    # Initialize the spliced video.
    spliced_video = np.zeros((L, H*Y, W*X, C), dtype=np.uint8)
    for x in range(X):
        for y in range(Y):
            grid_id = grid_ids[y, x]
            if grid_id == -1:
                continue
            spliced_video[:, y*H:(y+1)*H, x*W:(x+1)*W, :] = video_grids[grid_id]

    return spliced_video


def crop_img(
    img  : np.ndarray,
    lurb : Union[np.ndarray, List],
):
    '''
    Crop the image with the given bounding box.
    The data should be represented in uint8.
    If the bounding box is out of the image, pad the image with zeros.

    ### Args
    - img: np.ndarray, (H, W, C)
    - lurb: np.ndarray or list, (4,)
        - The bounding box in the format of left, up, right, bottom.

    ### Returns
    - np.ndarray, (H', W', C)
        - The cropped image.
    '''

    return crop_video(img[None], lurb)[0]


def crop_video(
    frames : np.ndarray,
    lurb   : Union[np.ndarray, List],
):
    '''
    Crop the video with the given bounding box. 
    The data should be represented in uint8.
    If the bounding box is out of the video, pad the frames with zeros.

    ### Args
    - frames: np.ndarray, (L, H, W, C)
    - lurb: np.ndarray or list, (4,)
        - The bounding box in the format of left, up, right, bottom.

    ### Returns
    - np.ndarray, (L, H', W', C)
        - The cropped video.
    '''
    assert len(frames.shape) == 4, 'framess must have 4 dimensions.'
    if isinstance(lurb, List):
        lurb = np.array(lurb)

    l, u, r, b = lurb.astype(int)
    L, H, W = frames.shape[:3]
    l_, u_, r_, b_ = max(0, l), max(0, u), min(W, r), min(H, b)
    cropped_frames = np.zeros((L, b-u, r-l, 3), dtype=np.uint8)
    cropped_frames[:, u_-u:b_-u, l_-l:r_-l] = frames[:, u_:b, l_:r]

    return cropped_frames

def pad_img(
    img     : np.ndarray,
    tgt_wh  : Tuple[int, int],
    pad_val : int = 0,
    align   : str = 'c-c',
):
    '''
    Pad the image to the target width and height.

    ### Args
    - img: np.ndarray, (H, W, 3)
    - tgt_wh: Tuple[int, int]
        - The target width and height. Use -1 to indicate the original scale.
    - pad_value: int, default 0
        - The value to pad the image. 
    - align: str, default 'c-c'
        - The alignment of the image. It should be in the format of 'h-v', 
          where 'h' and 'v' can be 'l', 'c', 'r' and 't', 'c', 'b' respectively.

    ### Returns
    - np.ndarray, (H', W', 3)
        - The padded image.
    '''
    assert len(img.shape) == 3, 'img must have 3 dimensions.'
    return pad_video(img[None], tgt_wh, pad_val, align)[0]

def pad_video(
    frames  : np.ndarray,
    tgt_wh  : Tuple[int, int],
    pad_val : int = 0,
    align   : str = 'c-c',
):
    '''
    Pad the video to the target width and height.

    ### Args
    - frames: np.ndarray, (L, H, W, 3)
    - tgt_wh: Tuple[int, int]
        - The target width and height. Use -1 to indicate the original scale.
    - pad_value: int, default 0
        - The value to pad the frames.

    ### Returns
    - np.ndarray, (L, H', W', 3)
        - The padded frames.
    '''
    # Check data validity.
    assert len(frames.shape) == 4, 'frames must have 4 dimensions.'
    assert len(tgt_wh) == 2, 'tgt_wh must be a tuple of 2 elements.'
    H, W = frames.shape[1], frames.shape[2]
    if tgt_wh[0] == -1: tgt_wh = (W, tgt_wh[1])
    if tgt_wh[1] == -1: tgt_wh = (tgt_wh[0], H)
    assert tgt_wh[0] >= frames.shape[2] and tgt_wh[1] >= frames.shape[1], 'The target size must be larger than the original size.'
    assert pad_val >= 0 and pad_val <= 255, 'The pad value must be in the range of [0, 255].'
    # Check align pattern.
    align = align.split('-')
    assert len(align) == 2, 'align must be in the format of "h-v".'
    assert align[0] in ['l', 'c', 'r'] and align[1] in ['l', 'c', 'r'], 'align must be in ["l", "c", "r"].'

    tgt_w, tgt_h = tgt_wh
    pad_pix = [tgt_w - W, tgt_h - H]  # indicate how many pixels to be padded
    pad_lu  = [0, 0]  # how many pixels to pad on the left and the up side
    for direction in [0, 1]:
        if align[direction] == 'c':
            pad_lu[direction] = pad_pix[direction] // 2
        elif align[direction] == 'r':
            pad_lu[direction] = pad_pix[direction]
    pad_l, pad_r, pad_u, pad_b = pad_lu[0], pad_pix[0] - pad_lu[0], pad_lu[1], pad_pix[1] - pad_lu[1]

    padded_frames = np.pad(frames, ((0, 0), (pad_u, pad_b), (pad_l, pad_r), (0, 0)), 'constant', constant_values=pad_val)

    return padded_frames