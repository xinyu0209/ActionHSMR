from lib.kits.basic import *

from .data import to_tensor


def lurb_to_cwh(
    lurb : Union[list, np.ndarray, torch.Tensor],
):
    '''
    Convert the left-upper-right-bottom format to the center-width-height format.

    ### Args
    - lurb: Union[list, np.ndarray, torch.Tensor], (..., 4)
        - The left-upper-right-bottom format bounding box.

    ### Returns
    - Union[list, np.ndarray, torch.Tensor], (..., 4)
        - The center-width-height format bounding box.
    '''
    lurb, recover_type_back = to_tensor(lurb, device=None, temporary=True)
    assert lurb.shape[-1] == 4, f"Invalid shape: {lurb.shape}, should be (..., 4)"

    c = (lurb[..., :2] + lurb[..., 2:]) / 2  # (..., 2)
    wh = lurb[..., 2:] - lurb[..., :2]  # (..., 2)

    cwh = torch.cat([c, wh], dim=-1)  # (..., 4)
    return recover_type_back(cwh)


def cwh_to_lurb(
    cwh : Union[list, np.ndarray, torch.Tensor],
):
    '''
    Convert the center-width-height format to the left-upper-right-bottom format.

    ### Args
    - cwh: Union[list, np.ndarray, torch.Tensor], (..., 4)
        - The center-width-height format bounding box.

    ### Returns
    - Union[list, np.ndarray, torch.Tensor], (..., 4)
        - The left-upper-right-bottom format bounding box.
    '''
    cwh, recover_type_back = to_tensor(cwh, device=None, temporary=True)
    assert cwh.shape[-1] == 4, f"Invalid shape: {cwh.shape}, should be (..., 4)"

    l = cwh[..., :2] - cwh[..., 2:] / 2  # (..., 2)
    r = cwh[..., :2] + cwh[..., 2:] / 2  # (..., 2)

    lurb = torch.cat([l, r], dim=-1)  # (..., 4)
    return recover_type_back(lurb)


def cwh_to_cs(
    cwh    : Union[list, np.ndarray, torch.Tensor],
    reduce : Optional[str] = None,
):
    '''
    Convert the center-width-height format to the center-scale format.
    *Only works when width and height are the same.*

    ### Args
    - cwh: Union[list, np.ndarray, torch.Tensor], (..., 4)
        - The center-width-height format bounding box.
    - reduce: Optional[str], default None, valid values: None, 'max'
        - Determine how to reduce the width and height to a single scale.

    ### Returns
    - Union[list, np.ndarray, torch.Tensor], (..., 3)
        - The center-scale format bounding box.
    '''
    cwh, recover_type_back = to_tensor(cwh, device=None, temporary=True)
    assert cwh.shape[-1] == 4, f"Invalid shape: {cwh.shape}, should be (..., 4)"

    if reduce is None:
        if (cwh[..., 2] != cwh[..., 3]).any():
            get_logger().warning(f"Width and height are supposed to be the same, but they're not. The larger one will be used.")

    c = cwh[..., :2]  # (..., 2)
    s = cwh[..., 2:].max(dim=-1)[0]  # (...,)

    cs = torch.cat([c, s[..., None]], dim=-1)  # (..., 3)
    return recover_type_back(cs)


def cs_to_cwh(
    cs : Union[list, np.ndarray, torch.Tensor],
):
    '''
    Convert the center-scale format to the center-width-height format.

    ### Args
    - cs: Union[list, np.ndarray, torch.Tensor], (..., 3)
        - The center-scale format bounding box.

    ### Returns
    - Union[list, np.ndarray, torch.Tensor], (..., 4)
        - The center-width-height format bounding box.
    '''
    cs, recover_type_back = to_tensor(cs, device=None, temporary=True)
    assert cs.shape[-1] == 3, f"Invalid shape: {cs.shape}, should be (..., 3)"

    c = cs[..., :2]  # (..., 2)
    s = cs[..., 2]  # (...,)

    cwh = torch.cat([c, s[..., None], s[..., None]], dim=-1)  # (..., 4)
    return recover_type_back(cwh)


def lurb_to_cs(
    lurb : Union[list, np.ndarray, torch.Tensor],
):
    '''
    Convert the left-upper-right-bottom format to the center-scale format.
    *Only works when width and height are the same.*

    ### Args
    - lurb: Union[list, np.ndarray, torch.Tensor], (..., 4)
        - The left-upper-right-bottom format bounding box.

    ### Returns
    - Union[list, np.ndarray, torch.Tensor], (..., 3)
        - The center-scale format bounding box.
    '''
    return cwh_to_cs(lurb_to_cwh(lurb), reduce='max')


def cs_to_lurb(
    cs : Union[list, np.ndarray, torch.Tensor],
):
    '''
    Convert the center-scale format to the left-upper-right-bottom format.

    ### Args
    - cs: Union[list, np.ndarray, torch.Tensor], (..., 3)
        - The center-scale format bounding box.

    ### Returns
    - Union[list, np.ndarray, torch.Tensor], (..., 4)
        - The left-upper-right-bottom format bounding box.
    '''
    return cwh_to_lurb(cs_to_cwh(cs))


def lurb_to_luwh(
    lurb : Union[list, np.ndarray, torch.Tensor],
):
    '''
    Convert the left-upper-right-bottom format to the left-upper-width-height format.

    ### Args
    - lurb: Union[list, np.ndarray, torch.Tensor]
        - The left-upper-right-bottom format bounding box.

    ### Returns
    - Union[list, np.ndarray, torch.Tensor]
        - The left-upper-width-height format bounding box.
    '''
    lurb, recover_type_back = to_tensor(lurb, device=None, temporary=True)
    assert lurb.shape[-1] == 4, f"Invalid shape: {lurb.shape}, should be (..., 4)"

    lu = lurb[..., :2]  # (..., 2)
    wh = lurb[..., 2:] - lurb[..., :2]  # (..., 2)

    luwh = torch.cat([lu, wh], dim=-1)  # (..., 4)
    return recover_type_back(luwh)


def luwh_to_lurb(
    luwh : Union[list, np.ndarray, torch.Tensor],
):
    '''
    Convert the left-upper-width-height format to the left-upper-right-bottom format.

    ### Args
    - luwh: Union[list, np.ndarray, torch.Tensor]
        - The left-upper-width-height format bounding box.

    ### Returns
    - Union[list, np.ndarray, torch.Tensor]
        - The left-upper-right-bottom format bounding box.
    '''
    luwh, recover_type_back = to_tensor(luwh, device=None, temporary=True)
    assert luwh.shape[-1] == 4, f"Invalid shape: {luwh.shape}, should be (..., 4)"

    l = luwh[..., :2]  # (..., 2)
    r = luwh[..., :2] + luwh[..., 2:]  # (..., 2)

    lurb = torch.cat([l, r], dim=-1)  # (..., 4)
    return recover_type_back(lurb)


def crop_with_lurb(data, lurb, padding=0):
    """
    Crop the img-like data according to the lurb bounding box.
    
    ### Args
    - data: Union[np.ndarray, torch.Tensor], shape (H, W, C)
        - Data like image.
    - lurb: Union[list, np.ndarray, torch.Tensor], shape (4,)
        - Bounding box with [left, upper, right, bottom] coordinates.
    - padding: int, default 0
        - Padding value for out-of-bound areas.
        
    ### Returns
    - Union[np.ndarray, torch.Tensor], shape (H', W', C)
        - Cropped image with padding if necessary.
    """
    data, recover_type_back = to_tensor(data, device=None, temporary=True)

    # Ensure lurb is in numpy array format for indexing
    lurb = np.array(lurb).astype(np.int64)
    l_, u_, r_, b_ = lurb

    # Determine the shape of the data.
    H_raw, W_raw, C_raw = data.size()

    # Compute the cropped patch size.
    H_patch = b_ - u_
    W_patch = r_ - l_

    # Create an output buffer of the crop size, initialized to padding
    if isinstance(data, np.ndarray):
        output = np.full((H_patch, W_patch, C_raw), padding, dtype=data.dtype)
    else:
        output = torch.full((H_patch, W_patch, C_raw), padding, dtype=data.dtype)

    # Calculate the valid region in the original data
    valid_l_ = max(0, l_)
    valid_u_ = max(0, u_)
    valid_r_ = min(W_raw, r_)
    valid_b_ = min(H_raw, b_)

    # Calculate the corresponding valid region in the output
    target_l_ = valid_l_ - l_
    target_u_ = valid_u_ - u_
    target_r_ = target_l_ + (valid_r_ - valid_l_)
    target_b_ = target_u_ + (valid_b_ - valid_u_)

    # Copy the valid region into the output buffer
    output[target_u_:target_b_, target_l_:target_r_, :] = data[valid_u_:valid_b_, valid_l_:valid_r_, :]

    return recover_type_back(output)


def fit_bbox_to_aspect_ratio(
    bbox      : np.ndarray,
    tgt_ratio : Optional[Tuple[int, int]] = None,
    bbox_type : str = 'lurb'
):
    '''
    Fit a random bounding box to a target aspect ratio through enlarging the bounding box with least change.
    
    ### Args
    - bbox: np.ndarray, shape is determined by `bbox_type`, e.g. for 'lurb', shape is (4,)
        - The bounding box to be modified. The format is determined by `bbox_type`.
    - tgt_ratio: Optional[Tuple[int, int]], default None
        - The target aspect ratio to be matched.
    - bbox_type: str, default 'lurb', valid values: 'lurb', 'cwh'.
    
    ### Returns
    - np.ndarray, shape is determined by `bbox_type`, e.g. for 'lurb', shape is (4,)
        - The modified bounding box.
    '''
    bbox = bbox.copy()
    if bbox_type == 'lurb':
        bbx_cwh = lurb_to_cwh(bbox)
        bbx_wh = bbx_cwh[2:]
    elif bbox_type == 'cwh':
        bbx_wh = bbox[2:]
    else:
        raise ValueError(f"Unsupported bbox type: {bbox_type}")

    new_bbx_wh = expand_wh_to_aspect_ratio(bbx_wh, tgt_ratio)

    if bbox_type == 'lurb':
        bbx_cwh[2:] = new_bbx_wh
        new_bbox = cwh_to_lurb(bbx_cwh)
    elif bbox_type == 'cwh':
        new_bbox = np.concatenate([bbox[:2], new_bbx_wh])
    else:
        raise ValueError(f"Unsupported bbox type: {bbox_type}")

    return new_bbox


def expand_wh_to_aspect_ratio(bbx_wh:np.ndarray, tgt_aspect_ratio:Optional[Tuple[int, int]]=None):
    '''
    Increase the size of the bounding box to match the target shape.
    Modified from https://github.com/shubham-goel/4D-Humans/blob/6ec79656a23c33237c724742ca2a0ec00b398b53/hmr2/datasets/utils.py#L14-L33
    '''
    if tgt_aspect_ratio is None:
        return bbx_wh

    try:
        bbx_w , bbx_h = bbx_wh
    except (ValueError, TypeError):
        get_logger().warning(f"Invalid bbox_wh content: {bbx_wh}")
        return bbx_wh

    tgt_w, tgt_h = tgt_aspect_ratio
    if bbx_h / bbx_w < tgt_h / tgt_w:
        new_h = max(bbx_w * tgt_h / tgt_w, bbx_h)
        new_w = bbx_w
    else:
        new_h = bbx_h
        new_w = max(bbx_h * tgt_w / tgt_h, bbx_w)
    assert new_h >= bbx_h and new_w >= bbx_w

    return to_numpy([new_w, new_h])