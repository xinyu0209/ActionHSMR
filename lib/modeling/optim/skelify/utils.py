from lib.kits.basic import *

from lib.body_models.skel_utils.definition import JID2QIDS


def gmof(x, sigma=100):
    '''
    Geman-McClure error function, to be used as a robust loss function.
    '''
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def compute_rel_change(prev_val: float, curr_val: float) -> float:
    '''
    Compute the relative change between two values.
    Copied from:
    https://github.com/vchoutas/smplify-x

    ### Args:
        - prev_val: float
        - curr_val: float

    ### Returns:
        - float
    '''
    return np.abs(prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


INVALID_JIDS = [37, 38]  # These joints are not reliable.

def get_kp_active_j_masks(parts:Union[str, List[str]], device='cuda'):
    # Generate the masks performed on the keypoints to mask the loss.
    act_jids = get_kp_active_jids(parts)
    masks = torch.zeros(44).to(device)
    masks[act_jids] = 1.0

    return masks


def get_kp_active_jids(parts:Union[str, List[str]]):
    if isinstance(parts, str):
        if parts == 'all':
            return get_kp_active_jids(['torso', 'limbs', 'head'])
        elif parts == 'hips':
            return [8, 9, 12, 27, 28, 39]
        elif parts == 'torso-lite':
            return [2, 5, 9, 12]
        elif parts == 'torso':
            return [1, 2, 5, 8, 9, 12, 27, 28, 33, 34, 37, 39, 40, 41]
        elif parts == 'limbs':
            return get_kp_active_jids(['limbs_proximal', 'limbs_distal'])
        elif parts == 'head':
            return [0, 15, 16, 17, 18, 38, 42, 43]
        elif parts == 'limbs_proximal':
            return [3, 6, 10, 13, 26, 29, 32, 35]
        elif parts == 'limbs_distal':
            return [4, 7, 11, 14, 19, 20, 21, 22, 23, 24, 25, 30, 31, 36]
        else:
            raise ValueError(f'Unsupported parts: {parts}')
    else:
        jids = []
        for part in parts:
            jids.extend(get_kp_active_jids(part))
        jids = set(jids) - set(INVALID_JIDS)
        return sorted(list(jids))


def get_params_active_j_masks(parts:Union[str, List[str]], device='cuda'):
    # Generate the masks performed on the keypoints to mask the loss.
    act_jids = get_params_active_jids(parts)
    masks = torch.zeros(24).to(device)
    masks[act_jids] = 1.0

    return masks


def get_params_active_jids(parts:Union[str, List[str]]):
    if isinstance(parts, str):
        if parts == 'all':
            return get_params_active_jids(['torso', 'limbs', 'head'])
        elif parts == 'torso-lite':
            return get_params_active_jids('torso')
        elif parts == 'hips':  # Enable `hips` if `poses_orient` is enabled.
            return [0]
        elif parts == 'torso':
            return [0, 11]
        elif parts == 'limbs':
            return get_params_active_jids(['limbs_proximal', 'limbs_distal'])
        elif parts == 'head':
            return [12, 13]
        elif parts == 'limbs_proximal':
            return [1, 6, 14, 15, 19, 20]
        elif parts == 'limbs_distal':
            return [2, 3, 4, 5, 7, 8, 9, 10, 16, 17, 18, 21, 22, 23]
        else:
            raise ValueError(f'Unsupported parts: {parts}')
    else:
        qids = []
        for part in parts:
            qids.extend(get_params_active_jids(part))
        return sorted(list(set(qids)))


def get_params_active_q_masks(parts:Union[str, List[str]], device='cuda'):
    # Generate the masks performed on the keypoints to mask the loss.
    act_qids = get_params_active_qids(parts)
    masks = torch.zeros(46).to(device)
    masks[act_qids] = 1.0

    return masks


def get_params_active_qids(parts:Union[str, List[str]]):
    act_jids = get_params_active_jids(parts)
    qids = []
    for act_jid in act_jids:
        qids.extend(JID2QIDS[act_jid])
    return sorted(list(set(qids)))


def estimate_kp2d_scale(
    kp2d      : torch.Tensor,
    edge_idxs : List[Tuple[int, int]] = [[5, 12], [2, 9]],  # shoulders to hips
):
    diff2d = []
    for edge in edge_idxs:
        diff2d.append(kp2d[:, edge[0]] - kp2d[:, edge[1]])  # list of (B, 2)
    scale2d = torch.stack(diff2d, dim=1).norm(dim=-1)  # (B, E)
    return scale2d.mean(dim=1)  # (B,)


@torch.no_grad()
def guess_cam_z(
    pd_kp3d      : torch.Tensor,
    gt_kp2d      : torch.Tensor,
    focal_length : float,
    edge_idxs    : List[Tuple[int, int]] = [[5, 12], [2, 9]],  # shoulders to hips
):
    '''
    Initializes the camera depth translation (i.e. z value) according to the ground truth 2D 
    keypoints and the predicted 3D keypoints.
    Modified from: https://github.com/vchoutas/smplify-x/blob/68f8536707f43f4736cdd75a19b18ede886a4d53/smplifyx/fitting.py#L36-L110

    ### Args
    - pd_kp3d: torch.Tensor, (B, J, 3)
    - gt_kp2d: torch.Tensor, (B, J, 2)
        - Without confidence.
    - focal_length: float
    - edge_idxs: List[Tuple[int, int]], default=[[5, 12], [2, 9]], i.e. shoulders to hips
        - Identify the edge to evaluate the scale of the entity.
    '''
    diff3d, diff2d = [], []
    for edge in edge_idxs:
        diff3d.append(pd_kp3d[:, edge[0]] - pd_kp3d[:, edge[1]])  # list of (B, 3)
        diff2d.append(gt_kp2d[:, edge[0]] - gt_kp2d[:, edge[1]])  # list of (B, 2)

    diff3d = torch.stack(diff3d, dim=1)  # (B, E, 3)
    diff2d = torch.stack(diff2d, dim=1)  # (B, E, 2)

    length3d = diff3d.norm(dim=-1)  # (B, E)
    length2d = diff2d.norm(dim=-1)  # (B, E)

    height3d = length3d.mean(dim=1)  # (B,)
    height2d = length2d.mean(dim=1)  # (B,)

    z_estim = focal_length * (height3d / height2d)  # (B,)
    return z_estim  # (B,)