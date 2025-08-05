from lib.kits.basic import *

from lib.utils.data import *
from lib.utils.geometry.rotation import (
    euler_angles_to_matrix,
    axis_angle_to_matrix,
    matrix_to_euler_angles,
)
from lib.body_models.skel_utils.transforms import params_q2rot


def rot_q_orient(
    q       : Union[np.ndarray, torch.Tensor],
    rot_rad : Union[np.ndarray, torch.Tensor],
):
    '''
    ### Args
        - q: np.ndarray or tensor, shape = (B, 3)
            - SKEL style rotation representation.
        - rot: np.ndarray or tensor, shape = (B,)
            - Rotation angle in radian.
    ### Returns
        - np.ndarray: Rotated orientation SKEL q.
    '''
    # Transform skel q to rot mat.
    q, recover_type_back = to_tensor(q, device=None, temporary=True)  # (B, 3)
    q = q[:, [2, 1, 0]]
    Rp = euler_angles_to_matrix(q, convention="YXZ")
    # Rotate around z
    rot = to_tensor(-rot_rad, device=q.device).float()  # (B,)
    padding_zeros = torch.zeros_like(rot)  # (B,)
    R = torch.stack([rot, padding_zeros, padding_zeros], dim=1)  # (B, 3)
    R = axis_angle_to_matrix(R)
    R = torch.matmul(R, Rp)
    # Transform rot mat to skel q.
    q = matrix_to_euler_angles(R, convention="YXZ")
    q = q[:, [2, 1, 0]]
    q = recover_type_back(q)  # (B, 3)

    return q


def rot_skel_on_plane(
    params  : Union[Dict, np.ndarray, torch.Tensor],
    rot_deg : Union[np.ndarray, torch.Tensor, List[float]]
):
    '''
    Rotate the skel parameters on the plane (around the z-axis),
    in order to align the skel with the rotated image. To perform
    this operation, we need to modify the orientation of the skel
    parameters.

    ### Args
        - params: Dict or (np.ndarray or torch.Tensor)
            - If is dict, it should contain the following keys
                - 'poses': np.ndarray or torch.Tensor (B, 72)
                - ...
            - If is np.ndarray or torch.Tensor, it should be the 'poses' part.
        - rot_deg: np.ndarray, torch.Tensor or List[float]
            - Rotation angle in degrees.

    ### Returns
        - One of the following according to the input type:
            - Dict: Modified skel parameters.
            - np.ndarray or torch.Tensor: Modified skel poses parameters.
    '''
    rot_deg = to_numpy(rot_deg)  # (B,)
    rot_rad = np.deg2rad(rot_deg)  # (B,)

    if isinstance(params, Dict):
        ret = {}
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                ret[k] = v.copy()
            elif isinstance(v, torch.Tensor):
                ret[k] = v.clone()
            else:
                ret[k] = v
        ret['poses'][:, :3] = rot_q_orient(ret['poses'][:, :3], rot_rad)
    elif isinstance(params, (np.ndarray, torch.Tensor)):
        if isinstance(params, np.ndarray):
            ret = params.copy()
        elif isinstance(params, torch.Tensor):
            ret = params.clone()
        else:
            raise TypeError(f'Unsupported type: {type(params)}')
        ret[:, :3] = rot_q_orient(ret[:, :3], rot_rad)
    else:
        raise TypeError(f'Unsupported type: {type(params)}')
    return ret