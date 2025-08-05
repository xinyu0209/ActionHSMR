from lib.kits.basic import *

from lib.utils.geometry.rotation import axis_angle_to_matrix


def get_lim_cfg(tol_deg=5):
    tol_limit = np.deg2rad(tol_deg)
    lim_cfg = {
        'l_knee': {
            'jid': 4,
            'convention': 'XZY',
            'limitation': [
                [-tol_limit, 3/4*np.pi+tol_limit],
                [-tol_limit, tol_limit],
                [-tol_limit, tol_limit],
            ]
        },
        'r_knee': {
            'jid': 5,
            'convention': 'XZY',
            'limitation': [
                [-tol_limit, 3/4*np.pi+tol_limit],
                [-tol_limit, tol_limit],
                [-tol_limit, tol_limit],
            ]
        },
        'l_elbow': {
            'jid': 18,
            'convention': 'YZX',
            'limitation': [
                [-(3/4)*np.pi-tol_limit, tol_limit],
                [-tol_limit, tol_limit],
                [-3/4*np.pi/2-tol_limit, 3/4*np.pi/2+tol_limit],
            ]
        },
        'r_elbow': {
            'jid': 19,
            'convention': 'YZX',
            'limitation': [
                [-tol_limit, (3/4)*np.pi+tol_limit],
                [-tol_limit, tol_limit],
                [-3/4*np.pi/2-tol_limit, 3/4*np.pi/2+tol_limit],
            ]
        },
    }
    return lim_cfg


def matrix_to_possible_euler_angles(matrix: torch.Tensor, convention: str):
    '''
    Convert rotations given as rotation matrices to Euler angles in radians.

    ### Args
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    ### Returns
        List of possible euler angles in radians as tensor of shape (..., 3).
    '''
    from lib.utils.geometry.rotation import _index_from_letter, _angle_from_tan
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    central_angle_possible = []
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
        central_angle_possible = [central_angle, np.pi - central_angle]
    else:
        central_angle = torch.acos(matrix[..., i0, i0])
        central_angle_possible = [central_angle, -central_angle]

    o_possible = []
    for central_angle in central_angle_possible:
        o = (
            _angle_from_tan(
                convention[0], convention[1], matrix[..., i2], False, tait_bryan
            ),
            central_angle,
            _angle_from_tan(
                convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
            ),
        )
        o_possible.append(torch.stack(o, -1))
    return o_possible


def eval_rot_delta(body_pose, tol_deg=5):
    lim_cfg = get_lim_cfg(tol_deg)
    res ={}
    for name, cfg in lim_cfg.items():
        jid = cfg['jid'] - 1
        cvt = cfg['convention']
        lim = cfg['limitation']
        aa = body_pose[:, jid, :]  # (B, 3)
        mt = axis_angle_to_matrix(aa)  # (B, 3, 3)
        ea_possible = matrix_to_possible_euler_angles(mt, cvt)  # (B, 3)
        violation_reasonable = None
        for ea in ea_possible:
            violation = ea.new_zeros(ea.shape)  # (B, 3)

            for i in range(3):
                ea_i = ea[:, i]
                ea_i = (ea_i + np.pi) % (2 * np.pi) - np.pi  # Normalize to (-pi, pi)
                exceed_lb = torch.where(ea_i < lim[i][0], ea_i - lim[i][0], 0)
                exceed_ub = torch.where(ea_i > lim[i][1], ea_i - lim[i][1], 0)
                violation[:, i] = exceed_lb.abs() + exceed_ub.abs()  # (B, 3)
            if violation_reasonable is not None:  # minimize the violation
                upd_mask = violation.sum(-1) < violation_reasonable.sum(-1)
                violation_reasonable[upd_mask] = violation[upd_mask]
            else:
                violation_reasonable = violation

        res[name] = violation_reasonable
    return res