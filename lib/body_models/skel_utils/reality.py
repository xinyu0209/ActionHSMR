from lib.kits.basic import *

from lib.body_models.skel_utils.limits import SKEL_LIM_QID2IDX, SKEL_LIM_BOUNDS


qids_cfg = {
    'l_knee': [13],
    'r_knee': [6],
    'l_elbow': [42, 43],
    'r_elbow': [32, 33],
}


def eval_rot_delta(poses, tol_deg=5):
    tol_rad = np.deg2rad(tol_deg)

    res = {}
    for part in qids_cfg:
        qids = qids_cfg[part]
        violation_part = poses.new_zeros(poses.shape[0], len(qids))
        for i, qid in enumerate(qids):
            idx = SKEL_LIM_QID2IDX[qid]
            ea = poses[:, qid]
            ea = (ea + np.pi) % (2 * np.pi) - np.pi  # Normalize to (-pi, pi)
            exceed_lb = torch.where(
                    ea < SKEL_LIM_BOUNDS[idx][0] - tol_rad,
                    ea - SKEL_LIM_BOUNDS[idx][0] + tol_rad, 0
                )
            exceed_ub = torch.where(
                    ea > SKEL_LIM_BOUNDS[idx][1] + tol_rad,
                    ea - SKEL_LIM_BOUNDS[idx][1] - tol_rad, 0
                )
            violation_part[:, i] = exceed_lb.abs() + exceed_ub.abs()
        res[part] = violation_part

    return res