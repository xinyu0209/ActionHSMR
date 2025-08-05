from lib.kits.basic import *


def compute_kp3d_loss(gt_kp3d, pd_kp3d, ref_jid=25+14):
    conf = gt_kp3d[:, :, 3:].clone()  # (B, 43, 1)
    gt_kp3d_a = gt_kp3d[:, :, :3] - gt_kp3d[:, [ref_jid], :3]  # aligned, (B, J=44, 3)
    pd_kp3d_a = pd_kp3d[:, :, :3] - pd_kp3d[:, [ref_jid], :3]  # aligned, (B, J=44, 3)
    kp3d_loss = conf * F.l1_loss(pd_kp3d_a, gt_kp3d_a, reduction='none')  # (B, J=44, 3)
    return kp3d_loss.sum()  # (,)


def compute_kp2d_loss(gt_kp2d, pd_kp2d):
    conf = gt_kp2d[:, :, 2:].clone()  # (B, 44, 1)
    kp2d_loss = conf * F.l1_loss(pd_kp2d, gt_kp2d[:, :, :2], reduction='none')  # (B, 44, 2)
    return kp2d_loss.sum() # (,)