from lib.kits.basic import *

from lib.utils.camera import perspective_projection
from lib.modeling.losses import compute_poses_angle_prior_loss

from .utils import (
    gmof,
    guess_cam_z,
    estimate_kp2d_scale,
    get_kp_active_jids,
    get_params_active_q_masks,
)

def build_closure(
    self,
    cfg,
    optimizer,
    inputs,
    focal_length : float,
    gt_kp2d,
    log_data,
):
    B = len(gt_kp2d)

    act_parts = instantiate(cfg.parts)
    act_q_masks = None
    if not (act_parts == 'all' or 'all' in act_parts):
        act_q_masks = get_params_active_q_masks(act_parts)

    # Shortcuts for the inference of the skeleton model.
    def inference_skel(inputs):
        poses_active = torch.cat([inputs['poses_orient'], inputs['poses_body']], dim=-1)  # (B, 46)
        if act_q_masks is not None:
            poses_hidden = poses_active.clone().detach()  # (B, 46)
            poses = poses_active * act_q_masks + poses_hidden * (1 - act_q_masks)  # (B, 46)
        else:
            poses = poses_active
        skel_params = {
                'poses' : poses,  # (B, 46)
                'betas' : inputs['betas'],  # (B, 10)
            }
        skel_output = self.skel_model(**skel_params, skelmesh=False)
        return skel_params, skel_output

    # Estimate the camera depth as an initialization if depth loss is enabled.
    gs_cam_z = None
    if 'w_depth' in cfg.losses:
        with torch.no_grad():
            _, skel_output = inference_skel(inputs)
            gs_cam_z = guess_cam_z(
                    pd_kp3d      = skel_output.joints,
                    gt_kp2d      = gt_kp2d,
                    focal_length = focal_length,
                )

    # Prepare the focal length for the perspective projection.
    focal_length_xy = np.ones((B, 2)) * focal_length  # (B, 2)


    def closure():
        optimizer.zero_grad()

        # 📦 Data preparation.
        with PM.time_monitor('SKEL-forward'):
            skel_params, skel_output = inference_skel(inputs)

        with PM.time_monitor('reproj'):
            pd_kp2d = perspective_projection(
                    points       = to_tensor(skel_output.joints, device=self.device),
                    translation  = to_tensor(inputs['cam_t'], device=self.device),
                    focal_length = to_tensor(focal_length_xy, device=self.device),
                )

        with PM.time_monitor('compute_losses'):
            loss, losses = compute_losses(
                    # Loss configuration.
                    loss_cfg = instantiate(cfg.losses),
                    parts    = act_parts,
                    # Data inputs.
                    gt_kp2d   = gt_kp2d,
                    pd_kp2d   = pd_kp2d,
                    pd_params = skel_params,
                    pd_cam_z  = inputs['cam_t'][:, 2],
                    gs_cam_z  = gs_cam_z,
                )

        with PM.time_monitor('visualization'):
            VISUALIZE = True
            if VISUALIZE:
                # For visualize the optimization process.
                kp2d_err = torch.sum((pd_kp2d - gt_kp2d[..., :2]) ** 2, dim=-1) * gt_kp2d[..., 2]  # (B, J)
                kp2d_err = kp2d_err.sum(dim=-1) / (torch.sum(gt_kp2d[..., 2], dim=-1) + 1e-6)  # (B,)

                # Store logging data.
                if self.tb_logger is not None:
                    log_data.update({
                        'losses'      : losses,
                        'pd_kp2d'     : pd_kp2d[:self.n_samples].detach().clone(),
                        'pd_verts'    : skel_output.skin_verts[:self.n_samples].detach().clone(),
                        'cam_t'       : inputs['cam_t'][:self.n_samples].detach().clone(),
                        'optim_betas' : inputs['betas'][:self.n_samples].detach().clone(),
                        'kp2d_err'    : kp2d_err[:self.n_samples].detach().clone(),
                    })

        with PM.time_monitor('backwards'):
            loss.backward()
        return loss.item()

    return closure


def compute_losses(
    loss_cfg  : Dict[str, Union[bool, float]],
    parts     : List[str],
    gt_kp2d   : torch.Tensor,
    pd_kp2d   : torch.Tensor,
    pd_params : Dict[str, torch.Tensor],
    pd_cam_z  : torch.Tensor,
    gs_cam_z  : Optional[torch.Tensor] = None,
):
    '''
    ### Args
    - loss_cfg: Dict[str, Union[bool, float]]
        - Special option flags (`f_xxx`) or loss weights (`w_xxx`).
    - parts: List[str]
        - The list of the active joint parts groups. 
        - Among {'all', 'torso', 'torso-lite', 'limbs', 'head', 'limbs_proximal', 'limbs_distal'}.
    - gt_kp2d: torch.Tensor (B, 44, 3)
        - The ground-truth 2D keypoints with confidence.
    - pd_kp2d: torch.Tensor (B, 44, 2)
        - The predicted 2D keypoints.
    - pd_params: Dict[str, torch.Tensor]
        - poses: torch.Tensor (B, 46)
        - betas: torch.Tensor (B, 10)
    - pd_cam_z: torch.Tensor (B,)
        - The predicted camera depth translation.
    - gs_cam_z: Optional[torch.Tensor] (B,)
        - The guessed camera depth translation.
        
    ### Returns
    - loss: torch.Tensor (,)
        - The weighted loss value for optimization.
    - losses: Dict[str, float]
        - The dictionary of the loss values for logging.
    '''

    losses = {}
    loss = torch.tensor(0.0, device=gt_kp2d.device)
    kp2d_conf = gt_kp2d[:, :, 2]  # (B, J)
    gt_kp2d = gt_kp2d[:, :, :2]  # (B, J, 2)

    # Special option flags.
    f_normalize_kp2d = loss_cfg.get('f_normalize_kp2d', False)

    if f_normalize_kp2d:
        scale2mean = loss_cfg.get('f_normalize_kp2d_to_mean', False)
        scale2d = estimate_kp2d_scale(gt_kp2d)  # (B,)
        pd_kp2d = pd_kp2d / (scale2d[:, None, None] + 1e-6)  # (B, J, 2)
        gt_kp2d = gt_kp2d / (scale2d[:, None, None] + 1e-6)  # (B, J, 2)
        if scale2mean:
            scale2d_mean = scale2d.mean()
            pd_kp2d = pd_kp2d * scale2d_mean  # (B, J, 2)
            gt_kp2d = gt_kp2d * scale2d_mean  # (B, J, 2)

    # Mask the keypoints.
    act_jids = get_kp_active_jids(parts)
    kp2d_conf = kp2d_conf[:, act_jids]  # (B, J)
    gt_kp2d = gt_kp2d[:, act_jids, :]  # (B, J, 2)
    pd_kp2d = pd_kp2d[:, act_jids, :]  # (B, J, 2)

    # Calculate weighted losses.
    w_depth = loss_cfg.get('w_depth', None)
    w_reprojection = loss_cfg.get('w_reprojection', None)
    w_shape_prior = loss_cfg.get('w_shape_prior', None)
    w_angle_prior = loss_cfg.get('w_angle_prior', None)

    if w_depth:
        assert gs_cam_z is not None, 'The guessed camera depth is required for the depth loss.'
        depth_loss = (gs_cam_z - pd_cam_z).pow(2)  # (B,)
        loss += (w_depth ** 2) * depth_loss.mean()  # (,)
        losses['depth'] = (w_depth ** 2) * depth_loss.mean().item()  # float

    if w_reprojection:
        reproj_err_j = gmof(pd_kp2d - gt_kp2d).sum(dim=-1) # (B, J)
        reproj_err_j = kp2d_conf.pow(2) * reproj_err_j  # (B, J)
        reproj_loss = reproj_err_j.sum(-1)  # (B,)
        loss += (w_reprojection ** 2) * reproj_loss.mean()  # (,)
        losses['reprojection'] = (w_reprojection ** 2) * reproj_loss.mean().item()  # float

    if w_shape_prior:
        shape_prior_loss = pd_params['betas'].pow(2).sum(dim=-1)  # (B,)
        loss += (w_shape_prior ** 2) * shape_prior_loss.mean()  # (,)
        losses['shape_prior'] = (w_shape_prior ** 2) * shape_prior_loss.mean().item()  # float

    if w_angle_prior:
        w_angle_prior *= loss_cfg.get('w_angle_prior_scale', 1.0)
        angle_prior_loss = compute_poses_angle_prior_loss(pd_params['poses'])  # (B,)
        loss += (w_angle_prior ** 2) * angle_prior_loss.mean()  # (,)
        losses['angle_prior'] = (w_angle_prior ** 2) * angle_prior_loss.mean().item()  # float

    losses['weighted_sum'] = loss.item()  # float
    return loss, losses