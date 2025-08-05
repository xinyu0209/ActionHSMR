from lib.kits.basic import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from omegaconf import OmegaConf

from lib.platform import PM
from lib.body_models.skel_utils.transforms import params_q2rep, params_rep2q

from .utils.pose_transformer import TransformerDecoder


class SKELTransformerDecoderHead(nn.Module):
    """ Cross-attention based SKEL Transformer decoder
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.pd_poses_repr == 'rotation_6d':
            n_poses = 24 * 6
        elif cfg.pd_poses_repr == 'euler_angle':
            n_poses = 46
        else:
            raise ValueError(f"Unknown pose representation: {cfg.pd_poses_repr}")

        n_betas = 10
        n_cam   = 3
        self.input_is_mean_shape = False

        # Build transformer decoder.
        transformer_args = {
                'num_tokens' : 1,
                'token_dim'  : (n_poses + n_betas + n_cam) if self.input_is_mean_shape else 1,
                'dim'        : 1024,
            }
        transformer_args.update(OmegaConf.to_container(cfg.transformer_decoder, resolve=True))  # type: ignore
        self.transformer = TransformerDecoder(**transformer_args)

        # Build decoders for parameters.
        dim = transformer_args['dim']
        self.poses_decoder = nn.Linear(dim, n_poses)
        self.betas_decoder = nn.Linear(dim, n_betas)
        self.cam_decoder   = nn.Linear(dim, n_cam)

        # Load mean shape parameters as initial values.
        skel_mean_path = Path(__file__).parent / 'SKEL_mean.npz'
        skel_mean_params = np.load(skel_mean_path)

        init_poses = torch.from_numpy(skel_mean_params['poses'].astype(np.float32)).unsqueeze(0) # (1, 46)
        if cfg.pd_poses_repr == 'rotation_6d':
            init_poses = params_q2rep(init_poses).reshape(1, 24*6)  # (1, 24*6)
        init_betas = torch.from_numpy(skel_mean_params['betas'].astype(np.float32)).unsqueeze(0)
        init_cam = torch.from_numpy(skel_mean_params['cam'].astype(np.float32)).unsqueeze(0)

        self.register_buffer('init_poses', init_poses)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, **kwargs):
        B = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        # Initialize the parameters.
        init_poses = self.init_poses.expand(B, -1)  # (B, 46)
        init_betas = self.init_betas.expand(B, -1)  # (B, 10)
        init_cam   = self.init_cam.expand(B, -1)    # (B, 3)

        # Input token to transformer is zero token.
        with PM.time_monitor('init_token'):
            if self.input_is_mean_shape:
                token = torch.cat([init_poses, init_betas, init_cam], dim=1)[:, None, :]  # (B, 1, C)
            else:
                token = x.new_zeros(B, 1, 1)

        # Pass through transformer.
        with PM.time_monitor('transformer'):
            token_out = self.transformer(token, context=x)
            token_out = token_out.squeeze(1)  # (B, C)

        # Parse the SKEL parameters out from token_out.
        with PM.time_monitor('decode'):
            pd_poses = self.poses_decoder(token_out) + init_poses
            pd_betas = self.betas_decoder(token_out) + init_betas
            pd_cam = self.cam_decoder(token_out) + init_cam

        with PM.time_monitor('rot_repr_transform'):
            if self.cfg.pd_poses_repr == 'rotation_6d':
                pd_poses = params_rep2q(pd_poses.reshape(-1, 24, 6))  # (B, 46)
            elif self.cfg.pd_poses_repr == 'euler_angle':
                pd_poses = pd_poses.reshape(-1, 46)  # (B, 46)
            else:
                raise ValueError(f"Unknown pose representation: {self.cfg.pd_poses_repr}")

        pd_skel_params = {
                'poses'        : pd_poses,
                'poses_orient' : pd_poses[:, :3],
                'poses_body'   : pd_poses[:, 3:],
                'betas'        : pd_betas
            }
        return pd_skel_params, pd_cam