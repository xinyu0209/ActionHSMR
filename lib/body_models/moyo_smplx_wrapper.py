from lib.kits.basic import *

import smplx
from psbody.mesh import Mesh

class MoYoSMPLX(smplx.SMPLX):

    def __init__(
        self,
        model_path      : Union[str, Path],
        v_template_path : Union[str, Path],
        batch_size = 1,
        n_betas    = 10,
        device     = 'cpu'
    ):

        if isinstance(v_template_path, Path):
            v_template_path = str(v_template_path)

        # Load the `v_template`.
        v_template_mesh = Mesh(filename=v_template_path)
        v_template = to_tensor(v_template_mesh.v, device=device)

        self.n_betas = n_betas

        # Create the `body_model_params`.
        body_model_params = {
                'model_path'             : model_path,
                'gender'                 : 'neutral',
                'v_template'             : v_template.float(),
                'batch_size'             : batch_size,
                'create_global_orient'   : True,
                'create_body_pose'       : True,
                'create_betas'           : True,
                'num_betas'              : self.n_betas,  # They actually don't use num_betas.
                'create_left_hand_pose'  : True,
                'create_right_hand_pose' : True,
                'create_expression'      : True,
                'create_jaw_pose'        : True,
                'create_leye_pose'       : True,
                'create_reye_pose'       : True,
                'create_transl'          : True,
                'use_pca'                : False,
                'flat_hand_mean'         : True,
                'dtype'                  : torch.float32,
            }

        super().__init__(**body_model_params)
        self = self.to(device)

    def forward(self, **kwargs):
        ''' Only all parameters are passed, the batch_size will be flexible adjusted. '''
        assert 'global_orient' in kwargs, '`global_orient` is required for the forward pass.'
        assert 'body_pose' in kwargs, '`body_pose` is required for the forward pass.'
        B = kwargs['global_orient'].shape[0]
        body_pose = kwargs['body_pose']

        if 'left_hand_pose' not in kwargs:
            kwargs['left_hand_pose'] = body_pose.new_zeros((B, 45))
            get_logger().warning('`left_hand_pose` is not provided, but it\'s expected, set to zeros.')
        if 'right_hand_pose' not in kwargs:
            kwargs['right_hand_pose'] = body_pose.new_zeros((B, 45))
            get_logger().warning('`left_hand_pose` is not provided, but it\'s expected, set to zeros.')
        if 'transl' not in kwargs:
            kwargs['transl'] = body_pose.new_zeros((B, 3))
        if 'betas' not in kwargs:
            kwargs['betas'] = body_pose.new_zeros((B, self.n_betas))
        if 'expression' not in kwargs:
            kwargs['expression'] = body_pose.new_zeros((B, 10))
        if 'jaw_pose' not in kwargs:
            kwargs['jaw_pose'] = body_pose.new_zeros((B, 3))
        if 'leye_pose' not in kwargs:
            kwargs['leye_pose'] = body_pose.new_zeros((B, 3))
        if 'reye_pose' not in kwargs:
            kwargs['reye_pose'] = body_pose.new_zeros((B, 3))

        return super().forward(**kwargs)