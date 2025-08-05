import torch

from typing import Optional

from .transforms import real_orient_mat2q, real_orient_q2mat


def update_params_after_orient_rotation(
    poses       : torch.Tensor,           # (B, 46)
    rot_mat     : torch.Tensor,           # the rotation orientation matrix
    root_offset : Optional[torch.Tensor] = None, # the offset from custom root to model root
):
    '''
    
    ### Args
    - `poses`: torch.Tensor, shape = (B, 46)
    - `rot_mat`: torch.Tensor, shape = (B, 3, 3)
    - `root_offset`: torch.Tensor or None, shape = (B, 3)
        - If None, the function won't update the translation.
        - If not None, the function will calculate the root translation offset that make the model 
           rotate around the custom root instead of the model root.
           
    ### Returns
    - If `root_offset` is None:
        - `poses`: torch.Tensor, shape = (B, 46)
    - If `root_offset` is not None:
        - `poses`: torch.Tensor, shape = (B, 46)
        - `trans_offset`: torch.Tensor, shape = (B, 3)
    '''
    poses = poses.clone()
    # 1. Transform the SKEL orientation to real matrix.
    orient_q = poses[:, :3]  # (B, 3)
    orient_mat = real_orient_q2mat(orient_q)  # (B, 3, 3)
    orient_mat = torch.einsum('bij,bjk->bik', rot_mat, orient_mat)  # (B, 3, 3)
    orient_q = real_orient_mat2q(orient_mat)  # (B, 3)
    poses[:, :3] = orient_q

    # 2. Update the translation if needed.
    if root_offset is not None:
        root_before = root_offset.clone()  # (B, 3)
        root_after = torch.einsum('bij,bj->bi', rot_mat, root_before)  # (B, 3)
        root_offset = root_after - root_before  # (B, 3)
        ret = poses, root_offset
    else:
        ret = poses

    return ret