import torch
import numpy as np

from typing import Dict


def fliplr_params(smpl_params: Dict):
    global_orient = smpl_params['global_orient'].copy().reshape(-1, 3)
    body_pose = smpl_params['body_pose'].copy().reshape(-1, 69)
    betas = smpl_params['betas'].copy()

    body_pose_permutation = [6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13,
                             14 ,18, 19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33,
                             34, 35, 30, 31, 32, 36, 37, 38, 42, 43, 44, 39, 40, 41,
                             45, 46, 47, 51, 52, 53, 48, 49, 50, 57, 58, 59, 54, 55,
                             56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66, 67, 68]
    body_pose_permutation = body_pose_permutation[:body_pose.shape[1]]
    body_pose_permutation = [i-3 for i in body_pose_permutation]

    body_pose = body_pose[:, body_pose_permutation]

    global_orient[:, 1::3] *= -1
    global_orient[:, 2::3] *= -1
    body_pose[:, 1::3] *= -1
    body_pose[:, 2::3] *= -1

    smpl_params = {'global_orient': global_orient.reshape(-1, 1, 3).astype(np.float32),
                   'body_pose': body_pose.reshape(-1, 23, 3).astype(np.float32),
                   'betas': betas.astype(np.float32)
                  }

    return smpl_params

