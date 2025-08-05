from lib.kits.basic import *

import rich
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from lib.version import DEFAULT_HSMR_ROOT
from lib.data.datasets.eval_3d.eval3d_dataset import Eval3DDataset
from lib.utils.device import recursive_to
from lib.evaluation import HSMR3DEvaluator, SKELRealityEvaluator
from lib.evaluation.hmr2_utils import Evaluator as HMR2Evaluator
from lib.body_models.common import make_SMPL, make_SKEL_smpl_joints


BATCH_SIZE = 300

DS_NAME2NPZ_FN = {
    'MOYO': PM.inputs / 'evaluation_data' / 'moyo_v2.npz',
}


# ================== Command Line Supports ==================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-r', '--exp_root', type=str, default=DEFAULT_HSMR_ROOT)
    parser.add_argument('-s', '--dataset', type=str, required=True, choices=['H36M-VAL-P2', '3DPW-TEST', 'LSP-EXTENDED', 'POSETRACK-VAL', 'COCO-VAL', 'MOYO'])
    args = parser.parse_args()
    return args


# ================== Data Tools ==================

def get_data(ds_name, cfg):
    print('cfg.data:', cfg.data)
    data = instantiate(cfg.data, _recursive_=False)
    print('data:', data)
    data._setup_eval(selected_ds_names=[ds_name])
    print('eval_datasets:', data.eval_datasets)
    print('eval_datasets keys:', data.eval_datasets.keys())
    dataset = data.eval_datasets[ds_name]

    assert ds_name in ['H36M-VAL-P2', '3DPW-TEST', 'LSP-EXTENDED', 'POSETRACK-VAL', 'COCO-VAL', 'MOYO']
    if ds_name in ['MOYO']:
        dataset = Eval3DDataset(DS_NAME2NPZ_FN[ds_name])
    elif ds_name in ['H36M-VAL-P2', '3DPW-TEST', 'LSP-EXTENDED', 'POSETRACK-VAL', 'COCO-VAL']:
        data = instantiate(cfg.data, _recursive_=False)
        data._setup_eval(selected_ds_names=[ds_name])
        print("已注册的 eval_datasets key:", data.eval_datasets.keys())
        print("你要找的 ds_name:", ds_name)
        dataset = data.eval_datasets[ds_name]
        data._setup_eval(selected_ds_names=[ds_name])


    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return {
        'name': ds_name,
        'dataset': dataset,
        'data_loader': data_loader,
    }



# ================== Body Models Tools ==================

def _reset_root(v3d, j3d):
    ''' Reset the root position to the origin. '''
    r3d = j3d[:, [0]]  # (B, 1, 3)
    v3d -= r3d  # (B, V, 3)
    j3d -= r3d  # (B, J, 3)
    return v3d, j3d


def _forward_gt_smpl24(gt):
    ''' Inference SMPL parameters to standard SMPL vertices and joints(24). '''
    # Lazily create body_model.
    if not hasattr(_forward_gt_smpl24, 'smpl_model'):
        _forward_gt_smpl24.smpl_model = make_SMPL(gender='neutral', device=gt['smpl']['betas'].device)
    smpl_model = _forward_gt_smpl24.smpl_model
    # SMPL inference.
    smpl_output = smpl_model(**gt['smpl'])
    gt_v = smpl_output.vertices
    gt_j = smpl_output.joints[:, :24]  # (B, J=24, 3)
    return _reset_root(gt_v, gt_j)


def _forward_pd_smpl24(pd):
    ''' Inference SKEL parameters to standard SMPL vertices and joints(24). '''
    # Lazily create body_model.
    if not hasattr(_forward_pd_smpl24, 'skel_model'):
        _forward_pd_smpl24.skel_model = make_SKEL_smpl_joints(device=pd['pd_params']['betas'].device)
    skel_model = _forward_pd_smpl24.skel_model
    # SKEL inference.
    skel_output = skel_model(**pd['pd_params'], skelmesh=False)
    pd_v = skel_output.skin_verts  # (B, V=6890, 3)
    pd_j = skel_output.joints_custom  # (B, J=24, 3)
    return _reset_root(pd_v, pd_j)

#  ================== Evaluator Tools ==================

class UniformEvaluator():
    MODE_STD = 'std'
    MODE_EXT = 'ext'


    def __init__(self, data, device='cuda:0'):
        ''' Determine which evaluator to use. '''
        self.device = device
        if data['name'] in ['MOYO']:
            self.mode = self.MODE_EXT
            self.accuracy_ext = HSMR3DEvaluator()
            self.reality = SKELRealityEvaluator()
        else:
            self.mode = self.MODE_STD
            if data['name'] in ['H36M-VAL-P2','3DPW-TEST']:
                metrics = ['mode_re', 'mode_mpjpe']
                pck_thresholds = None
            elif data['name'] in ['LSP-EXTENDED', 'POSETRACK-VAL', 'COCO-VAL']:
                metrics = ['mode_kpl2']
                pck_thresholds = [0.05, 0.1]

            self.accuracy_std = HMR2Evaluator(
                dataset_length = int(1e8),
                keypoint_list  = data['dataset']._kp_list_,
                pelvis_ind     = 39,
                metrics        = metrics,
                pck_thresholds = pck_thresholds,
            )


    def eval(self, pd, gt):
        ''' Uniform evaluation interface. '''
        if self.mode == self.MODE_EXT:
            pd_v, pd_j = _forward_pd_smpl24(pd)
            gt_v, gt_j = _forward_gt_smpl24(gt)
            self.accuracy_ext.eval(
                pd = {'v3d_pose': pd_v, 'j3d_pose': pd_j},
                gt = {'v3d_pose': gt_v, 'j3d_pose': gt_j},
            )
            self.reality.eval(
                pd = {'poses': pd['pd_params']['poses']},
            )
        elif self.mode == self.MODE_STD:
            self.accuracy_std(pd, gt)


    def get_results(self):
        ''' Uniform results interface. '''
        if self.mode == self.MODE_EXT:
            results = {
                **self.accuracy_ext.get_results(),
                **self.reality.get_results(),
            }
        elif self.mode == self.MODE_STD:
            results = self.accuracy_std.get_metrics_dict()
        return results
