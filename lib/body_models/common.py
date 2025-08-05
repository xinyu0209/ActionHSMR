from lib.kits.basic import *

from smplx import SMPL

from lib.platform import PM
from lib.body_models.skel_wrapper import SKELWrapper as SKEL
from lib.body_models.smpl_wrapper import SMPLWrapper

def make_SMPL(gender='neutral', device='cuda:0'):
    return SMPL(
        gender = gender,
        model_path = PM.inputs / 'body_models' / 'smpl',
    ).to(device)


def make_SMPL_hmr2(gender='neutral', device='cuda:0'):
    ''' SKEL doesn't have neutral model, so align with SKEL, using male. '''
    return SMPLWrapper(
        gender = gender,
        model_path = PM.inputs / 'body_models' / 'smpl',
        num_body_joints = 23,
        joint_regressor_extra = PM.inputs / 'body_models/SMPL_to_J19.pkl',
    ).to(device)



def make_SKEL(gender='male', device='cuda:0'):
    ''' We don't have neutral model for SKEL, so use male for now. '''
    return make_SKEL_mix_joints(gender, device)


def make_SKEL_smpl_joints(gender='male', device='cuda:0'):
    ''' We don't have neutral model for SKEL, so use male for now. '''
    return SKEL(
        gender = gender,
        model_path = PM.inputs / 'body_models' / 'skel',
        joint_regressor_extra = PM.inputs / 'body_models' / 'SMPL_to_J19.pkl',
        joint_regressor_custom = PM.inputs / 'body_models' / 'J_regressor_SMPL_MALE.pkl',
    ).to(device)


def make_SKEL_skel_joints(gender='male', device='cuda:0'):
    ''' We don't have neutral model for SKEL, so use male for now. '''
    return SKEL(
        gender = gender,
        model_path = PM.inputs / 'body_models' / 'skel',
        joint_regressor_extra = PM.inputs / 'body_models' / 'SMPL_to_J19.pkl',
    ).to(device)


def make_SKEL_mix_joints(gender='male', device='cuda:0'):
    ''' We don't have neutral model for SKEL, so use male for now. '''
    return SKEL(
        gender = gender,
        model_path = PM.inputs / 'body_models' / 'skel',
        joint_regressor_extra = PM.inputs / 'body_models' / 'SMPL_to_J19.pkl',
        joint_regressor_custom = PM.inputs / 'body_models' / 'J_regressor_SKEL_mix_MALE.pkl',
    ).to(device)


def make_SMPLX_moyo(v_template_path:Union[str, Path], batch_size:int=1, device='cuda:0'):
    from lib.body_models.moyo_smplx_wrapper import MoYoSMPLX

    return MoYoSMPLX(
        model_path      = PM.inputs / 'body_models' / 'smplx',
        v_template_path = v_template_path,
        batch_size      = batch_size,
        device          = device,
    )