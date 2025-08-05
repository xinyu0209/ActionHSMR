from lib.kits.basic import *

from torch.utils import data
from lib.utils.media import load_img, flex_resize_img
from lib.utils.bbox import cwh_to_cs, cs_to_lurb, crop_with_lurb, fit_bbox_to_aspect_ratio
from lib.utils.data import to_numpy

IMG_MEAN_255 = to_numpy([0.485, 0.456, 0.406]) * 255.
IMG_STD_255  = to_numpy([0.229, 0.224, 0.225]) * 255.


class Eval3DDataset(data.Dataset):

    def __init__(self, npz_fn:Union[str, Path], ignore_img=False):
        super().__init__()
        self.data = None
        self._load_data(npz_fn)
        self.ds_root = self._get_ds_root()
        self.bbox_ratio = (192, 256)  # the ViT Backbone's input size is w=192, h=256
        self.ignore_img = ignore_img  # For speed up, if True, don't process images.

    def _load_data(self, npz_fn:Union[str, Path]):
        supported_datasets = ['MoYo']
        raw_data = np.load(npz_fn, allow_pickle=True)

        # Load some meta data.
        self.extra_info = raw_data['extra_info'].item()
        self.ds_name    = self.extra_info.pop('dataset_name')
        # Load basic information.
        self.seq_names    = raw_data['names']  # (L,)
        self.img_paths    = raw_data['img_paths']  # (L,)
        self.bbox_centers = raw_data['centers'].astype(np.float32)  # (L, 2)
        self.bbox_scales  = raw_data['scales'].astype(np.float32)  # (L, 2)
        self.L = len(self.seq_names)
        # Load the g.t. SMPL parameters.
        self.genders      = raw_data.get('genders', None)  # (L, 2) or None
        self.global_orient = raw_data['smpl'].item()['global_orient'].reshape(-1, 1 ,3).astype(np.float32)  # (L, 1, 3)
        self.body_pose     = raw_data['smpl'].item()['body_pose'].reshape(-1, 23 ,3).astype(np.float32)  # (L, 23, 3)
        self.betas         = raw_data['smpl'].item()['betas'].reshape(-1, 10).astype(np.float32)  # (L, 10)
        # Check validity.
        assert self.ds_name in supported_datasets, f'Unsupported dataset: {self.ds_name}'


    def __len__(self):
        return self.L


    def _process_img_patch(self, idx):
        ''' Load and crop according to bbox. '''
        if self.ignore_img:
            return np.zeros((1), dtype=np.float32)

        img, _ = load_img(self.ds_root / self.img_paths[idx])  # (H, W, RGB)
        scale = self.bbox_scales[idx]  # (2,)
        center = self.bbox_centers[idx]  # (2,)
        bbox_cwh = np.concatenate([center, scale], axis=0)  # (4,) lurb format
        bbox_cwh = fit_bbox_to_aspect_ratio(
                bbox      = bbox_cwh,
                tgt_ratio = self.bbox_ratio,
                bbox_type = 'cwh'
            )
        bbox_cs = cwh_to_cs(bbox_cwh, reduce='max')  # (3,), make it to square
        bbox_lurb = cs_to_lurb(bbox_cs)  # (4,)
        img_patch = crop_with_lurb(img, bbox_lurb)  # (H', W', RGB)
        img_patch = flex_resize_img(img_patch, tgt_wh=(256, 256))
        img_patch_normalized = (img_patch - IMG_MEAN_255) / IMG_STD_255  # (H', W', RGB)
        img_patch_normalized = img_patch_normalized.transpose(2, 0, 1)  # (RGB, H', W')
        return img_patch_normalized.astype(np.float32)


    def _get_ds_root(self):
        return PM.inputs / 'datasets' / self.ds_name.lower()


    def __getitem__(self, idx):
        ret = {}
        ret['seq_name'] = self.seq_names[idx]
        ret['smpl'] = {
                'global_orient': self.global_orient[idx],
                'body_pose'    : self.body_pose[idx],
                'betas'        : self.betas[idx],
            }
        if self.genders is not None:
            ret['gender'] = self.genders[idx]
        ret['img_patch'] = self._process_img_patch(idx)

        return ret

