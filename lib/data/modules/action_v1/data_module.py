from lib.kits.basic import *

from lib.data.datasets.action_v1.mocap_dataset import MoCapDataset
from lib.data.datasets.action_v1.wds_loader import load_tars_as_wds
import webdataset as wds

class MixedWebDataset(wds.WebDataset):
    def __init__(self) -> None:
        super(wds.WebDataset, self).__init__()


class DataModule(pl.LightningDataModule):
    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.cfg_eval = self.cfg.pop('eval', None)
        self.cfg_train = self.cfg.pop('train', None)
        self.cfg_mocap = self.cfg.pop('mocap', None)


    def setup(self, stage=None):
        if stage in ['test', None, '_debug_eval'] and self.cfg_eval is not None:
            # get_logger().info('Evaluation dataset will be enabled.')
            self._setup_eval()

        if stage in ['fit', None, '_debug_train'] and self.cfg_train is not None:
            # get_logger().info('Training dataset will be enabled.')
            self._setup_train()

        if stage in ['fit', None, '_debug_mocap'] and self.cfg_mocap is not None:
            # get_logger().info('Mocap dataset will be enabled.')
            self._setup_mocap()


    def train_dataloader(self):
        img_dataset = torch.utils.data.DataLoader(
                dataset = self.train_dataset,
                **self.cfg_train.dataloader,
            )
        ret = {'img_ds' : img_dataset }

        if self.cfg_mocap is not None:
            mocap_dataset = torch.utils.data.DataLoader(
                    dataset = self.mocap_dataset,
                    **self.cfg_mocap.dataloader,
                )
            ret['mocap_ds'] = mocap_dataset
        for b in img_dataset:
            print("batch[img_patch] shape:", b['img_patch'].shape)
            break

        return ret

    # ========== Internal Modules to Setup Datasets ==========


    def _setup_train(self):
        names, datasets, weights = [], [], []
        ld_cfg = self.cfg_train.cfg  # cfg for initializing wds loading pipeline

        for ds_cfg in self.cfg_train.datasets:
            dataset = load_tars_as_wds(
                    ld_cfg,
                    ds_cfg.item.urls,
                    ds_cfg.item.epoch_size
                ) 

            names.append(ds_cfg.name)
            datasets.append(dataset)
            weights.append(ds_cfg.weight)

            # get_logger().info(f"Dataset '{ds_cfg.name}' loaded.")

        # Normalize the weights and mix the datasets.
        weights = to_numpy(weights)
        weights = weights / weights.sum()
        self.train_datasets = datasets
        self.train_dataset = MixedWebDataset()
        self.train_dataset.append(wds.RandomMix(datasets, weights))
        self.train_dataset = self.train_dataset.with_epoch(50_000).shuffle(1000, initial=1000)


    def _setup_mocap(self):
        self.mocap_dataset = MoCapDataset(**self.cfg_mocap.cfg)


    def _setup_eval(self, selected_ds_names: Optional[List[str]] = None):
        from lib.data.datasets.skel_hmr2_fashion.image_dataset import ImageDataset
        from lib.data.datasets.eval_3d.eval3d_dataset import Eval3DDataset
        hack_cfg = {
                'IMAGE_SIZE'              : 256,
                'IMAGE_MEAN'              : [0.485, 0.456, 0.406],
                'IMAGE_STD'               : [0.229, 0.224, 0.225],
                'BBOX_SHAPE'              : [192, 256],
                'augm'                    : self.cfg.image_augmentation,
                'SUPPRESS_KP_CONF_THRESH' : 0.3,
                'FILTER_NUM_KP'           : 4,
                'FILTER_NUM_KP_THRESH'    : 0.0,
                'FILTER_REPROJ_THRESH'    : 31000,
                'SUPPRESS_BETAS_THRESH'   : 3.0,
                'SUPPRESS_BAD_POSES'      : False,
                'POSES_BETAS_SIMULTANEOUS': True,
                'FILTER_NO_POSES'         : False,
                'BETAS_REG'               : True,
            }

        self.eval_datasets = {}
        for dataset_cfg in self.cfg_eval.datasets:
            if selected_ds_names is not None and dataset_cfg.name not in selected_ds_names:
                continue

            if dataset_cfg.name == "MOYO":
                dataset = Eval3DDataset(
                    npz_fn=dataset_cfg.item.dataset_file,  # 👈 只取出 .npz 路径
                    ignore_img=False
                )
            else:
                dataset = ImageDataset(
                    cfg=hack_cfg,
                    dataset_file=dataset_cfg.item.dataset_file,
                    img_dir=dataset_cfg.item.img_root,
                    train=False,
                )
                dataset._kp_list_ = dataset_cfg.item.kp_list

            self.eval_datasets[dataset_cfg.name] = dataset
            self.eval_datasets[dataset_cfg.name] = dataset