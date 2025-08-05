from lib.kits.basic import *

import webdataset as wds


from lib.data.datasets.skel_hmr2_fashion.image_dataset import ImageDataset


class MixedWebDataset(wds.WebDataset):
    def __init__(self) -> None:
        super(wds.WebDataset, self).__init__()


class DataModule(pl.LightningDataModule):
    def __init__(self, name:str, cfg:DictConfig):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.cfg_eval = self.cfg.pop('eval', None)
        self.cfg_train = self.cfg.pop('train', None)


    def setup(self, stage=None):
        if stage in ['test', None, '_debug_eval']:
            self._setup_eval()

        if stage in ['fit', None, '_debug_train']:
            self._setup_train()


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset = self.train_dataset,
            **self.cfg_train.dataloader,
        )


    def val_dataloader(self):
        # Since we don't need validation here.
        return self.test_dataloader()


    def test_dataloader(self):
        # return torch.utils.data.DataLoader(
        #     dataset = self.eval_datasets['LSP-EXTENDED'],  # TODO: Support multiple datasets through ConcatDataset (but to figure out how to mix with weights)
        #     **self.cfg_eval.dataloader,
        # )
        return torch.utils.data.DataLoader(
            dataset = self.eval_datasets,  # TODO: Support multiple datasets through ConcatDataset (but to figure out how to mix with weights)
            **self.cfg_eval.dataloader,
        )


    # ========== Internal Modules to Setup Datasets ==========

    def _setup_train(self):
        hack_cfg = {
                'IMAGE_SIZE': self.cfg.policy.img_patch_size,
                'IMAGE_MEAN': self.cfg.policy.img_mean,
                'IMAGE_STD' : self.cfg.policy.img_std,
                'BBOX_SHAPE': None,
                'augm': self.cfg.augm,
            }

        self.train_datasets = []  # [(dataset:Dataset, weight:float), ...]
        datasets, weights = [], []
        opt = self.cfg_train.get('shared_ds_opt', {})
        for dataset_cfg in self.cfg_train.datasets:
            cur_cfg = {**hack_cfg, **opt}
            dataset = ImageDataset.load_tars_as_webdataset(
                    cfg        = cur_cfg,
                    urls       = dataset_cfg.item.urls,
                    train      = True,
                    epoch_size = dataset_cfg.item.epoch_size,
                )
            weights.append(dataset_cfg.weight)
            datasets.append(dataset)
        weights = to_numpy(weights)
        weights = weights / weights.sum()
        self.train_dataset = MixedWebDataset()
        self.train_dataset.append(wds.RandomMix(datasets, weights, longest=False))
        self.train_dataset = self.train_dataset.with_epoch(100_000).shuffle(4000)


    def _setup_eval(self):
        hack_cfg = {
                'IMAGE_SIZE' : self.cfg.policy.img_patch_size,
                'IMAGE_MEAN' : self.cfg.policy.img_mean,
                'IMAGE_STD'  : self.cfg.policy.img_std,
                'BBOX_SHAPE' : [192, 256],
                'augm'       : self.cfg.augm,
            }

        self.eval_datasets = {}
        opt = self.cfg_train.get('shared_ds_opt', {})
        for dataset_cfg in self.cfg_eval.datasets:
            cur_cfg = {**hack_cfg, **opt}
            dataset = ImageDataset(
                    cfg          = hack_cfg,
                    dataset_file = dataset_cfg.item.dataset_file,
                    img_dir      = dataset_cfg.item.img_root,
                    train        = False,
                )
            dataset._kp_list_ = dataset_cfg.item.kp_list
            self.eval_datasets[dataset_cfg.name] = dataset

