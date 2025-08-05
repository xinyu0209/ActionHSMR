import numpy as np
from typing import Dict

class MoCapDataset:

    def __init__(self, dataset_file: str, threshold: float = 0.01) -> None:
        """
        Dataset class used for loading a dataset of unpaired SMPL parameter annotations
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            threshold (float): Threshold for PVE filtering.
        """
        data = np.load(dataset_file)
        # pve = data['pve']
        pve = data['pve_max']
        # pve = data['pve_mean']
        mask = pve < threshold
        self.pose = data['poses'].astype(np.float32)[mask, 3:]
        self.betas = data['betas'].astype(np.float32)[mask]
        self.length = len(self.pose)
        print(f'Loaded {self.length} among {len(pve)} samples from {dataset_file} (using threshold = {threshold})')

    def __getitem__(self, idx: int) -> Dict:
        pose = self.pose[idx].copy()
        betas = self.betas[idx].copy()
        item = {'body_pose': pose, 'betas': betas}
        return item

    def __len__(self) -> int:
        return self.length
