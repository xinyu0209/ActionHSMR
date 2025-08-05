from lib.kits.basic import *


class MoCapDataset:
    def __init__(self, dataset_file:str, pve_threshold:Optional[float]=None):
        '''
        Dataset class used for loading a dataset of unpaired SMPL parameter annotations
        ### Args
        - dataset_file: str
            - Path to the dataset npz file.
        - pve_threshold: float
            - Threshold for PVE quality filtering.
        '''
        data = np.load(dataset_file)
        if pve_threshold is not None:
            pve = data['pve_max']
            mask = pve < pve_threshold
        else:
            mask = np.ones(len(data['poses']), dtype=np.bool)
        self.poses = data['poses'].astype(np.float32)[mask, 3:]
        self.betas = data['betas'].astype(np.float32)[mask]
        self.length = len(self.poses)
        get_logger().info(f'Loaded {self.length} items among {len(pve)} samples filtered from {dataset_file} (using threshold = {pve_threshold})')

    def __getitem__(self, idx: int) -> Dict:
        poses = self.poses[idx].copy()
        betas = self.betas[idx].copy()
        item = {'poses_body': poses, 'betas': betas}
        return item

    def __len__(self) -> int:
        return self.length
