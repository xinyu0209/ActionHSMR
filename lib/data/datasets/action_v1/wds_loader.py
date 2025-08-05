from lib.kits.basic import *
import webdataset as wds
from .utils import *
from .stream_pipelines import *
import itertools
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch


def load_tars_as_wds(
        cfg: DictConfig,
        urls: Union[str, List[str]],
        resampled: bool = False,
        epoch_size: int = None,
        cache_dir: str = None,
        train: bool = True,
):
    urls = expand_urls(urls)

    # 创建基础数据集管道，顺序读取图像
    dataset = wds.WebDataset(
        urls,
        nodesplitter=wds.split_by_node,
        shardshuffle=False,
        resampled=resampled,
        cache_dir=cache_dir,
    )

    # 构建流式处理管道
    dataset = apply_corrupt_filter(dataset)
    dataset = dataset.decode('rgb8').rename(jpg='jpg;jpeg;png')
    dataset = apply_multi_ppl_splitter(dataset)
    dataset = apply_keys_adapter(dataset)
    dataset = apply_bad_pgt_params_nan_suppressor(dataset)
    dataset = apply_bad_pgt_params_kp2d_err_suppressor(dataset, cfg.get('suppress_pgt_params_kp2d_err_thresh', 0.0))
    dataset = apply_bad_pgt_params_pve_max_suppressor(dataset, cfg.get('suppress_pgt_params_pve_max_thresh', 0.0))
    dataset = apply_bad_kp_suppressor(dataset, cfg.get('suppress_kp_conf_thresh', 0.0))
    dataset = apply_bad_betas_suppressor(dataset, cfg.get('suppress_betas_thresh', 0.0))
    dataset = apply_params_synchronizer(dataset, cfg.get('poses_betas_simultaneous', False))
    dataset = apply_insuff_kp_filter(
        dataset,
        cfg.get('filter_insufficient_kp_cnt', 4),
        cfg.get('suppress_insufficient_kp_thresh', 0.0)
    )
    dataset = apply_bbox_size_filter(dataset, cfg.get('filter_bbox_size_thresh', None))
    dataset = apply_reproj_err_filter(dataset, cfg.get('filter_reproj_err_thresh', 0.0))
    dataset = apply_invalid_betas_regularizer(dataset, cfg.get('regularize_invalid_betas', False))
    dataset = apply_three_frame_sequence(dataset)

    # 格式化处理
    dataset = apply_example_formatter(dataset, cfg)

    if epoch_size is not None:
        dataset = dataset.with_epoch(epoch_size)

    for i, sample in enumerate(itertools.islice(dataset, 2)):
        print(f"[Check] Sample {i} kp2d min/max: {sample['kp2d'][:, :2].min()}, {sample['kp2d'][:, :2].max()}")
        print(f"[Check] Sample {i} kp2d (first 5):\n", sample['kp2d'][:5])
        if i > 0: break
            
    return dataset