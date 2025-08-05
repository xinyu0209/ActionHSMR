from lib.kits.basic import *

from detectron2.config import LazyConfig
from .utils_detectron2 import DefaultPredictor_Lazy


def build_detector(batch_size, max_img_size, device):
    local_weights_path = Path(__file__).parent / 'model_final_f05665.pkl'
    cfg_path = Path(__file__).parent / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint =str(local_weights_path)
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25

    detector = DefaultPredictor_Lazy(
            cfg          = detectron2_cfg,
            batch_size   = batch_size,
            max_img_size = max_img_size,
            device       = device,
        )
    print(detectron2_cfg)
    print(detector)

    return detector
    
