from lib.kits.basic import *

from tqdm import tqdm

from lib.utils.media import flex_resize_img

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate as instantiate_detectron2
from detectron2.data import MetadataCatalog


class DefaultPredictor_Lazy:
    '''
    Create a simple end-to-end predictor with the given config that runs on single device for a
    several input images.
    Compared to using the model directly, this class does the following additions:

    Modified from: https://github.com/shubham-goel/4D-Humans/blob/6ec79656a23c33237c724742ca2a0ec00b398b53/hmr2/utils/utils_detectron2.py#L9-L93

    1. Load checkpoint from the weights specified in config (cfg.MODEL.WEIGHTS).
    2. Always take BGR image as the input and apply format conversion internally.
    3. Apply resizing defined by the parameter `max_img_size`.
    4. Take input images and produce outputs, and filter out only the `instances` data.
    5. Use an auto-tuned batch size to process the images in a batch.
        - Start with the given batch size, if failed, reduce the batch size by half.
        - If the batch size is reduced to 1 and still failed, skip the image.
        - The implementation is abstracted to `lib.platform.sliding_batches`.
    '''

    def __init__(self, cfg, batch_size=20, max_img_size=512, device='cuda:0'):
        self.batch_size = batch_size
        self.max_img_size = max_img_size
        self.device = device
        self.model = instantiate_detectron2(cfg.model)

        test_dataset = OmegaConf.select(cfg, 'dataloader.test.dataset.names', default=None)
        if isinstance(test_dataset, (List, Tuple)):
            test_dataset = test_dataset[0]

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(OmegaConf.select(cfg, 'train.init_checkpoint', default=''))

        mapper = instantiate_detectron2(cfg.dataloader.test.mapper)
        self.aug = mapper.augmentations
        self.input_format = mapper.image_format

        self.model.eval().to(self.device)
        if test_dataset:
            self.metadata = MetadataCatalog.get(test_dataset)

        assert self.input_format in ['RGB'], f'Invalid input format: {self.input_format}'
        # assert self.input_format in ['RGB', 'BGR'], f'Invalid input format: {self.input_format}'

    def __call__(self, imgs):
        '''
        ### Args
        - `imgs`: List[np.ndarray], a list of image of shape (Hi, Wi, RGB). 
            - Shapes of each image may be different.

        ### Returns
        - `predictions`: dict,
            - the output of the model for one image only.
            - See :doc:`/tutorials/models` for details about the format.
        '''
        with torch.no_grad():
            inputs = []
            downsample_ratios = []
            for img in imgs:
                img_size = max(img.shape[:2])
                if img_size > self.max_img_size:  # exceed the max size, make it smaller
                    downsample_ratio = self.max_img_size / img_size
                    img = flex_resize_img(img, ratio=downsample_ratio)
                    downsample_ratios.append(downsample_ratio)
                else:
                    downsample_ratios.append(1.0)
                h, w, _ = img.shape
                img = self.aug(T.AugInput(img)).apply_image(img)
                img = to_tensor(img.astype('float32').transpose(2, 0, 1), 'cpu')
                inputs.append({'image': img, 'height': h, 'width': w})

            preds = []
            N_imgs = len(inputs)
            prog_bar = tqdm(total=N_imgs, desc='Batch Detection')
            sid, last_fail_id = 0, 0
            cur_bs = self.batch_size
            while sid < N_imgs:
                eid = min(sid + cur_bs, N_imgs)
                try:
                    preds_round = self.model(inputs[sid:eid])
                except Exception as e:
                    get_logger(brief=True).error(f'Image No.{sid}: {e}. Try to fix it.')
                    if cur_bs > 1:
                        cur_bs = (cur_bs - 1) // 2 + 1  # reduce the batch size by half
                        assert cur_bs > 0, 'Invalid batch size.'
                        get_logger(brief=True).info(f'Adjust the batch size to {cur_bs}.')
                    else:
                        get_logger(brief=True).error(f'Can\'t afford image No.{sid} even with batch_size=1, skip.')
                        preds.append(None)  # placeholder for the failed image
                        sid += 1
                    last_fail_id = sid
                    continue
                # Save the results.
                preds.extend([{
                        'pred_classes' : pred['instances'].pred_classes.cpu(),
                        'scores'       : pred['instances'].scores.cpu(),
                        'pred_boxes'   : pred['instances'].pred_boxes.tensor.cpu(),
                    } for pred in preds_round])

                prog_bar.update(eid - sid)
                sid = eid
                # # Adjust the batch size.
                # if last_fail_id < sid - cur_bs * 2:
                #     cur_bs = min(cur_bs * 2, self.batch_size)  # gradually recover the batch size
            prog_bar.close()

            return preds, downsample_ratios
