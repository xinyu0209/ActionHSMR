import imageio
import numpy as np

from tqdm import tqdm
from typing import Union, List
from pathlib import Path
from glob import glob

from .edit import flex_resize_img, flex_resize_video


def load_img_meta(
    img_path : Union[str, Path],
):
    ''' Read the image meta from the given path without opening image. '''
    assert Path(img_path).exists(), f'Image not found: {img_path}'
    H, W = imageio.v3.improps(img_path).shape[:2]
    meta = {'w': W, 'h': H}
    return meta


def load_img(
    img_path : Union[str, Path],
    mode     : str = 'RGB',
):
    ''' Read the image from the given path. '''
    assert Path(img_path).exists(), f'Image not found: {img_path}'

    img = imageio.v3.imread(img_path, plugin='pillow', mode=mode)

    meta = {
        'w': img.shape[1],
        'h': img.shape[0],
    }
    return img, meta


def save_img(
    img          : np.ndarray,
    output_path  : Union[str, Path],
    resize_ratio : Union[float, None] = None,
    **kwargs,
):
    ''' Save the image. '''
    assert img.ndim == 3, f'Invalid image shape: {img.shape}'

    if resize_ratio is not None:
        img = flex_resize_img(img, ratio=resize_ratio)

    imageio.v3.imwrite(output_path, img, **kwargs)


def load_video(
    video_path : Union[str, Path],
):
    ''' Read the video from the given path. '''
    if isinstance(video_path, str):
        video_path = Path(video_path)

    assert video_path.exists(), f'Video not found: {video_path}'

    if video_path.is_dir():
        print(f'Found {video_path} is a directory. It will be regarded as a image folder.')
        imgs_path = sorted(glob(str(video_path / '*')))
        frames = []
        for img_path in tqdm(imgs_path):
            frames.append(imageio.imread(img_path))
        fps = 30 # default fps
    else:
        print(f'Found {video_path} is a file. It will be regarded as a video file.')
        reader = imageio.get_reader(video_path, format='FFMPEG')
        frames = []
        for frame in tqdm(reader, total=reader.count_frames()):
            frames.append(frame)
        fps = reader.get_meta_data()['fps']
    frames = np.stack(frames, axis=0) # (L, H, W, 3)
    meta = {
        'fps': fps,
        'w'  : frames.shape[2],
        'h'  : frames.shape[1],
        'L'  : frames.shape[0],
    }

    return frames, meta


def save_video(
    frames       : Union[np.ndarray, List[np.ndarray]],
    output_path  : Union[str, Path],
    fps          : float = 30,
    resize_ratio : Union[float, None] = None,
    quality      : Union[int, None]   = None,
):
    ''' Save the frames as a video. '''
    if isinstance(frames, List):
        frames = np.stack(frames, axis=0)
    assert frames.ndim == 4, f'Invalid frames shape: {frames.shape}'

    if resize_ratio is not None:
        frames = flex_resize_video(frames, ratio=resize_ratio)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(output_path, fps=fps, quality=quality)
    output_seq_name = str(output_path).split('/')[-1]
    for frame in tqdm(frames, desc=f'Saving {output_seq_name}'):
        writer.append_data(frame)
    writer.close()