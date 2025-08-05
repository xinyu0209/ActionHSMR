# Provides methods to summarize the information of data, giving a brief overview in text.

import torch
import numpy as np

from typing import Optional

from .log import get_logger


def look_tensor(
    x      : torch.Tensor,
    prompt : Optional[str] = None,
    silent : bool = False,
):
    '''
    Summarize the information of a tensor, including its shape, value range (min, max, mean, std), and dtype.
    Then return a string containing the information.

    ### Args
    - x: torch.Tensor
    - silent: bool, default `False`
        - If not silent, the function will print the message itself. The information string will always be returned.
    - prompt: Optional[str], default `None`
        - If have prompt, it will be printed at the very beginning.

    ### Returns
    - str
    '''
    info_list = [] if prompt is None else [prompt]
    # Convert to float to calculate the statistics.
    x_num = x.float()
    info_list.append(f'📐 [{x_num.min():06f} -> {x_num.max():06f}] ~ ({x_num.mean():06f}, {x_num.std():06f})')
    info_list.append(f'📦 {tuple(x.shape)}')
    info_list.append(f'🏷️ {x.dtype}')
    info_list.append(f'🖥️ {x.device}')
    # Generate the final information and print it if necessary.
    ret = '\t'.join(info_list)
    if not silent:
        get_logger().info(ret)
    return ret


def look_ndarray(
    x      : np.ndarray,
    silent : bool = False,
    prompt : Optional[str] = None,
):
    '''
    Summarize the information of a numpy array, including its shape, value range (min, max, mean, std), and dtype.
    Then return a string containing the information.

    ### Args
    - x: np.ndarray
    - silent: bool, default `False`
        - If not silent, the function will print the message itself. The information string will always be returned.
    - prompt: Optional[str], default `None`
        - If have prompt, it will be printed at the very beginning.

    ### Returns
    - str
    '''
    info_list = [] if prompt is None else [prompt]
    # Convert to float to calculate the statistics.
    x_num = x.astype(np.float32)
    info_list.append(f'📐 [ {x_num.min():06f} -> {x_num.max():06f} ] ~ ( {x_num.mean():06f}, {x_num.std():06f} )')
    info_list.append(f'📦 {tuple(x.shape)}')
    info_list.append(f'🏷️  {x.dtype}')
    # Generate the final information and print it if necessary.
    ret = '\t'.join(info_list)
    if not silent:
        get_logger().info(ret)
    return ret


def look_dict(
    d      : dict,
    silent : bool = False,
):
    '''
    Summarize the information of a dictionary, including the keys and the information of the values.
    Then return a string containing the information.

    ### Args
    - d: dict
    - silent: bool, default `False`
        - If not silent, the function will print the message itself. The information string will always be returned.

    ### Returns
    - str
    '''
    info_list = ['{']

    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            info_list.append(f'{k} : tensor: {look_tensor(v, silent=True)}')
        elif isinstance(v, np.ndarray):
            info_list.append(f'{k} : ndarray: {look_ndarray(v, silent=True)}')
        elif isinstance(v, str):
            info_list.append(f'{k} : {v[:32]}')
        else:
            info_list.append(f'{k} : {type(v)}')

    info_list.append('}')
    ret = '\n'.join(info_list)
    if not silent:
        get_logger().info(ret)
    return ret