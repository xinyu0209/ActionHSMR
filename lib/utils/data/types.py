import torch
import numpy as np

from typing import Any, List
from omegaconf import ListConfig

def to_numpy(x, temporary:bool=False) -> Any:
    if isinstance(x, torch.Tensor):
        if temporary:
            recover_type_back = lambda x_: torch.from_numpy(x_).type_as(x).to(x.device)
            return x.detach().cpu().numpy(), recover_type_back
        else:
            return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if temporary:
            recover_type_back = lambda x_: x_
            return x.copy(), recover_type_back
        else:
            return x
    if isinstance(x, List):
        if temporary:
            recover_type_back = lambda x_: x_.tolist()
            return np.array(x), recover_type_back
        else:
            return np.array(x)
    raise ValueError(f"Unsupported type: {type(x)}")

def to_tensor(x, device, temporary:bool=False) -> Any:
    '''
    Simply unify the type transformation to torch.Tensor. 
    If device is None, don't change the device if device is not CPU. 
    '''
    if isinstance(x, torch.Tensor):
        device = x.device if device is None else device
        if temporary:
            recover_type_back = lambda x_: x_.to(x.device)  # recover the device
            return x.to(device), recover_type_back
        else:
            return x.to(device)

    device = 'cpu' if device is None else device
    if isinstance(x, np.ndarray):
        if temporary:
            recover_type_back = lambda x_: x_.detach().cpu().numpy()
            return torch.from_numpy(x).to(device), recover_type_back
        else:
            return torch.from_numpy(x).to(device)
    if isinstance(x, List):
        if temporary:
            recover_type_back = lambda x_: x_.tolist()
            return torch.from_numpy(np.array(x)).to(device), recover_type_back
        else:
            return torch.from_numpy(np.array(x)).to(device)
    raise ValueError(f"Unsupported type: {type(x)}")


def to_list(x, temporary:bool=False) -> Any:
    if isinstance(x, List):
        if temporary:
            recover_type_back = lambda x_: x_
            return x.copy(), recover_type_back
        else:
            return x
    if isinstance(x, torch.Tensor):
        if temporary:
            recover_type_back = lambda x_: torch.tensor(x_, device=x.device, dtype=x.dtype)
            return x.detach().cpu().numpy().tolist(), recover_type_back
        else:
            return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        if temporary:
            recover_type_back = lambda x_: np.array(x_)
            return x.tolist(), recover_type_back
        else:
            return x.tolist()
    if isinstance(x, ListConfig):
        if temporary:
            recover_type_back = lambda x_: ListConfig(x_)
            return list(x), recover_type_back
        else:
            return list(x)
    raise ValueError(f"Unsupported type: {type(x)}")