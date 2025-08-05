import torch
from typing import Any, Dict, List


def recursive_to(x: Any, target: torch.device):
    '''
    Recursively transfer data to the target device.
    Modified from: https://github.com/shubham-goel/4D-Humans/blob/6ec79656a23c33237c724742ca2a0ec00b398b53/hmr2/utils/__init__.py#L9-L25

    ### Args
    - x: Any
    - target: torch.device
    
    ### Returns
    - Data transferred to the target device.
    '''
    if isinstance(x, Dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(target)
    elif isinstance(x, List):
        return [recursive_to(i, target) for i in x]
    else:
        return x
