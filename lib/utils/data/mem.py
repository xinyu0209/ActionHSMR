import torch
from typing import List, Dict, Tuple


def recursive_detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    elif isinstance(x, Dict):
        return {k: recursive_detach(v) for k, v in x.items()}
    elif isinstance(x, (List, Tuple)):
        return [recursive_detach(v) for v in x]
    else:
        return x

