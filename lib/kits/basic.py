# Misc.
import os
import sys
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple, Any


# Machine Learning Related
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

# Framework Supports
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Framework Supports - Customized Part
from lib.utils.data import *
from lib.platform import PM
from lib.info.log import get_logger

try:
    import oven
except ImportError:
    get_logger(brief=True).warning('ExpOven is not installed. Will not be able to use the oven related functions. Check https://github.com/IsshikiHugh/ExpOven for more information.')