from ipdb import set_trace
import os
import inspect as frame_inspect
from tqdm import tqdm
from rich import inspect, pretty, print

from lib.platform import PM
from lib.platform.monitor import GPUMonitor
from lib.info.log import get_logger

def who_imported_me():
    # Get the current stack frames.
    stack = frame_inspect.stack()

    # Traverse the stack to find the first external caller.
    for frame_info in stack:
        # Filter out the internal importlib calls and the current file.
        if 'importlib' not in frame_info.filename and frame_info.filename != __file__:
            return os.path.abspath(frame_info.filename)

    # If no external file is found, it might be running as the main script.
    return None

get_logger(brief=True).warning(f'DEBUG kits are imported at {who_imported_me()}, remember to remove them.')

from lib.info.look import *
from lib.info.show import *
