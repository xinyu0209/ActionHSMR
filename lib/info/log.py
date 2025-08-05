import inspect
import logging
import torch
from time import time
from colorlog import ColoredFormatter


def fold_path(fn:str):
    ''' Fold a path like `from/to/file.py` to relative `f/t/file.py`. '''
    from lib.platform.proj_manager import ProjManager as PM
    root_abs = str(PM.root.absolute())
    if fn.startswith(root_abs):
        fn = fn[len(root_abs)+1:]

    return '/'.join([p[:1] for p in fn.split('/')[:-1]]) + '/' + fn.split('/')[-1]


def sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time()


def get_logger(brief:bool=False, show_stack:bool=False):
    # 1. Get the caller's file name and function name to identify the logging position.
    caller_frame = inspect.currentframe().f_back
    line_num = caller_frame.f_lineno
    file_name = caller_frame.f_globals["__file__"]
    file_name = fold_path(file_name)
    func_name = caller_frame.f_code.co_name
    frames_stack = inspect.stack()

    # 2. Add a trace method to the logger.
    def trace_handler(self, message, *args, **kws):
        if self.isEnabledFor(TRACE):
            self._log(TRACE, message, args, **kws)

    TRACE = 15  # DEBUG is 10 and INFO is 20
    logging.addLevelName(TRACE, 'TRACE')
    logging.Logger.trace = trace_handler

    # 3. Set up the logger.
    logger = logging.getLogger()
    logger.time = time
    logger.sync_time = sync_time

    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.StreamHandler()

    if brief:
        prefix = f'[%(cyan)s%(asctime)s%(reset)s]'
    else:
        prefix = f'[%(cyan)s%(asctime)s%(reset)s @ %(cyan)s{func_name}%(reset)s @ %(cyan)s{file_name}%(reset)s:%(cyan)s{line_num}%(reset)s]'
    if show_stack:
        suffix = '\n STACK: ' + ' @ '.join([f'{fold_path(frame.filename)}:{frame.lineno}' for frame in frames_stack[1:]])
    else:
        suffix = ''
    formatstring = f'{prefix}[%(log_color)s%(levelname)s%(reset)s] %(message)s{suffix}'
    datefmt = '%m/%d %H:%M:%S'
    ch.setFormatter(ColoredFormatter(formatstring, datefmt=datefmt))
    logger.addHandler(ch)

    # Modify the logging level here.
    logger.setLevel(TRACE)
    ch.setLevel(TRACE)

    return logger

if __name__ == '__main__':
    get_logger().trace('Test TRACE')
    get_logger().info('Test INFO')
    get_logger().warning('Test WARN')
    get_logger().error('Test ERROR')
    get_logger().fatal('Test FATAL')