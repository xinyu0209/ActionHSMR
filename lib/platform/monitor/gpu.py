import time
import torch
import inspect


def fold_path(fn:str):
    ''' Fold a path like `from/to/file.py` to relative `f/t/file.py`. '''
    return '/'.join([p[:1] for p in fn.split('/')[:-1]]) + '/' + fn.split('/')[-1]


def summary_frame_info(frame:inspect.FrameInfo):
    ''' Convert a FrameInfo object to a summary string. '''
    return f'{frame.function} @ {fold_path(frame.filename)}:{frame.lineno}'


class GPUMonitor():
    '''
    This monitor is designed for GPU memory analysis. It records the peak memory usage in a period of time.
    A snapshot will record the peak memory usage until the snapshot is taken. (After init / reset / previous snapshot.)
    '''

    def __init__(self):
        self.reset()
        self.clear()
        self.log_fn = 'gpu_monitor.log'


    def snapshot(self, desc:str='snapshot'):
        timestamp = time.time()
        caller_frame = inspect.stack()[1]
        peak_MB = torch.cuda.max_memory_allocated() / 1024 / 1024
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        free_mem_MB, total_mem_MB = free_mem / 1024 / 1024, total_mem / 1024 / 1024

        record = {
                'until'     : time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)),
                'until_raw' : timestamp,
                'position'  : summary_frame_info(caller_frame),
                'peak'      : peak_MB,
                'peak_msg'  : f'{peak_MB:.2f} MB',
                'free'      : free_mem_MB,
                'total'     : total_mem_MB,
                'free_msg'  : f'{free_mem_MB:.2f} MB',
                'total_msg' : f'{total_mem_MB:.2f} MB',
                'desc'      : desc,
            }

        self.max_peak = max(self.max_peak_MB, peak_MB)

        self.records.append(record)
        self._update_log(record)

        self.reset()
        return record


    def report_latest(self, k:int=1):
        import rich
        caller_frame = inspect.stack()[1]
        caller_info = summary_frame_info(caller_frame)
        rich.print(f'{caller_info} -> latest {k} records:')
        for rid, record in enumerate(self.records[-k:]):
            msg = self._generate_log_msg(record)
            rich.print(msg)


    def report_all(self):
        self.report_latest(len(self.records))


    def reset(self):
        torch.cuda.reset_peak_memory_stats()
        return


    def clear(self):
        self.records = []
        self.max_peak_MB = 0

    def _generate_log_msg(self, record):
        time = record['until']
        peak = record['peak']
        desc = record['desc']
        position = record['position']
        msg = f'[{time}] ⛰️ {peak:>8.2f} MB 📌 {desc} 🌐 {position}'
        return msg


    def _update_log(self, record):
        msg = self._generate_log_msg(record)
        with open(self.log_fn, 'a') as f:
            f.write(msg + '\n')
