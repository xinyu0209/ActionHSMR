import time
import atexit
import inspect
import torch

from typing import Optional, Union, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def fold_path(fn:str):
    ''' Fold a path like `from/to/file.py` to relative `f/t/file.py`. '''
    return '/'.join([p[:1] for p in fn.split('/')[:-1]]) + '/' + fn.split('/')[-1]


def summary_frame_info(frame:inspect.FrameInfo):
    ''' Convert a FrameInfo object to a summary string. '''
    return f'{frame.function} @ {fold_path(frame.filename)}:{frame.lineno}'


class TimeMonitorDisabled:
    def foo(self, *args, **kwargs):
        return

    def __init__(self, log_folder:Optional[Union[str, Path]]=None, record_birth_block:bool=False):
        self.tick = self.foo
        self.report = self.foo
        self.clear = self.foo
        self.dump_statistics = self.foo

    def __call__(self, *args, **kwargs):
        return self
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return


class TimeMonitor:
    '''
    It is supposed to be used like this: 

    time_monitor = TimeMonitor() 

    with time_monitor('test_block', 'Block that does something.') as tm: 
        do_something()

    time_monitor.report()
    '''

    def __init__(self, log_folder:Optional[Union[str, Path]]=None, record_birth_block:bool=False):
        if log_folder is not None:
            self.log_folder = Path(log_folder) if isinstance(log_folder, str) else log_folder
            self.log_folder.mkdir(parents=True, exist_ok=True)
            log_fn = self.log_folder / 'readable.log'
            self.log_fh = open(log_fn, 'w')  # Log file handler.
            self.log_fh.write('=== New Exp ===\n')
        else:
            self.log_folder = None
        self.clear()
        self.current_block_uid_stack : List = []  # Unique block id stack for recording.
        self.current_block_aid_stack : List = []  # Block id stack for accumulated cost analysis.

        # Specially add a global start and end block.
        self.record_birth_block = record_birth_block and log_folder is not None
        if self.record_birth_block:
            self.__call__('monitor_birth', 'Since the monitor is constructed.')
            self.__enter__()

        # Register the exit hook to dump the data safely.
        atexit.register(self._die_hook)


    def __call__(self, block_name:str, block_desc:Optional[str]=None):
        ''' Set up the name of the context for a block. '''
        # 1. Format the block name.
        block_name = block_name.replace('/', '-').replace(' ', '-')
        block_name_recursive = '/'.join([s.split('/')[-1] for s in self.current_block_aid_stack] + [block_name])  # Tree structure block name.
        # 2. Get a unique name for the block record.
        block_postfixed = 0
        while f'{block_name_recursive}_{block_postfixed}' in self.block_info:
            block_postfixed += 1
        # 3. Get the caller frame information.
        caller_frame = inspect.stack()[1]
        block_position = summary_frame_info(caller_frame)
        # 4. Initialize the block information.
        self.current_block_uid_stack.append(f'{block_name_recursive}_{block_postfixed}')
        self.current_block_aid_stack.append(block_name)
        self.block_info[self.current_block_uid_stack[-1]] = {
                'records'  : [],
                'position' : block_position,
                'desc'     : block_desc,
            }

        return self


    def __enter__(self):
        caller_frame = inspect.stack()[1]
        record = self._tick_record(caller_frame, 'Start of the block.')
        self.block_info[self.current_block_uid_stack[-1]]['records'].append(record)
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        caller_frame = inspect.stack()[1]
        record = self._tick_record(caller_frame, 'End of the block.')
        self.block_info[self.current_block_uid_stack[-1]]['records'].append(record)

        # Finish one block.
        curr_block_uid = self.current_block_uid_stack.pop()
        curr_block_aid = '/'.join(self.current_block_aid_stack)
        self.current_block_aid_stack.pop()

        self.finished_blocks.append(curr_block_uid)
        elapsed = self.block_info[curr_block_uid]['records'][-1]['timestamp'] \
                - self.block_info[curr_block_uid]['records'][0]['timestamp']
        self.block_cost[curr_block_aid] = self.block_cost.get(curr_block_aid, 0) + elapsed

        if hasattr(self, 'dump_thread'):
            self.dump_thread.result()
        with ThreadPoolExecutor() as executor:
            self.dump_thread = executor.submit(self.dump_statistics)



    def tick(self, desc:str=''):
        ''' 
        Record a intermediate timestamp. These records are only for in-block analysis, 
        and will be ignored when analyzing in global view. 
        '''
        caller_frame = inspect.stack()[1]
        record = self._tick_record(caller_frame, desc)
        self.block_info[self.current_block_uid_stack[-1]]['records'].append(record)
        return


    def report(self, level:Union[str, List[str]]='global'):
        import rich

        caller_frame = inspect.stack()[1]
        caller_info = summary_frame_info(caller_frame)

        if isinstance(level, str):
            level = [level]

        for lv in level:  # To make sure we can output in order.
            if lv == 'block':
                rich.print(f'[bold underline][EA-B][/bold underline] {caller_info} -> blocks level records:')
                for block_name in self.finished_blocks:
                    msg = '\t' + self._generate_block_msg(block_name).replace('\n\t', '\n\t\t')
                    rich.print(msg)
            elif lv == 'global':
                rich.print(f'[bold underline][EA-G][/bold underline] {caller_info} -> global efficiency analysis:')
                for block_name, cost in self.block_cost.items():
                    rich.print(f'\t{block_name}: {cost:.2f} sec')


    def clear(self):
        self.finished_blocks = []
        self.block_info = {}
        self.block_cost = {}


    def dump_statistics(self):
        ''' Dump the logging raw data for post analysis. '''
        if self.log_folder is None:
            return

        dump_fn = self.log_folder / 'statistics.pkl'
        with open(dump_fn, 'wb') as f:
            import pickle
            pickle.dump({
                'finished_blocks' : self.finished_blocks,
                'block_info'      : self.block_info,
                'block_cost'      : self.block_cost,
                'curr_aid_stack'  : self.current_block_aid_stack,  # nonempty when when errors happen inside a block
            }, f)


    # TODO: Draw a graph to visualize the time consumption.


    def _tick_record(self, caller_frame, desc:Optional[str]=''):
        # 1. Generate the record.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timestamp = time.time()
        readable_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        position = summary_frame_info(caller_frame)
        record = {
            'time'      : readable_time,
            'timestamp' : timestamp,
            'position'  : position,
            'desc'      : desc,
        }

        # 2. Log the record.
        if self.log_folder is not None:
            block_uid = self.current_block_uid_stack[-1]
            log_msg = f'[{readable_time}] 🗂️ {block_uid} 📌 {desc} 🌐 {position}'
            self.log_fh.write(log_msg + '\n')

        return record


    def _generate_block_msg(self, block_name):
        block_info = self.block_info[block_name]
        block_position = block_info['position']
        block_desc = block_info['desc']
        records = block_info['records']
        msg = f'🗂️ {block_name} 📌 {block_desc} 🌐 {block_position}'
        for rid, record in enumerate(records):
            readable_time = record['time']
            tick_desc = record['desc']
            tick_position = record['position']
            if rid > 0:
                prev_record = records[rid-1]
                tick_elapsed = record['timestamp'] - prev_record['timestamp']
                tick_elapsed = f'{tick_elapsed:.2f} s'
            else:
                tick_elapsed = 'N/A'
            msg += f'\n\t[{readable_time}] ⏳ {tick_elapsed} 📌 {tick_desc} 🌐 {tick_position}'
        return msg


    def _die_hook(self):
        if self.record_birth_block:
            self.__exit__(None, None, None)

        self.dump_statistics()

        if self.log_folder is not None:
            self.log_fh.close()