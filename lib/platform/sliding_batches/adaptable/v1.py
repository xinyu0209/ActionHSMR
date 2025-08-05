import math
from tqdm import tqdm
from contextlib import contextmanager
from typing import Tuple, Union
from ..basic import BasicBatchWindow, bsb

class AdaptableBatchWindow(BasicBatchWindow):
    def __init__(self, sid, eid, min_B):
        self.start_id = sid
        self.end_id = eid
        self.min_B = min_B
        self.shrinking = False

    def shrink(self):
        if self.size <= self.min_B:
            return False
        else:
            self.shrinking = True
            return True


class asb(bsb):
    def __init__(
        self,
        total       : int,
        bs_scope    : Union[Tuple[int, int], int],
        enable_tqdm : bool = False,
    ):
        ''' Simple binary strategy. '''
        # Static hyperparameters.
        self.total = int(total)
        if isinstance(bs_scope, int):
            self.min_B = 1
            self.max_B = bs_scope
        else:
            self.min_B, self.max_B = bs_scope  # lower & upper bound of batch size
        # Dynamic state.
        self.B = self.max_B  # current batch size
        self.tqdm = tqdm(total=self.total) if enable_tqdm else None
        self.cur_window = AdaptableBatchWindow(sid=-1, eid=0, min_B=self.min_B)  # starting window
        self.last_shrink_id = None

    def __next__(self):
        if self.cur_window.shrinking:
            sid = self.cur_window.sid
            self.shrink_B(sid)
        else:
            sid = self.cur_window.eid
            self.recover_B(sid)

        if sid >= self.total:
            if self.tqdm: self.tqdm.close()
            raise StopIteration

        eid = min(sid + self.B, self.total)
        self.cur_window = AdaptableBatchWindow(sid, eid, min_B=self.min_B)
        if self.tqdm: self.tqdm.update(eid - sid)
        return self.cur_window

    def shrink_B(self, cur_id:int):
        self.last_shrink_id = cur_id
        self.cur_window.shrinking = False
        self.B = max(math.ceil(self.B/2), self.min_B)

    def recover_B(self, cur_id:int):
        if self.last_shrink_id and self.B < self.max_B:
            newer_B = min(self.B * 2, self.max_B)
            if self.last_shrink_id < cur_id - newer_B:
                self.B = newer_B