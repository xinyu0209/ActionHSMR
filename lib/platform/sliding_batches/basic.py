from tqdm import tqdm

class BasicBatchWindow():
    def __init__(self, sid, eid):
        self.start_id = sid
        self.end_id = eid

    @property
    def sid(self):
        return self.start_id

    @property
    def eid(self):
        return self.end_id

    @property
    def size(self):
        return self.eid - self.sid


class bsb():
    def __init__(
        self,
        total       : int,
        batch_size  : int,
        enable_tqdm : bool = False,
    ):
        # Static hyperparameters.
        self.total = int(total)
        self.B = batch_size
        # Dynamic state.
        self.tqdm = tqdm(total=self.total) if enable_tqdm else None
        self.cur_window = BasicBatchWindow(-1, 0)  # starting window

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_window.eid >= self.total:
            if self.tqdm: self.tqdm.close()
            raise StopIteration
        if self.tqdm: self.tqdm.update(self.cur_window.eid - self.cur_window.sid)

        sid = self.cur_window.eid
        eid = min(sid + self.B, self.total)
        self.cur_window = BasicBatchWindow(sid, eid)
        return self.cur_window