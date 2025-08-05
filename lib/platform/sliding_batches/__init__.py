from .basic import bsb
from .adaptable.v1 import asb

# The batch manager is used to quickly initialize a batched task.
# The Usage of the batch manager is as follows:


def eg_bbm():
    ''' Basic version of the batch manager. '''
    task_len = 1e6
    task_things = [i for i in range(int(task_len))]

    for bw in bsb(total=task_len, batch_size=300, enable_tqdm=True):
        sid = bw.sid
        eid = bw.eid
        round_things = task_things[sid:eid]
        # Do something with `round_things`.


def eg_asb():
    ''' Basic version of the batch manager. '''
    task_len = 1024
    task_things = [i for i in range(int(task_len))]

    lb, ub = 1, 300  # lower & upper bound of batch size
    for bw in asb(total=task_len, bs_scope=(lb, ub), enable_tqdm=True):
        sid = bw.sid
        eid = bw.eid
        round_things = task_things[sid:eid]
        # Do something with `round_things`.

        try:
            # Do something with `round_things`.
            pass
        except Exception as e:
            if not bw.shrink():
                # In this case, it means task_things[sid:sid+lb] is still too large to handle.
                # So you need to do something to handle this situation.
                pass
            continue  #! DO NOT FORGET CONTINUE

        # Do something with `round_things` if no exception is raised.
