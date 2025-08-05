import torch
import numpy as np
from typing import Dict, List


def disassemble_dict(d, keep_dim=False):
    '''
    Unpack a dictionary into a list of dictionaries. The values should be in the same length.
    If not keep dim: {k: [...] * N} -> [{k: [...]}] * N.
    If keep dim: {k: [...] * N} -> [{k: [[...]]}] * N.
    '''
    Ls = [len(v) for v in d.values()]
    assert len(set(Ls)) == 1, 'The lengths of the values should be the same!'

    N = Ls[0]
    if keep_dim:
        return [{k: v[[i]] for k, v in d.items()} for i in range(N)]
    else:
        return [{k: v[i] for k, v in d.items()} for i in range(N)]


def assemble_dict(d, expand_dim=False, keys=None):
    '''
    Pack a list of dictionaries into one dictionary.
    If expand dim, perform stack, else, perform concat.
    '''
    keys = list(d[0].keys()) if keys is None else keys
    if isinstance(d[0][keys[0]], np.ndarray):
        if expand_dim:
            return {k: np.stack([v[k] for v in d], axis=0) for k in keys}
        else:
            return {k: np.concatenate([v[k] for v in d], axis=0) for k in keys}
    elif isinstance(d[0][keys[0]], torch.Tensor):
        if expand_dim:
            return {k: torch.stack([v[k] for v in d], dim=0) for k in keys}
        else:
            return {k: torch.cat([v[k] for v in d], dim=0) for k in keys}


def filter_dict(d:Dict, keys:List, full:bool=False, strict:bool=False):
    '''
    Use path-like syntax to filter the embedded dictionary.
    The `'*'` string is regarded as a wildcard, and will return the matched keys.
    For control flags:
    - If `full`, return the full path, otherwise, only return the matched values.
    - If `strict`, raise error if the key is not found, otherwise, simply ignore.

    Eg.
    - `x = {'fruit': {'yellow': 'banana', 'red': 'apple'}, 'recycle': {'yellow': 'trash', 'blue': 'recyclable'}}`
        - `filter_dict(x, ['*', 'yellow'])` -> `{'fruit': 'banana', 'recycle': 'trash'}`
        - `filter_dict(x, ['*', 'yellow'], full=True)` -> `{'fruit': {'yellow': 'banana'}, 'recycle': {'yellow': 'trash'}}`
        - `filter_dict(x, ['*', 'blue'])` -> `{'recycle': 'recyclable'}`
        - `filter_dict(x, ['*', 'blue'], strict=True)` -> `KeyError: 'blue'`
    '''

    ret = {}
    if keys:
        cur_key, rest_keys = keys[0], keys[1:]
        if cur_key == '*':
            for match in d.keys():
                try:
                    res = filter_dict(d[match], rest_keys, full=full, strict=strict)
                    if res:
                        ret[match] = res
                except Exception as e:
                    if strict:
                        raise e
        else:
            try:
                res = filter_dict(d[cur_key], rest_keys, full=full, strict=strict)
                if res:
                    ret = { cur_key : res } if full else res
            except Exception as e:
                if strict:
                    raise e
    else:
        ret = d

    return ret