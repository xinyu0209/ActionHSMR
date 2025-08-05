
import pickle
from pathlib import Path


def load_pickle(fn, mode='rb', encoding=None, pickle_encoding='ASCII'):
    if isinstance(fn, Path):
        fn = str(fn)
    with open(fn, mode=mode, encoding=encoding) as f:
        data = pickle.load(f, encoding=pickle_encoding)
    return data