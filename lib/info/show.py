# Provides methods to visualize the information of data, giving a brief overview in figure.

import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Union, List, Dict
from pathlib import Path

from lib.utils.data import to_numpy


def show_distribution(
    data       : Dict,
    fn         : Union[str, Path],  # File name of the saved figure.
    bins       : int  = 100,        # Number of bins in the histogram.
    annotation : bool = False,
    title      : str = 'Data Distribution',
    axis_names : List = ['Value', 'Frequency'],
    bounds     : Optional[List] = None,  # Left and right bounds of the histogram.
):
    '''
    Visualize the distribution of the data using histogram.
    The data should be a dictionary with keys as the labels and values as the data.
    '''
    labels = list(data.keys())
    data = np.stack([ to_numpy(x) for x in data.values() ], axis=0)
    assert data.ndim == 2, f"Data dimension should be 2, but got {data.ndim}."
    assert bounds is None or len(bounds) == 2, f"Bounds should be a list of length 2, but got {bounds}."
    # Preparation.
    N, K = data.shape
    data = data.transpose(1, 0)  # (K, N)
    # Plot.
    plt.hist(data, bins=bins, alpha=0.7, label=labels)
    if annotation:
        for i in range(K):
            for j in range(N):
                plt.text(data[i, j], 0, f'{data[i, j]:.2f}', va='bottom', fontsize=6)
    plt.title(title)
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.legend()
    if bounds:
        plt.xlim(bounds)
    # Save.
    plt.savefig(fn)
    plt.close()



def show_history(
    data       : Dict,
    fn         : Union[str, Path],  # file name of the saved figure
    annotation : bool = False,
    title      : str  = 'Data History',
    axis_names : List = ['Time', 'Value'],
    ex_starts  : Dict[str, int] = {},  # starting points of the history if not starting from 0
):
    '''
    Visualize the value of changing across time.
    The history should be a dictionary with keys as the metric names and values as the metric values.
    '''
    # Make sure the fn's parent exists.
    if isinstance(fn, str):
        fn = Path(fn)
    fn.parent.mkdir(parents=True, exist_ok=True)

    # Preparation.
    history_name = list(data.keys())
    history_data = [ to_numpy(x) for x in data.values() ]
    N = len(history_name)
    Ls = [len(x) for x in history_data]
    Ss = [
            ex_starts[history_name[i]]
            if (history_name[i] in ex_starts.keys()) else 0
            for i in range(N)
        ]

    # Plot.
    for i in range(N):
        plt.plot(range(Ss[i], Ss[i]+Ls[i]), history_data[i], label=history_name[i])
    if annotation:
        for i in range(N):
            for j in range(Ls[i]):
                plt.text(Ss[i]+j, history_data[i][j], f'{history_data[i][j]:.2f}', fontsize=6)

    plt.title(title)
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.legend()
    # Save.
    plt.savefig(fn)
    plt.close()