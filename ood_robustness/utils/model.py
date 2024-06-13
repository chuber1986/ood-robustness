from contextlib import contextmanager
from itertools import count

import torch
from torch import nn
import numpy as np

def evaluate(loader: torch.utils.data.DataLoader, model, device='cpu', max_samples=None):
    xs = []

    if max_samples:
        r = range(int(np.ceil(max_samples / loader.batch_size)))
    else:
        r = count(0)

    with torch.no_grad():
        for _, (x, y) in zip(r, loader):
            x = x.to(device)
            x = model(x)
            xs.append(x)

        x = torch.concat(xs, dim=0)

    return x
