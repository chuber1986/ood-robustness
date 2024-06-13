"""Collection of energy functions."""

import torch
from torch import nn


class ClassifierEnergy(nn.Module):

    def __init__(self, *_) -> None:
        super().__init__()

    def forward(self, logits):
        return -torch.logsumexp(logits, dim=-1)
