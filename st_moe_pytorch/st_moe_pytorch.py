import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce, repeat

# helper functoins

def exists(val):
    return val is not None

# main class

class MoE(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return x
