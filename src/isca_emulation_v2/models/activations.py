import torch
from torch import nn

def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(0.1)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'silu':
        return nn.SiLU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'elu':
        return nn.ELU()
    else:
        return nn.ReLU()