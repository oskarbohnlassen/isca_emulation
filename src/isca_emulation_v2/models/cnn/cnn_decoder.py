import torch
import torch.nn as nn


class CNN2DDecoder(nn.Module):
    """Project hidden feature maps back to output channels."""

    def __init__(self, hidden_dim: int, out_channels: int):
        super().__init__()
        self.main = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class CNN3DDecoder(nn.Module):
    """Project hidden feature maps back to output channels."""

    def __init__(self, hidden_dim: int, out_channels: int):
        super().__init__()
        self.main = nn.Conv3d(hidden_dim, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
