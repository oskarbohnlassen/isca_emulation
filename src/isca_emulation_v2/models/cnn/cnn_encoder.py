import torch
import torch.nn as nn

from isca_emulation_v2.models.activations import get_activation


class CNN2DEncoder(nn.Module):
    """Project input channels to a shared hidden feature space."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(hidden_dim) if use_batch_norm else nn.Identity(),
            get_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class CNN3DEncoder(nn.Module):
    """Project input channels to a shared hidden feature space."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=1, padding=0),
            nn.BatchNorm3d(hidden_dim) if use_batch_norm else nn.Identity(),
            get_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
