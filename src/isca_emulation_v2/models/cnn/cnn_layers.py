import torch
import torch.nn as nn

from isca_emulation_v2.models.cnn.cnn_padding import get_padding, get_padding_3d
from isca_emulation_v2.models.activations import get_activation

class CNN2DLayer(nn.Module):
    """One hidden-space CNN layer: pad -> conv -> norm -> activation."""

    def __init__(
        self,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        dilation: int = 1,
        padding_type: str = "lonlat",
        latlon_padding: int | None = None,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ):
        super().__init__()
        pre_pad, conv_padding = get_padding(
            kernel_size=kernel_size,
            padding_mode=padding_type,
            pad=latlon_padding,
            dilation=dilation,
        )

        self.main = nn.Sequential(
            pre_pad,
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
            ),
            nn.BatchNorm2d(hidden_dim) if use_batch_norm else nn.Identity(),
            get_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class CNNResBlock2D(nn.Module):
    """Residual block in hidden space: (CNN2DLayer -> CNN2DLayer) + skip."""
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        dilation: int = 1,
        padding_type: str = "lonlat",
        latlon_padding: int | None = None,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.f = nn.Sequential(
            CNN2DLayer(
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                padding_type=padding_type,
                latlon_padding=latlon_padding,
                activation=activation,
                use_batch_norm=use_batch_norm,
            ),
            CNN2DLayer(
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                padding_type=padding_type,
                latlon_padding=latlon_padding,
                activation=activation,
                use_batch_norm=use_batch_norm,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.f(x)


class CNN3DLayer(nn.Module):
    """One hidden-space CNN layer: pad -> conv -> norm -> activation."""

    def __init__(
        self,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        dilation: int = 1,
        padding_type: str = "lonlat",
        latlon_padding: int | None = None,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ):
        super().__init__()
        pre_pad, conv_padding = get_padding_3d(
            kernel_size=kernel_size,
            padding_mode=padding_type,
            pad=latlon_padding,
            dilation=dilation,
        )

        self.main = nn.Sequential(
            pre_pad,
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
            ),
            nn.BatchNorm3d(hidden_dim) if use_batch_norm else nn.Identity(),
            get_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class CNNResBlock3D(nn.Module):
    """Residual block in hidden space: (CNN3DLayer -> CNN3DLayer) + skip."""

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        dilation: int = 1,
        padding_type: str = "lonlat",
        latlon_padding: int | None = None,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.f = nn.Sequential(
            CNN3DLayer(
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                padding_type=padding_type,
                latlon_padding=latlon_padding,
                activation=activation,
                use_batch_norm=use_batch_norm,
            ),
            CNN3DLayer(
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                padding_type=padding_type,
                latlon_padding=latlon_padding,
                activation=activation,
                use_batch_norm=use_batch_norm,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.f(x)
