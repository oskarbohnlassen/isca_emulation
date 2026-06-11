import torch
from torch import nn
from isca_emulation_v2.models.cnn.cnn_encoder import CNN2DEncoder, CNN3DEncoder
from isca_emulation_v2.models.cnn.cnn_padding import get_padding, get_padding_3d
from isca_emulation_v2.models.activations import get_activation


class Downsample2D(nn.Module):
    def __init__(
        self,
        method: str,
        in_ch: int,
        out_ch: int,
        padding_type: str,
        latlon_padding: int | None = None,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ):
        super().__init__()
        method = str(method).lower()

        if method == "pool_max":
            self.main = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                CNN2DEncoder(
                    in_channels=in_ch,
                    hidden_dim=out_ch,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                ),
            )
        elif method == "pool_mean":
            self.main = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                CNN2DEncoder(
                    in_channels=in_ch,
                    hidden_dim=out_ch,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                ),
            )
        elif method == "strided_conv":
            pre_pad, conv_padding = get_padding(
                kernel_size=3,
                padding_mode=padding_type,
                pad=latlon_padding,
            )
            self.main = nn.Sequential(
                pre_pad,
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=2,
                    padding=conv_padding,
                ),
                nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity(),
                get_activation(activation),
            )
        else:
            raise ValueError("Unknown downsample method. Choose from ['pool_max', 'pool_mean', 'strided_conv'].")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class Downsample3D(nn.Module):
    def __init__(
        self,
        method: str,
        in_ch: int,
        out_ch: int,
        padding_type: str,
        latlon_padding: int | None = None,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ):
        super().__init__()
        method = str(method).lower()

        if method == "pool_max":
            self.main = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                CNN3DEncoder(
                    in_channels=in_ch,
                    hidden_dim=out_ch,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                ),
            )
        elif method == "pool_mean":
            self.main = nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                CNN3DEncoder(
                    in_channels=in_ch,
                    hidden_dim=out_ch,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                ),
            )
        elif method == "strided_conv":
            pre_pad, conv_padding = get_padding_3d(
                kernel_size=3,
                padding_mode=padding_type,
                pad=latlon_padding,
            )
            self.main = nn.Sequential(
                pre_pad,
                nn.Conv3d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=2,
                    padding=conv_padding,
                ),
                nn.BatchNorm3d(out_ch) if use_batch_norm else nn.Identity(),
                get_activation(activation),
            )
        else:
            raise ValueError("Unknown downsample method. Choose from ['pool_max', 'pool_mean', 'strided_conv'].")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
