from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from isca_emulation_v2.models.cnn.cnn_decoder import CNN2DDecoder
from isca_emulation_v2.models.cnn.cnn_down_samplers import Downsample2D
from isca_emulation_v2.models.cnn.cnn_encoder import CNN2DEncoder
from isca_emulation_v2.models.cnn.cnn_layers import CNN2DLayer, CNNResBlock2D


class SimpleCNN2D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        hidden: int = 64,
        kernel_size: int = 3,
        padding_type: str = "lonlat",
        latlon_padding: int | None = None,
        activation: str = "relu",
        use_batch_norm: bool = False,
        num_layers: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.padding_type = padding_type
        self.latlon_padding = latlon_padding
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers

        self.encoder = CNN2DEncoder(
            in_channels=self.channels,
            hidden_dim=self.hidden,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
        )

        self.hidden_layers = nn.Sequential(
            *[
                CNN2DLayer(
                    hidden_dim=self.hidden,
                    kernel_size=self.kernel_size,
                    padding_type=self.padding_type,
                    latlon_padding=self.latlon_padding,
                    activation=self.activation,
                    use_batch_norm=self.use_batch_norm,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.decoder = CNN2DDecoder(hidden_dim=self.hidden, out_channels=self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.hidden_layers(h)
        out = self.decoder(h)
        x_state = x[:, : self.out_channels]
        return x_state + out


class UnetCNN2D_Fixed(nn.Module):
    """
    Minimal fixed U-Net-style CNN:
    - input proj -> 5x5 conv -> pool
    - 3x3 conv -> pool
    - 3x3 bottleneck
    - upsample -> 3x3 conv
    - upsample -> 5x5 conv -> output proj
    - channels: base -> 2*base -> 4*base -> 2*base -> base
    """

    def __init__(
        self,
        channels: int,
        hidden_dim_base: int,
        down_sample_method: str,
        activation: str,
        padding_type: str,
        latlon_padding: int | None = None,
        use_batch_norm: bool = False,
        bottleneck_dilation: int = 1,
        out_channels: int | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.out_channels = int(out_channels)
        self.hidden_dim_base = int(hidden_dim_base)
        self.down_sample_method = str(down_sample_method).lower()
        self.bottleneck_dilation = int(bottleneck_dilation)
        self.activation = activation
        self.padding_type = padding_type
        self.latlon_padding = latlon_padding
        self.use_batch_norm = bool(use_batch_norm)

        if self.hidden_dim_base < 1:
            raise ValueError("hidden_dim_base must be >= 1.")
        if self.down_sample_method not in {"pool_max", "pool_mean", "strided_conv"}:
            raise ValueError("down_sample_method must be one of ['pool_max', 'pool_mean', 'strided_conv'].")
        if self.bottleneck_dilation < 1:
            raise ValueError("bottleneck_dilation must be an integer >= 1.")

        c1 = self.hidden_dim_base
        c2 = 2 * self.hidden_dim_base
        c3 = 4 * self.hidden_dim_base

        self.input_proj = CNN2DEncoder(
            in_channels=self.channels,
            hidden_dim=c1,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
        )

        self.down_1 = Downsample2D(
            method=self.down_sample_method,
            in_ch=c1,
            out_ch=c2,
            padding_type=self.padding_type,
            latlon_padding=self.latlon_padding,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
        )
        self.down_2 = Downsample2D(
            method=self.down_sample_method,
            in_ch=c2,
            out_ch=c3,
            padding_type=self.padding_type,
            latlon_padding=self.latlon_padding,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
        )

        self.enc_5x5 = self._make_layer(hidden_dim=c1, kernel_size=5)
        self.enc_3x3 = self._make_layer(hidden_dim=c2, kernel_size=3)
        self.bottleneck = self._make_layer(
            hidden_dim=c3,
            kernel_size=3,
            dilation=self.bottleneck_dilation,
        )

        self.from_c3_to_c2 = CNN2DEncoder(
            in_channels=c3,
            hidden_dim=c2,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
        )
        self.from_c2_to_c1 = CNN2DEncoder(
            in_channels=c2,
            hidden_dim=c1,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
        )
        self.fuse_c2 = CNN2DEncoder(
            in_channels=2 * c2,
            hidden_dim=c2,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
        )
        self.fuse_c1 = CNN2DEncoder(
            in_channels=2 * c1,
            hidden_dim=c1,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
        )
        self.dec_3x3 = self._make_layer(hidden_dim=c2, kernel_size=3)
        self.dec_5x5 = self._make_layer(hidden_dim=c1, kernel_size=5)

        self.output_proj = CNN2DDecoder(hidden_dim=c1, out_channels=self.out_channels)

    def _make_layer(self, hidden_dim: int, kernel_size: int, dilation: int = 1) -> nn.Sequential:
        return nn.Sequential(
            CNNResBlock2D(
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                padding_type=self.padding_type,
                latlon_padding=self.latlon_padding,
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)

        h = self.enc_5x5(h)
        skip1 = h
        size_after_enc1 = h.shape[-2:]
        h = self.down_1(h)
        h = self.enc_3x3(h)
        skip2 = h
        size_after_enc2 = h.shape[-2:]
        h = self.down_2(h)
        h = self.bottleneck(h)

        h = F.interpolate(h, size=size_after_enc2, mode="bilinear", align_corners=False)
        h = self.from_c3_to_c2(h)
        h = torch.cat([h, skip2], dim=1)
        h = self.fuse_c2(h)
        h = self.dec_3x3(h)

        h = F.interpolate(h, size=size_after_enc1, mode="bilinear", align_corners=False)
        h = self.from_c2_to_c1(h)
        h = torch.cat([h, skip1], dim=1)
        h = self.fuse_c1(h)
        h = self.dec_5x5(h)

        delta = self.output_proj(h)
        x_state = x[:, : self.out_channels]
        return x_state + delta
