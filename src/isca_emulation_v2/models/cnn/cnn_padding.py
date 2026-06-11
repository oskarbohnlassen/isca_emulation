import torch
import torch.nn as nn
import torch.nn.functional as F


def get_padding(
    kernel_size: int,
    padding_mode: str,
    pad: int | None = None,
    dilation: int = 1,
) -> tuple[nn.Module, int]:
    """Return (pre_conv_padding_layer, conv2d_padding)."""
    base_pad = kernel_size // 2 if pad is None else int(pad)
    pad_amt = base_pad * int(dilation)

    if padding_mode == "lonlat":
        return pad_latlon(pad=pad_amt), 0
    if padding_mode in {"zeros", "same"}:
        return nn.Identity(), pad_amt

    raise ValueError(f"Unsupported padding mode: {padding_mode}")


def get_padding_3d(
    kernel_size: int,
    padding_mode: str,
    pad: int | None = None,
    dilation: int = 1,
) -> tuple[nn.Module, int]:
    """Return (pre_conv_padding_layer, conv3d_padding)."""
    base_pad = kernel_size // 2 if pad is None else int(pad)
    pad_amt = base_pad * int(dilation)

    if padding_mode == "lonlat":
        return pad_level_latlon(pad=pad_amt), 0
    if padding_mode in {"zeros", "same"}:
        return nn.Identity(), pad_amt

    raise ValueError(f"Unsupported padding mode: {padding_mode}")


class pad_latlon(nn.Module):
    def __init__(self, pad: int = 1):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.pad, self.pad), mode="replicate")
        return x


class pad_level_latlon(nn.Module):
    def __init__(self, pad: int = 1):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, C, L, H, W].
        if self.pad <= 0:
            return x

        # Circular only along lon (W) by concatenating edge slices.
        x = torch.cat([x[..., -self.pad :], x, x[..., : self.pad]], dim=-1)
        # Replicate along lat (H) and level (L) in one pass.
        x = F.pad(x, (0, 0, self.pad, self.pad, self.pad, self.pad), mode="replicate")
        return x
