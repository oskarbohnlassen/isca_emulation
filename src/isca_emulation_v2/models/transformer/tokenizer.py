from __future__ import annotations

import math

import torch
from torch import nn


class CNN2DTokenizer(nn.Module):
    """Patchify [B, C, H, W] inputs into a structured [B, H_p, W_p, D] token grid."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        grid_height: int,
        grid_width: int,
        patch_size: tuple[int, int] | int,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        if isinstance(patch_size, int):
            self.patch_size = (int(patch_size), int(patch_size))
        else:
            self.patch_size = tuple(int(v) for v in patch_size)
        self.patch_height, self.patch_width = self.patch_size

        if self.grid_height % self.patch_height != 0 or self.grid_width % self.patch_width != 0:
            raise ValueError(
                "CNN2DTokenizer requires grid_height and grid_width to be divisible by patch_size. "
                f"Received grid=({self.grid_height}, {self.grid_width}) and patch_size={self.patch_size}."
            )

        self.num_patch_rows = self.grid_height // self.patch_height
        self.num_patch_cols = self.grid_width // self.patch_width
        self.num_patches = self.num_patch_rows * self.num_patch_cols
        self.patch_area = self.patch_height * self.patch_width

        self.patch_embed = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.patch_embed.weight, a=math.sqrt(5))
        if self.patch_embed.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.patch_embed.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.patch_embed.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(x)
        return patches.permute(0, 2, 3, 1).contiguous()


class CNN3DTokenizer(nn.Module):
    """Patchify [B, C, L, H, W] inputs into a structured [B, L_p, H_p, W_p, D] token grid."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        grid_depth: int,
        grid_height: int,
        grid_width: int,
        patch_size: tuple[int, int, int] | int,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.grid_depth = int(grid_depth)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)

        if isinstance(patch_size, int):
            self.patch_size = (int(patch_size), int(patch_size), int(patch_size))
        else:
            self.patch_size = tuple(int(v) for v in patch_size)
        self.patch_depth, self.patch_height, self.patch_width = self.patch_size

        if (
            self.grid_depth % self.patch_depth != 0
            or self.grid_height % self.patch_height != 0
            or self.grid_width % self.patch_width != 0
        ):
            raise ValueError(
                "CNN3DTokenizer requires grid_depth, grid_height, and grid_width to be divisible by patch_size. "
                f"Received grid=({self.grid_depth}, {self.grid_height}, {self.grid_width}) "
                f"and patch_size={self.patch_size}."
            )

        self.num_patch_depths = self.grid_depth // self.patch_depth
        self.num_patch_rows = self.grid_height // self.patch_height
        self.num_patch_cols = self.grid_width // self.patch_width
        self.num_patches = self.num_patch_depths * self.num_patch_rows * self.num_patch_cols
        self.patch_volume = self.patch_depth * self.patch_height * self.patch_width

        self.patch_embed = nn.Conv3d(
            in_channels=self.channels,
            out_channels=self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.patch_embed.weight, a=math.sqrt(5))
        if self.patch_embed.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.patch_embed.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.patch_embed.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(x)
        return patches.permute(0, 2, 3, 4, 1).contiguous()
