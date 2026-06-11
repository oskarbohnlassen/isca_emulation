from __future__ import annotations

import torch
from torch import nn


class CNN2DDetokenizer(nn.Module):
    """Decode a structured [B, H_p, W_p, D] token grid back to [B, C, H, W] fields."""

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        grid_height: int,
        grid_width: int,
        patch_size: tuple[int, int] | int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.out_channels = int(out_channels)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        if isinstance(patch_size, int):
            self.patch_size = (int(patch_size), int(patch_size))
        else:
            self.patch_size = tuple(int(v) for v in patch_size)
        self.patch_height, self.patch_width = self.patch_size

        if self.grid_height % self.patch_height != 0 or self.grid_width % self.patch_width != 0:
            raise ValueError(
                "CNN2DDetokenizer requires grid_height and grid_width to be divisible by patch_size. "
                f"Received grid=({self.grid_height}, {self.grid_width}) and patch_size={self.patch_size}."
            )

        self.num_patch_rows = self.grid_height // self.patch_height
        self.num_patch_cols = self.grid_width // self.patch_width
        self.num_patches = self.num_patch_rows * self.num_patch_cols
        self.patch_area = self.patch_height * self.patch_width

        self.patch_decoder = nn.Linear(self.hidden_dim, self.out_channels * self.patch_area)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.patch_decoder.weight, std=0.02)
        nn.init.zeros_(self.patch_decoder.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        patch_values = self.patch_decoder(tokens)
        batch_size = int(patch_values.shape[0])
        out = patch_values.view(
            batch_size,
            self.num_patch_rows,
            self.num_patch_cols,
            self.out_channels,
            self.patch_height,
            self.patch_width,
        )
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
        return out.view(batch_size, self.out_channels, self.grid_height, self.grid_width)


class CNN3DDetokenizer(nn.Module):
    """Decode a structured [B, L_p, H_p, W_p, D] token grid back to [B, C, L, H, W] fields."""

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        grid_depth: int,
        grid_height: int,
        grid_width: int,
        patch_size: tuple[int, int, int] | int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.out_channels = int(out_channels)
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
                "CNN3DDetokenizer requires grid_depth, grid_height, and grid_width to be divisible by patch_size. "
                f"Received grid=({self.grid_depth}, {self.grid_height}, {self.grid_width}) "
                f"and patch_size={self.patch_size}."
            )

        self.num_patch_depths = self.grid_depth // self.patch_depth
        self.num_patch_rows = self.grid_height // self.patch_height
        self.num_patch_cols = self.grid_width // self.patch_width
        self.num_patches = self.num_patch_depths * self.num_patch_rows * self.num_patch_cols
        self.patch_volume = self.patch_depth * self.patch_height * self.patch_width

        self.patch_decoder = nn.Linear(self.hidden_dim, self.out_channels * self.patch_volume)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.patch_decoder.weight, std=0.02)
        nn.init.zeros_(self.patch_decoder.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        patch_values = self.patch_decoder(tokens)
        batch_size = int(patch_values.shape[0])
        out = patch_values.view(
            batch_size,
            self.num_patch_depths,
            self.num_patch_rows,
            self.num_patch_cols,
            self.out_channels,
            self.patch_depth,
            self.patch_height,
            self.patch_width,
        )
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return out.view(
            batch_size,
            self.out_channels,
            self.grid_depth,
            self.grid_height,
            self.grid_width,
        )
