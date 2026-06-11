from __future__ import annotations

import torch
from torch import nn

from isca_emulation_v2.models.activations import get_activation
from isca_emulation_v2.models.transformer.swin_transformer_block_2d import ShiftedWindowAttention
from isca_emulation_v2.models.transformer.swin_transformer_block_3d import ShiftedWindowAttention3d


class GlobalAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        activation: str,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm_1(x)
        attn_out, _ = self.attention(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.norm_2(x))
        return x


class GlobalAttentionTokenProcessor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        activation: str,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        mlp_hidden_dim = max(1, int(round(self.hidden_dim * self.mlp_ratio)))

        self.blocks = nn.ModuleList(
            [
                GlobalAttentionTransformerBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    activation=activation,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


class SwinTokenProcessorBlock3D(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        window_size: tuple[int, ...],
        shift_size: tuple[int, ...],
        activation: str,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.window_size = tuple(int(v) for v in window_size)
        self.shift_size = tuple(int(v) for v in shift_size)

        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.attention = ShiftedWindowAttention3d(
            dim=hidden_dim,
            window_size=list(self.window_size),
            shift_size=list(self.shift_size),
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            dropout=0.0,
        )
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm_1(tokens)
        x = self.attention(x)
        tokens = tokens + self.dropout(x)
        tokens = tokens + self.mlp(self.norm_2(tokens))
        return tokens


class SwinTokenProcessorBlock2D(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        window_size: tuple[int, ...],
        shift_size: tuple[int, ...],
        activation: str,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.window_size = tuple(int(v) for v in window_size)
        self.shift_size = tuple(int(v) for v in shift_size)

        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.attention = ShiftedWindowAttention(
            dim=hidden_dim,
            window_size=list(self.window_size),
            shift_size=list(self.shift_size),
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            dropout=0.0,
        )
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm_1(tokens)
        x = self.attention(x)
        tokens = tokens + self.dropout(x)
        tokens = tokens + self.mlp(self.norm_2(tokens))
        return tokens


class SwinTokenProcessor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        window_size: tuple[int, ...],
        activation: str,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.window_size = tuple(int(v) for v in window_size)
        self.shift_size = tuple(v // 2 for v in self.window_size)
        self.spatial_rank = len(self.window_size)
        if self.spatial_rank == 2:
            self.block_cls = SwinTokenProcessorBlock2D
            self.block_window_size = self.window_size
            self.block_shift_size = self.shift_size
        elif self.spatial_rank == 3:
            self.block_cls = SwinTokenProcessorBlock3D
            self.block_window_size = self.window_size
            self.block_shift_size = self.shift_size
        else:
            raise ValueError(
                "SwinTokenProcessor currently supports 2D and 3D token grids. "
                f"Received window_size={self.window_size}."
            )

        mlp_hidden_dim = max(1, int(round(self.hidden_dim * self.mlp_ratio)))

        self.blocks = nn.ModuleList(
            [
                self.block_cls(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    window_size=self.block_window_size,
                    shift_size=((0,) * self.spatial_rank) if layer_idx % 2 == 0 else self.block_shift_size,
                    activation=activation,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for layer_idx in range(self.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)
