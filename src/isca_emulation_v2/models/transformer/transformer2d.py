import torch
from torch import nn

from .detokenizer import CNN2DDetokenizer
from .token_processor import GlobalAttentionTokenProcessor, SwinTokenProcessor
from .tokenizer import CNN2DTokenizer


class Transformer2DGlobalAttentionForecaster(nn.Module):
    """Patch-based global-attention transformer forecaster for [B, C, H, W] climate fields."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        grid_height: int,
        grid_width: int,
        hidden_dim: int = 128,
        patch_size: tuple[int, int] | int = (4, 4),
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        activation: str = "gelu",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.out_channels = int(out_channels)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout_rate = float(dropout)
        self.attention_dropout = float(attention_dropout)
        if isinstance(patch_size, int):
            self.patch_size = (int(patch_size), int(patch_size))
        else:
            self.patch_size = tuple(int(v) for v in patch_size)
        self.patch_height, self.patch_width = self.patch_size

        if self.num_heads < 1 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.tokenizer = CNN2DTokenizer(
            channels=self.channels,
            hidden_dim=self.hidden_dim,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            patch_size=self.patch_size,
        )
        self.detokenizer = CNN2DDetokenizer(
            hidden_dim=self.hidden_dim,
            out_channels=self.out_channels,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            patch_size=self.patch_size,
        )
        self.num_patch_rows = self.tokenizer.num_patch_rows
        self.num_patch_cols = self.tokenizer.num_patch_cols
        self.num_patches = self.tokenizer.num_patches
        self.patch_area = self.tokenizer.patch_area

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_dim))
        self.input_dropout = nn.Dropout(self.dropout_rate)
        self.token_processor = GlobalAttentionTokenProcessor(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            activation=activation,
            dropout=self.dropout_rate,
            attention_dropout=self.attention_dropout,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _flatten_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = int(tokens.shape[0])
        return tokens.view(batch_size, self.num_patches, self.hidden_dim)

    def _unflatten_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = int(tokens.shape[0])
        return tokens.view(batch_size, self.num_patch_rows, self.num_patch_cols, self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        tokens = self._flatten_tokens(tokens)
        tokens = self.input_dropout(tokens + self.pos_embed)
        tokens = self.token_processor(tokens)
        tokens = self._unflatten_tokens(tokens)
        delta = self.detokenizer(tokens)
        x_state = x[:, : self.out_channels]
        return x_state + delta


class Transformer2DSwinForecaster(nn.Module):
    """Patch-based shifted-window transformer forecaster for [B, C, H, W] climate fields."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        grid_height: int,
        grid_width: int,
        hidden_dim: int = 128,
        patch_size: tuple[int, int] | int = (4, 4),
        window_size: tuple[int, int] | int = (4, 4),
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        activation: str = "gelu",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.out_channels = int(out_channels)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout_rate = float(dropout)
        self.attention_dropout = float(attention_dropout)

        if isinstance(patch_size, int):
            self.patch_size = (int(patch_size), int(patch_size))
        else:
            self.patch_size = tuple(int(v) for v in patch_size)
        if isinstance(window_size, int):
            self.window_size = (int(window_size), int(window_size))
        else:
            self.window_size = tuple(int(v) for v in window_size)

        if self.num_heads < 1 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.tokenizer = CNN2DTokenizer(
            channels=self.channels,
            hidden_dim=self.hidden_dim,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            patch_size=self.patch_size,
        )
        self.detokenizer = CNN2DDetokenizer(
            hidden_dim=self.hidden_dim,
            out_channels=self.out_channels,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            patch_size=self.patch_size,
        )
        self.num_patch_rows = self.tokenizer.num_patch_rows
        self.num_patch_cols = self.tokenizer.num_patch_cols

        if (
            self.num_patch_rows % self.window_size[0] != 0
            or self.num_patch_cols % self.window_size[1] != 0
        ):
            raise ValueError(
                "window_size must divide the 2D token grid. "
                f"Received token_grid=({self.num_patch_rows}, {self.num_patch_cols}) "
                f"and window_size={self.window_size}."
            )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patch_rows, self.num_patch_cols, self.hidden_dim)
        )
        self.input_dropout = nn.Dropout(self.dropout_rate)
        self.token_processor = SwinTokenProcessor(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            window_size=self.window_size,
            activation=activation,
            dropout=self.dropout_rate,
            attention_dropout=self.attention_dropout,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        tokens = self.input_dropout(tokens + self.pos_embed)
        tokens = self.token_processor(tokens)
        delta = self.detokenizer(tokens)
        x_state = x[:, : self.out_channels]
        return x_state + delta
