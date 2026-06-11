from __future__ import annotations

import torch
from torch import nn

from isca_emulation_v2.models.gnn.gnn_decoders import get_decoder
from isca_emulation_v2.models.gnn.gnn_encoders import get_edge_encoder, get_node_encoder
from isca_emulation_v2.models.gnn.gnn_layers import get_gnn_block


class SimpleGNN2D(nn.Module):
    """Simple GNN2D composed from encoder, message-passing blocks, and decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        batch_size: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
        gnn_layer_type: str = "GATResBlock",
        node_encoder_type: str = "linear",
        edge_encoder_type: str = "linear",
        decoder_type: str = "linear",
        edge_encoder_hidden_dim: int | None = None,
        node_encoder_layers: int = 1,
        edge_encoder_layers: int = 1,
        decoder_layers: int = 1,
        use_edge_features: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_nodes = int(num_nodes)
        self.batch_size = int(batch_size)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.gnn_layer_type = str(gnn_layer_type)
        self.use_edge_features = bool(use_edge_features)
        self.edge_encoder_hidden_dim = int(edge_encoder_hidden_dim) if edge_encoder_hidden_dim is not None else self.hidden_dim

        self.node_encoder = get_node_encoder(
            encoder_type=node_encoder_type,
            in_dim=self.in_channels,
            hidden_dim=self.hidden_dim,
            num_layers=node_encoder_layers,
            activation=activation,
        )
        self.edge_encoder = (
            get_edge_encoder(
                encoder_type=edge_encoder_type,
                in_dim=self.edge_dim,
                hidden_dim=self.edge_encoder_hidden_dim,
                num_layers=edge_encoder_layers,
                activation=activation,
            )
            if (self.edge_dim > 0 and self.use_edge_features)
            else None
        )

        edge_hidden_dim = self.edge_encoder_hidden_dim if (self.edge_dim > 0 and self.use_edge_features) else 0

        self.gnn_block = get_gnn_block(
            block_type=self.gnn_layer_type,
            hidden_dim=self.hidden_dim,
            edge_dim=edge_hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            activation=activation,
            dropout=self.dropout,
        )
        self.decoder = get_decoder(
            decoder_type=decoder_type,
            in_dim=self.hidden_dim,
            out_dim=self.out_channels,
            hidden_dim=self.hidden_dim,
            num_layers=decoder_layers,
            activation=activation,
        )

        self.edge_index: torch.Tensor | None = None
        self.edge_index_batch: torch.Tensor | None = None
        self.edge_attr: torch.Tensor | None = None

    def _expand_edge_index(self, batch_size: int) -> torch.Tensor:
        node_offsets = torch.arange(
            int(batch_size),
            device=self.edge_index.device,
            dtype=self.edge_index.dtype,
        ) * self.num_nodes
        num_edges = self.edge_index.shape[1]
        edge_index_batch = self.edge_index.unsqueeze(0) + node_offsets.view(int(batch_size), 1, 1)
        return edge_index_batch.permute(1, 0, 2).reshape(2, int(batch_size) * num_edges)

    def set_graph(self, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None) -> None:
        model_device = next(self.parameters()).device
        self.edge_index = edge_index.to(device=model_device, dtype=torch.long)
        self.edge_index_batch = self._expand_edge_index(self.batch_size)
        if self.edge_dim > 0 and self.use_edge_features:
            if edge_attr is None:
                raise ValueError("edge_attr is required when edge_dim > 0.")
            self.edge_attr = edge_attr.to(device=model_device, dtype=torch.float32)
        else:
            self.edge_attr = torch.empty((self.edge_index.shape[1], 0), device=model_device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape

        x_state = x[..., : self.out_channels]
        x_flat = x.reshape(batch_size * num_nodes, -1)
        h = self.node_encoder(x_flat)

        edge_index_batch = self.edge_index_batch if batch_size == self.batch_size else self._expand_edge_index(batch_size)
        edge_attr_hidden = self.edge_encoder(self.edge_attr) if self.edge_encoder is not None else None
        edge_attr_batch = edge_attr_hidden.repeat(batch_size, 1) if edge_attr_hidden is not None else None

        h = self.gnn_block(h, edge_index_batch, edge_attr_batch)
        delta = self.decoder(h).reshape(batch_size, num_nodes, self.out_channels)
        return x_state + delta
