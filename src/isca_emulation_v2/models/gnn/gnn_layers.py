from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, MessagePassing
from torch_geometric.utils import scatter

from ..activations import get_activation


def get_gnn_block(
    block_type: str,
    hidden_dim: int,
    edge_dim: int,
    num_layers: int,
    num_heads: int = 4,
    activation: str = "gelu",
    dropout: float = 0.0,
) -> nn.Module:
    block_type = str(block_type)

    if block_type == "GCNResBlock":
        return GCNResBlock(hidden_dim=hidden_dim, num_layers=num_layers, activation=activation, dropout=dropout)
    if block_type == "GATResBlock":
        return GATResBlock(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            activation=activation,
            dropout=dropout,
        )
    if block_type == "GatedGCNResBlock":
        return GatedGCNResBlock(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )
    raise ValueError(
        "block_type must be one of "
        "['GCNResBlock', 'GATResBlock', 'GatedGCNResBlock'] "
    )


class GCNResBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, activation: str = "gelu", dropout: float = 0.0) -> None:
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(int(num_layers))])
        self.residuals = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(int(num_layers))])
        self.activation = get_activation(activation)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None) -> torch.Tensor:
        _ = edge_attr
        h = x
        for conv, residual in zip(self.convs, self.residuals):
            h_res = h
            h = conv(h, edge_index)
            h = self.activation(h + residual(h_res))
            if self.dropout > 0.0:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GATResBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_layers: int,
        num_heads: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        head_dim = hidden_dim // int(num_heads)
        conv_edge_dim = edge_dim if edge_dim > 0 else None
        self.convs = nn.ModuleList(
            [
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=head_dim,
                    heads=int(num_heads),
                    concat=True,
                    edge_dim=conv_edge_dim,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.residuals = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(int(num_layers))])
        self.activation = get_activation(activation)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None) -> torch.Tensor:
        h = x
        for conv, residual in zip(self.convs, self.residuals):
            h_res = h
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = self.activation(h + residual(h_res))
            if self.dropout > 0.0:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GatedGCNResBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_layers: int,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.edge_in_proj = nn.Identity() if edge_dim == hidden_dim else nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GatedGCNConv(hidden_dim=hidden_dim, activation=activation, dropout=dropout) for _ in range(int(num_layers))]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None) -> torch.Tensor:
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.shape[1], x.shape[-1]), dtype=x.dtype, device=x.device)
        e = self.edge_in_proj(edge_attr)
        h = x
        for layer in self.layers:
            h, e = layer(h, edge_index, e)
        return h


class GatedGCNConv(MessagePassing):
    def __init__(self, hidden_dim: int, activation: str = "gelu", dropout: float = 0.0) -> None:
        super().__init__(aggr="add")
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.activation = get_activation(activation)

        self.A = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.B = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.C = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.D = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.E = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

        self.bn_node = nn.BatchNorm1d(self.hidden_dim)
        self.bn_edge = nn.BatchNorm1d(self.hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index  # src -> dst

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(edge_attr)
        Dx = self.D(x)
        Ex = self.E(x)

        # i = dst (receiver), j = src (sender)
        e_ij = Dx[dst] + Ex[src] + Ce
        sigma_ij = torch.sigmoid(e_ij)
        msg = sigma_ij * Bx[src]

        node_update = scatter(msg, dst, dim=0, dim_size=x.size(0), reduce="sum")
        node_out = Ax + node_update

        node_out = self.bn_node(node_out)
        edge_out = self.bn_edge(e_ij)

        node_out = self.activation(node_out)
        edge_out = self.activation(edge_out)

        if self.dropout > 0:
            node_out = F.dropout(node_out, p=self.dropout, training=self.training)
            edge_out = F.dropout(edge_out, p=self.dropout, training=self.training)

        node_out = node_out + x
        edge_out = edge_out + edge_attr
        return node_out, edge_out
