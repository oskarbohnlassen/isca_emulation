from __future__ import annotations

import torch
from torch import nn

from ..gnn.gnn_layers import GatedGCNConv


def _expand_homogeneous_edge_index(
    edge_index: torch.Tensor,
    *,
    batch_size: int,
    num_nodes: int,
) -> torch.Tensor:
    device = edge_index.device
    dtype = edge_index.dtype
    node_offsets = torch.arange(batch_size, device=device, dtype=dtype) * int(num_nodes)
    return (edge_index.unsqueeze(0) + node_offsets.view(batch_size, 1, 1)).permute(1, 0, 2).reshape(2, -1)


def _expand_bipartite_edge_index(
    edge_index: torch.Tensor,
    *,
    batch_size: int,
    num_sender_nodes: int,
    num_receiver_nodes: int,
) -> torch.Tensor:
    device = edge_index.device
    dtype = edge_index.dtype
    total_nodes = int(num_sender_nodes) + int(num_receiver_nodes)
    node_offsets = torch.arange(batch_size, device=device, dtype=dtype) * total_nodes
    expanded = edge_index.unsqueeze(0).repeat(batch_size, 1, 1)
    expanded[:, 1, :] = expanded[:, 1, :] + int(num_sender_nodes)
    expanded = expanded + node_offsets.view(batch_size, 1, 1)
    return expanded.permute(1, 0, 2).reshape(2, -1)


def get_mesh_gnn_mpnn_layer(
    layer_type: str,
    hidden_dim: int,
    activation: str = "gelu",
    dropout: float = 0.0,
) -> nn.Module:
    layer_type = str(layer_type)
    if layer_type == "GatedGCNConv":
        return GatedGCNConv(hidden_dim=hidden_dim, activation=activation, dropout=dropout)
    raise NotImplementedError(f"MeshGNN MPNN layer type '{layer_type}' not implemented yet.")


class GridToMeshMPNN(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        layer_type: str,
        num_layers: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                get_mesh_gnn_mpnn_layer(
                    layer_type=layer_type,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(
        self,
        *,
        grid_hidden: torch.Tensor,
        mesh_hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_grid_nodes, hidden_dim = grid_hidden.shape
        num_mesh_nodes = mesh_hidden.shape[1]
        combined = torch.cat([grid_hidden, mesh_hidden], dim=1)
        flat_nodes = combined.reshape(batch_size * (num_grid_nodes + num_mesh_nodes), hidden_dim)
        edge_index_batch = _expand_bipartite_edge_index(
            edge_index,
            batch_size=batch_size,
            num_sender_nodes=num_grid_nodes,
            num_receiver_nodes=num_mesh_nodes,
        )
        flat_edges = edge_hidden.reshape(batch_size * edge_hidden.shape[1], hidden_dim)
        for layer in self.layers:
            flat_nodes, flat_edges = layer(flat_nodes, edge_index_batch, flat_edges)
        combined = flat_nodes.view(batch_size, num_grid_nodes + num_mesh_nodes, hidden_dim)
        updated_mesh = combined[:, num_grid_nodes:, :]
        updated_edges = flat_edges.view(batch_size, edge_hidden.shape[1], hidden_dim)
        return updated_mesh, updated_edges


class MeshToMeshMPNN(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        layer_type: str,
        num_layers: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                get_mesh_gnn_mpnn_layer(
                    layer_type=layer_type,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(
        self,
        *,
        mesh_hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_mesh_nodes, hidden_dim = mesh_hidden.shape
        edge_index_batch = _expand_homogeneous_edge_index(
            edge_index,
            batch_size=batch_size,
            num_nodes=num_mesh_nodes,
        )
        flat_nodes = mesh_hidden.reshape(batch_size * num_mesh_nodes, hidden_dim)
        flat_edges = edge_hidden.reshape(batch_size * edge_hidden.shape[1], hidden_dim)
        for layer in self.layers:
            flat_nodes, flat_edges = layer(flat_nodes, edge_index_batch, flat_edges)
        updated_mesh = flat_nodes.view(batch_size, num_mesh_nodes, hidden_dim)
        updated_edges = flat_edges.view(batch_size, edge_hidden.shape[1], hidden_dim)
        return updated_mesh, updated_edges


class MeshToGridMPNN(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        layer_type: str,
        num_layers: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                get_mesh_gnn_mpnn_layer(
                    layer_type=layer_type,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(
        self,
        *,
        mesh_hidden: torch.Tensor,
        grid_hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_mesh_nodes, hidden_dim = mesh_hidden.shape
        num_grid_nodes = grid_hidden.shape[1]
        combined = torch.cat([mesh_hidden, grid_hidden], dim=1)
        flat_nodes = combined.reshape(batch_size * (num_mesh_nodes + num_grid_nodes), hidden_dim)
        edge_index_batch = _expand_bipartite_edge_index(
            edge_index,
            batch_size=batch_size,
            num_sender_nodes=num_mesh_nodes,
            num_receiver_nodes=num_grid_nodes,
        )
        flat_edges = edge_hidden.reshape(batch_size * edge_hidden.shape[1], hidden_dim)
        for layer in self.layers:
            flat_nodes, flat_edges = layer(flat_nodes, edge_index_batch, flat_edges)
        combined = flat_nodes.view(batch_size, num_mesh_nodes + num_grid_nodes, hidden_dim)
        updated_grid = combined[:, num_mesh_nodes:, :]
        updated_edges = flat_edges.view(batch_size, edge_hidden.shape[1], hidden_dim)
        return updated_grid, updated_edges
