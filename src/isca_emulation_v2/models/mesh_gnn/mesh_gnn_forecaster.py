from __future__ import annotations

import torch
from torch import nn

from .mesh_gnn_decoders import get_mesh_gnn_decoder
from .mesh_gnn_encoders import MeshGNNInputEncoders
from .mesh_gnn_mpnn import GridToMeshMPNN, MeshToGridMPNN, MeshToMeshMPNN


class MeshGNN2D(nn.Module):
    """MeshGNN-style grid->mesh->grid forecaster with edge-aware message passing at every stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_node_feature_dim: int,
        mesh_node_feature_dim: int,
        g2m_edge_dim: int,
        mesh_edge_dim: int,
        m2g_edge_dim: int,
        hidden_dim: int = 128,
        mpnn_layer_type: str = "GatedGCNConv",
        grid2mesh_num_layers: int = 1,
        mesh2mesh_num_layers: int = 6,
        mesh2grid_num_layers: int = 1,
        node_encoder_type: str = "mlp",
        edge_encoder_type: str = "mlp",
        decoder_type: str = "mlp",
        node_encoder_layers: int = 2,
        edge_encoder_layers: int = 2,
        decoder_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.grid_node_feature_dim = int(grid_node_feature_dim)
        self.mesh_node_feature_dim = int(mesh_node_feature_dim)
        self.g2m_edge_dim = int(g2m_edge_dim)
        self.mesh_edge_dim = int(mesh_edge_dim)
        self.m2g_edge_dim = int(m2g_edge_dim)
        self.hidden_dim = int(hidden_dim)

        self.encoders = MeshGNNInputEncoders(
            in_channels=self.in_channels,
            hidden_dim=self.hidden_dim,
            grid_node_feature_dim=self.grid_node_feature_dim,
            mesh_node_feature_dim=self.mesh_node_feature_dim,
            g2m_edge_dim=self.g2m_edge_dim,
            mesh_edge_dim=self.mesh_edge_dim,
            m2g_edge_dim=self.m2g_edge_dim,
            node_encoder_type=node_encoder_type,
            edge_encoder_type=edge_encoder_type,
            node_encoder_layers=node_encoder_layers,
            edge_encoder_layers=edge_encoder_layers,
            activation=activation,
        )
        self.grid_to_mesh = GridToMeshMPNN(
            hidden_dim=self.hidden_dim,
            layer_type=mpnn_layer_type,
            num_layers=grid2mesh_num_layers,
            activation=activation,
            dropout=dropout,
        )
        self.mesh_to_mesh = MeshToMeshMPNN(
            hidden_dim=self.hidden_dim,
            layer_type=mpnn_layer_type,
            num_layers=mesh2mesh_num_layers,
            activation=activation,
            dropout=dropout,
        )
        self.mesh_to_grid = MeshToGridMPNN(
            hidden_dim=self.hidden_dim,
            layer_type=mpnn_layer_type,
            num_layers=mesh2grid_num_layers,
            activation=activation,
            dropout=dropout,
        )
        self.decoder = get_mesh_gnn_decoder(
            decoder_type=decoder_type,
            in_dim=self.hidden_dim,
            out_dim=self.out_channels,
            hidden_dim=self.hidden_dim,
            num_layers=decoder_layers,
            activation=activation,
        )

        self.register_buffer("grid_node_features", torch.empty((0, 0), dtype=torch.float32), persistent=False)
        self.register_buffer("mesh_node_features", torch.empty((0, 0), dtype=torch.float32), persistent=False)
        self.register_buffer("g2m_edge_index", torch.empty((2, 0), dtype=torch.long), persistent=False)
        self.register_buffer("g2m_edge_features", torch.empty((0, 0), dtype=torch.float32), persistent=False)
        self.register_buffer("mesh_edge_index", torch.empty((2, 0), dtype=torch.long), persistent=False)
        self.register_buffer("mesh_edge_features", torch.empty((0, 0), dtype=torch.float32), persistent=False)
        self.register_buffer("m2g_edge_index", torch.empty((2, 0), dtype=torch.long), persistent=False)
        self.register_buffer("m2g_edge_features", torch.empty((0, 0), dtype=torch.float32), persistent=False)
        self._static_graph_ready = False

    def set_static_graph(
        self,
        *,
        grid_node_features: torch.Tensor,
        mesh_node_features: torch.Tensor,
        g2m_edge_index: torch.Tensor,
        g2m_edge_features: torch.Tensor,
        mesh_edge_index: torch.Tensor,
        mesh_edge_features: torch.Tensor,
        m2g_edge_index: torch.Tensor,
        m2g_edge_features: torch.Tensor,
    ) -> None:
        model_device = next(self.parameters()).device
        self.grid_node_features = grid_node_features.to(device=model_device, dtype=torch.float32)
        self.mesh_node_features = mesh_node_features.to(device=model_device, dtype=torch.float32)
        self.g2m_edge_index = g2m_edge_index.to(device=model_device, dtype=torch.long)
        self.g2m_edge_features = g2m_edge_features.to(device=model_device, dtype=torch.float32)
        self.mesh_edge_index = mesh_edge_index.to(device=model_device, dtype=torch.long)
        self.mesh_edge_features = mesh_edge_features.to(device=model_device, dtype=torch.float32)
        self.m2g_edge_index = m2g_edge_index.to(device=model_device, dtype=torch.long)
        self.m2g_edge_features = m2g_edge_features.to(device=model_device, dtype=torch.float32)
        self._static_graph_ready = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._static_graph_ready:
            raise RuntimeError("Static graph must be set before calling forward on MeshGNN2D.")

        x_state = x[..., : self.out_channels]
        grid_hidden, mesh_hidden, g2m_edge_hidden, mesh_edge_hidden, m2g_edge_hidden = self.encoders(
            x=x,
            grid_node_features=self.grid_node_features,
            mesh_node_features=self.mesh_node_features,
            g2m_edge_features=self.g2m_edge_features,
            mesh_edge_features=self.mesh_edge_features,
            m2g_edge_features=self.m2g_edge_features,
        )

        mesh_hidden, _ = self.grid_to_mesh(
            grid_hidden=grid_hidden,
            mesh_hidden=mesh_hidden,
            edge_index=self.g2m_edge_index,
            edge_hidden=g2m_edge_hidden,
        )
        mesh_hidden, _ = self.mesh_to_mesh(
            mesh_hidden=mesh_hidden,
            edge_index=self.mesh_edge_index,
            edge_hidden=mesh_edge_hidden,
        )
        grid_hidden, _ = self.mesh_to_grid(
            mesh_hidden=mesh_hidden,
            grid_hidden=grid_hidden,
            edge_index=self.m2g_edge_index,
            edge_hidden=m2g_edge_hidden,
        )

        delta = self.decoder(grid_hidden)
        return x_state + delta

