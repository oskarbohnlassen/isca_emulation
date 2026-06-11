from __future__ import annotations

import torch
from torch import nn

from ..activations import get_activation


def _build_projection(
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    activation_module: nn.Module,
) -> nn.Sequential:
    n_layers = max(1, int(num_layers))
    layers: list[nn.Module] = []
    if n_layers == 1:
        layers.append(nn.Linear(in_dim, hidden_dim))
        return nn.Sequential(*layers)

    for layer_idx in range(n_layers):
        in_features = in_dim if layer_idx == 0 else hidden_dim
        layers.append(nn.Linear(in_features, hidden_dim))
        if layer_idx < n_layers - 1:
            layers.append(activation_module)
    return nn.Sequential(*layers)


class MeshGNNFeatureEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation_module: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = _build_projection(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation_module=activation_module,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def get_mesh_gnn_node_encoder(
    encoder_type: str,
    in_dim: int,
    hidden_dim: int,
    num_layers: int = 1,
    activation: str = "gelu",
) -> nn.Module:
    encoder_type = str(encoder_type).lower()
    if encoder_type == "linear":
        act = nn.Identity()
    elif encoder_type == "mlp":
        act = get_activation(activation)
    else:
        raise NotImplementedError(f"MeshGNN node encoder type '{encoder_type}' not implemented yet.")
    return MeshGNNFeatureEncoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation_module=act,
    )


def get_mesh_gnn_edge_encoder(
    encoder_type: str,
    in_dim: int,
    hidden_dim: int,
    num_layers: int = 1,
    activation: str = "gelu",
) -> nn.Module:
    encoder_type = str(encoder_type).lower()
    if encoder_type == "linear":
        act = nn.Identity()
    elif encoder_type == "mlp":
        act = get_activation(activation)
    else:
        raise NotImplementedError(f"MeshGNN edge encoder type '{encoder_type}' not implemented yet.")
    return MeshGNNFeatureEncoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation_module=act,
    )


class MeshGNNInputEncoders(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        hidden_dim: int,
        grid_node_feature_dim: int,
        mesh_node_feature_dim: int,
        g2m_edge_dim: int,
        mesh_edge_dim: int,
        m2g_edge_dim: int,
        node_encoder_type: str,
        edge_encoder_type: str,
        node_encoder_layers: int,
        edge_encoder_layers: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.grid_node_encoder = get_mesh_gnn_node_encoder(
            encoder_type=node_encoder_type,
            in_dim=int(in_channels) + int(grid_node_feature_dim),
            hidden_dim=self.hidden_dim,
            num_layers=node_encoder_layers,
            activation=activation,
        )
        self.mesh_node_encoder = get_mesh_gnn_node_encoder(
            encoder_type=node_encoder_type,
            in_dim=int(mesh_node_feature_dim),
            hidden_dim=self.hidden_dim,
            num_layers=node_encoder_layers,
            activation=activation,
        )
        self.g2m_edge_encoder = get_mesh_gnn_edge_encoder(
            encoder_type=edge_encoder_type,
            in_dim=int(g2m_edge_dim),
            hidden_dim=self.hidden_dim,
            num_layers=edge_encoder_layers,
            activation=activation,
        )
        self.mesh_edge_encoder = get_mesh_gnn_edge_encoder(
            encoder_type=edge_encoder_type,
            in_dim=int(mesh_edge_dim),
            hidden_dim=self.hidden_dim,
            num_layers=edge_encoder_layers,
            activation=activation,
        )
        self.m2g_edge_encoder = get_mesh_gnn_edge_encoder(
            encoder_type=edge_encoder_type,
            in_dim=int(m2g_edge_dim),
            hidden_dim=self.hidden_dim,
            num_layers=edge_encoder_layers,
            activation=activation,
        )

    def forward(
        self,
        *,
        x: torch.Tensor,
        grid_node_features: torch.Tensor,
        mesh_node_features: torch.Tensor,
        g2m_edge_features: torch.Tensor,
        mesh_edge_features: torch.Tensor,
        m2g_edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        grid_static = grid_node_features.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=x.dtype)
        mesh_static = mesh_node_features.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=x.dtype)

        g2m_edges = g2m_edge_features.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=x.dtype)
        mesh_edges = mesh_edge_features.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=x.dtype)
        m2g_edges = m2g_edge_features.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=x.dtype)

        grid_hidden = self.grid_node_encoder(torch.cat([x, grid_static], dim=-1))
        mesh_hidden = self.mesh_node_encoder(mesh_static)
        g2m_edge_hidden = self.g2m_edge_encoder(g2m_edges)
        mesh_edge_hidden = self.mesh_edge_encoder(mesh_edges)
        m2g_edge_hidden = self.m2g_edge_encoder(m2g_edges)

        return grid_hidden, mesh_hidden, g2m_edge_hidden, mesh_edge_hidden, m2g_edge_hidden
