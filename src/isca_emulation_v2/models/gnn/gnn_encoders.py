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
        out_features = hidden_dim
        layers.append(nn.Linear(in_features, out_features))
        if layer_idx < n_layers - 1:
            layers.append(activation_module)
    return nn.Sequential(*layers)


def get_node_encoder(
    encoder_type: str,
    in_dim: int,
    hidden_dim: int,
    num_layers: int = 1,
    activation: str = "gelu",
) -> nn.Module:
    if encoder_type not in {"linear", "mlp"}:
        raise ValueError("encoder_type must be one of ['linear', 'mlp'].")
    act = nn.Identity() if encoder_type == "linear" else get_activation(activation)
    return FeatureEncoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation_module=act,
    )


def get_edge_encoder(
    encoder_type: str,
    in_dim: int,
    hidden_dim: int,
    num_layers: int = 1,
    activation: str = "gelu",
) -> nn.Module:
    if encoder_type not in {"linear", "mlp"}:
        raise ValueError("encoder_type must be one of ['linear', 'mlp'].")
    
    act = nn.Identity() if encoder_type == "linear" else get_activation(activation)
    
    return FeatureEncoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation_module=act,
    )


class FeatureEncoder(nn.Module):
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
