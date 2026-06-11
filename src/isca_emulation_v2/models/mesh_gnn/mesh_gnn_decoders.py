from __future__ import annotations

import torch
from torch import nn

from ..activations import get_activation


def _build_projection(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
    activation_module: nn.Module,
) -> nn.Sequential:
    n_layers = max(1, int(num_layers))
    layers: list[nn.Module] = []
    if n_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    for layer_idx in range(n_layers):
        if layer_idx == 0:
            in_features = in_dim
            out_features = hidden_dim
        elif layer_idx == n_layers - 1:
            in_features = hidden_dim
            out_features = out_dim
        else:
            in_features = hidden_dim
            out_features = hidden_dim
        layers.append(nn.Linear(in_features, out_features))
        if layer_idx < n_layers - 1:
            layers.append(activation_module)
    return nn.Sequential(*layers)


class MeshGNNNodeDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        activation_module: nn.Module,
    ) -> None:
        super().__init__()
        self.decoder = _build_projection(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            activation_module=activation_module,
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        return self.decoder(node_embeddings)


def get_mesh_gnn_decoder(
    decoder_type: str,
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    num_layers: int = 1,
    activation: str = "gelu",
) -> nn.Module:
    decoder_type = str(decoder_type).lower()
    if decoder_type == "linear":
        act = nn.Identity()
    elif decoder_type == "mlp":
        act = get_activation(activation)
    else:
        raise NotImplementedError(f"MeshGNN decoder type '{decoder_type}' not implemented yet.")
    return MeshGNNNodeDecoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=num_layers,
        activation_module=act,
    )
