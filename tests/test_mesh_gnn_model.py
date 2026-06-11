import importlib.machinery
import sys
import types

import torch
from omegaconf import OmegaConf

if "xarray" not in sys.modules:
    xr_stub = types.ModuleType("xarray")
    xr_stub.DataArray = type("DataArray", (), {})
    xr_stub.Dataset = type("Dataset", (), {})
    xr_stub.__spec__ = importlib.machinery.ModuleSpec("xarray", loader=None)
    sys.modules["xarray"] = xr_stub

from isca_emulation_v2.models.mesh_gnn import MeshGNN2D
from isca_emulation_v2.models.model import load_model
from isca_emulation_v2.utils import get_data_attributes_mesh_gnn


class _FakeDataset:
    def __init__(self):
        self.grid_static = {"shape": [4, 5], "node_features": torch.randn(20, 7)}
        self.mesh_static = {"node_features": torch.randn(8, 7)}
        self.grid2mesh = {"edge_features": torch.randn(30, 5)}
        self.mesh_graph = {"edge_features": torch.randn(40, 5)}
        self.mesh2grid = {"edge_features": torch.randn(60, 5)}


def _make_static_graph():
    grid_node_features = torch.randn(20, 7)
    mesh_node_features = torch.randn(8, 7)
    g2m_edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 0, 1, 1, 2, 2, 3, 3],
        ],
        dtype=torch.long,
    )
    mesh_edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ],
        dtype=torch.long,
    )
    m2g_edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ],
        dtype=torch.long,
    )
    return {
        "grid_node_features": grid_node_features,
        "mesh_node_features": mesh_node_features,
        "g2m_edge_index": g2m_edge_index,
        "g2m_edge_features": torch.randn(g2m_edge_index.shape[1], 5),
        "mesh_edge_index": mesh_edge_index,
        "mesh_edge_features": torch.randn(mesh_edge_index.shape[1], 5),
        "m2g_edge_index": m2g_edge_index,
        "m2g_edge_features": torch.randn(m2g_edge_index.shape[1], 5),
    }


def test_mesh_gnn_forward_shape():
    model = MeshGNN2D(
        in_channels=12,
        out_channels=12,
        grid_node_feature_dim=7,
        mesh_node_feature_dim=7,
        g2m_edge_dim=5,
        mesh_edge_dim=5,
        m2g_edge_dim=5,
        hidden_dim=16,
        mpnn_layer_type="GatedGCNConv",
        grid2mesh_num_layers=1,
        mesh2mesh_num_layers=2,
        mesh2grid_num_layers=1,
        node_encoder_type="mlp",
        edge_encoder_type="mlp",
        decoder_type="mlp",
        node_encoder_layers=2,
        edge_encoder_layers=2,
        decoder_layers=2,
    )
    model.set_static_graph(**_make_static_graph())

    x = torch.randn(3, 20, 12)
    y = model(x)

    assert y.shape == (3, 20, 12)
    assert torch.isfinite(y).all()


def test_load_model_builds_mesh_gnn():
    cfg = OmegaConf.create(
        {
            "model": {
                "model_type": "MeshGNN2D",
                "hidden_dim": 16,
                "mpnn_layer_type": "GatedGCNConv",
                "grid2mesh_num_layers": 1,
                "mesh2mesh_num_layers": 2,
                "mesh2grid_num_layers": 1,
                "node_encoder_type": "mlp",
                "edge_encoder_type": "mlp",
                "decoder_type": "mlp",
                "node_encoder_layers": 2,
                "edge_encoder_layers": 2,
                "decoder_layers": 2,
                "activation": "gelu",
                "dropout": 0.0,
            },
            "data": {
                "in_channels": 12,
                "out_channels": 12,
                "grid_node_feature_dim": 7,
                "mesh_node_feature_dim": 7,
                "g2m_edge_feature_dim": 5,
                "mesh_edge_feature_dim": 5,
                "m2g_edge_feature_dim": 5,
            },
        }
    )

    model = load_model(cfg)

    assert isinstance(model, MeshGNN2D)


def test_get_data_attributes_mesh_gnn_populates_cfg():
    cfg = OmegaConf.create({"data": {}})
    data_point = (torch.randn(2, 20, 12), torch.randn(2, 20, 12))
    dataset = _FakeDataset()

    cfg = get_data_attributes_mesh_gnn(cfg, data_point, dataset)

    assert int(cfg.data.in_channels) == 12
    assert int(cfg.data.out_channels) == 12
    assert int(cfg.data.grid_height) == 4
    assert int(cfg.data.grid_width) == 5
    assert int(cfg.data.mesh_num_nodes) == 8
