from pathlib import Path
import importlib.machinery
import sys
import types

import numpy as np
import torch

if "xarray" not in sys.modules:
    xr_stub = types.ModuleType("xarray")
    xr_stub.DataArray = type("DataArray", (), {})
    xr_stub.Dataset = type("Dataset", (), {})
    xr_stub.__spec__ = importlib.machinery.ModuleSpec("xarray", loader=None)
    sys.modules["xarray"] = xr_stub

from isca_emulation_v2.data import mesh_gnn as gc


class FakeCoord:
    def __init__(self, values):
        self.values = np.asarray(values)


class FakeDataArray:
    def __init__(self, values, dims):
        self._values = np.asarray(values)
        self._dims = tuple(dims)

    def transpose(self, *dims):
        axes = [self._dims.index(dim) for dim in dims]
        return FakeDataArray(np.transpose(self._values, axes=axes), dims)

    def load(self):
        return self

    def to_numpy(self):
        return np.asarray(self._values)


class FakeDataset:
    def __init__(self, data_vars, coords):
        self._data_vars = dict(data_vars)
        self._coords = {key: np.asarray(value) for key, value in coords.items()}
        self.sizes = {key: int(value.shape[0]) for key, value in self._coords.items()}

    def __getitem__(self, key):
        if key in self._data_vars:
            return self._data_vars[key]
        if key in self._coords:
            return FakeCoord(self._coords[key])
        raise KeyError(key)

    def isel(self, indexers):
        next_data_vars = {}
        for key, arr in self._data_vars.items():
            values = arr._values
            dims = arr._dims
            slices = []
            for dim in dims:
                selector = indexers.get(dim, slice(None))
                slices.append(selector)
            next_data_vars[key] = FakeDataArray(values[tuple(slices)], dims)

        next_coords = {}
        for key, values in self._coords.items():
            selector = indexers.get(key, slice(None))
            next_coords[key] = values[selector]
        return FakeDataset(next_data_vars, next_coords)


def _split_ranges(total_time: int, split_cfg):
    if split_cfg is False:
        return {"all": (0, total_time)}
    n_train = int(total_time * float(split_cfg[0]))
    n_val = int(total_time * float(split_cfg[1]))
    n_test = total_time - n_train - n_val
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, total_time),
    }


def test_build_mesh_gnn_static_graph_shapes():
    lat = np.linspace(-60.0, 60.0, 4, dtype=np.float32)
    lon = np.linspace(0.0, 300.0, 6, dtype=np.float32)

    static = gc.build_mesh_gnn_static_graph(lat, lon, mesh_splits=1)

    assert int(static["grid"]["num_nodes"]) == 24
    assert int(static["mesh"]["num_nodes"]) == 42
    assert int(static["grid2mesh"]["edge_index"].shape[0]) == 2
    assert int(static["mesh_graph"]["edge_index"].shape[0]) == 2
    assert int(static["mesh2grid"]["num_edges"]) == 24 * 3
    assert tuple(static["grid"]["node_features"].shape) == (24, 7)
    assert int(static["mesh_graph"]["edge_features"].shape[0]) == int(static["mesh_graph"]["num_edges"])
    assert int(static["mesh_graph"]["edge_features"].shape[1]) == 5
    assert static["feature_schema"]["edge_feature_mode"] == "global_xyz"


def test_build_mesh_gnn_static_graph_receiver_local_edge_features():
    lat = np.linspace(-60.0, 60.0, 4, dtype=np.float32)
    lon = np.linspace(0.0, 300.0, 6, dtype=np.float32)

    static = gc.build_mesh_gnn_static_graph(lat, lon, mesh_splits=1, edge_feature_mode="receiver_local")

    edge_features = static["mesh_graph"]["edge_features"]
    assert int(edge_features.shape[0]) == int(static["mesh_graph"]["num_edges"])
    assert int(edge_features.shape[1]) == 4
    assert static["feature_schema"]["edge_feature_mode"] == "receiver_local"
    assert static["feature_schema"]["edge_features"] == [
        "local_distance",
        "local_receiver_x",
        "local_receiver_y",
        "local_receiver_z",
    ]
    assert torch.all(edge_features[:, 0] >= 0.0)
    assert torch.all(edge_features[:, 0] <= 1.0 + 1e-6)


def test_simple_dataset_mesh_gnn_streams_pairs(tmp_path: Path):
    static_path = tmp_path / "mesh_gnn_static.pt"
    torch.save(
        {
            "grid": {"num_nodes": 8},
            "mesh": {"num_nodes": 12},
            "grid2mesh": {"edge_index": torch.zeros((2, 1), dtype=torch.long)},
            "mesh_graph": {"edge_index": torch.zeros((2, 1), dtype=torch.long)},
            "mesh2grid": {"edge_index": torch.zeros((2, 1), dtype=torch.long)},
        },
        static_path,
    )
    shard_path = tmp_path / "train_000.pt"
    torch.save({"x_dyn": torch.arange(5 * 8 * 3, dtype=torch.float32).reshape(5, 8, 3)}, shard_path)

    dataset = gc.SimpleDatasetMeshGNN(
        shard_paths=[str(shard_path)],
        shard_lengths=[5],
        static_path=str(static_path),
        mean=torch.zeros((1, 3, 1, 1), dtype=torch.float32),
        std=torch.ones((1, 3, 1, 1), dtype=torch.float32),
        batch_size=2,
    )

    x_batch, y_batch = next(iter(dataset))
    assert tuple(x_batch.shape) == (2, 8, 3)
    assert tuple(y_batch.shape) == (2, 8, 3)


def test_process_mesh_gnn_writes_static_and_shards(tmp_path: Path, monkeypatch):
    time = np.arange(6, dtype=np.int64)
    pfull = np.asarray([1000.0, 500.0], dtype=np.float32)
    lat = np.asarray([-45.0, 0.0, 45.0], dtype=np.float32)
    lon = np.asarray([0.0, 120.0, 240.0], dtype=np.float32)
    shape = (time.shape[0], pfull.shape[0], lat.shape[0], lon.shape[0])
    temp = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    ucomp = temp + 1.0
    vcomp = temp + 2.0

    fake_ds = FakeDataset(
        data_vars={
            "temp": FakeDataArray(temp, ("time", "pfull", "lat", "lon")),
            "ucomp": FakeDataArray(ucomp, ("time", "pfull", "lat", "lon")),
            "vcomp": FakeDataArray(vcomp, ("time", "pfull", "lat", "lon")),
        },
        coords={"time": time, "pfull": pfull, "lat": lat, "lon": lon},
    )

    monkeypatch.setattr(gc, "load_isca_result_data", lambda exp_folder_name, file_name: fake_ds)

    processed_dir = tmp_path / "mesh_gnn_dataset"
    manifest, split_shapes = gc.process_mesh_gnn(
        raw_cfg={"exp_folder_name": "exp", "file_name": "file.nc"},
        proc_cfg={
            "name": "mesh_gnn",
            "dataset_type": "SimpleDatasetMeshGNN",
            "params": {
                "vars": ["temp", "ucomp", "vcomp"],
                "level_dim": "pfull",
                "lat_dim": "lat",
                "lon_dim": "lon",
                "time_dim": "time",
                "time_chunk": 3,
                "time_start": 0,
                "max_timesteps": None,
                "shard_pairs": 2,
                "mesh_splits": 1,
                "radius_query_factor": 0.6,
                "candidate_face_k": 8,
                "edge_feature_mode": "receiver_local",
            },
        },
        out_cfg={"dataset_name": "mesh_gnn_dataset"},
        split_cfg=[0.5, 0.25, 0.25],
        split_ranges_fn=_split_ranges,
        stats_path=None,
        processed_dir=processed_dir,
    )

    assert manifest["processor"] == "mesh_gnn"
    assert split_shapes["train"][1:] == [9, 6]
    assert (processed_dir / "mesh_gnn_static.pt").exists()
    assert (processed_dir / "stats.pt").exists()
    assert (processed_dir / "manifest.yaml").exists()
    assert len(manifest["split_shards"]["train"]) >= 1
    static = torch.load(processed_dir / "mesh_gnn_static.pt", map_location="cpu", weights_only=False)
    assert "grid2mesh" in static
    assert "mesh_graph" in static
    assert "mesh2grid" in static
    assert manifest["mesh_gnn"]["edge_feature_mode"] == "receiver_local"
    assert static["feature_schema"]["edge_feature_mode"] == "receiver_local"
    assert int(static["mesh_graph"]["edge_features"].shape[-1]) == 4
