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

from isca_emulation_v2.data import gnn2d as g2d


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


def test_build_static_graph_legacy_edge_features():
    lat = np.linspace(-60.0, 60.0, 4, dtype=np.float32)
    lon = np.linspace(0.0, 300.0, 6, dtype=np.float32)

    edge_index, edge_attr, node_static, edge_feature_names = g2d.build_static_graph(
        lat,
        lon,
        nlat=4,
        nlon=6,
        add_distances_edge=True,
    )

    assert tuple(edge_index.shape) == (2, 24 * 8)
    assert tuple(edge_attr.shape) == (24 * 8, 3)
    assert tuple(node_static.shape) == (24, 4)
    assert edge_feature_names == ["dx", "dy", "distance"]


def test_build_static_graph_receiver_local_edge_features():
    lat = np.linspace(-60.0, 60.0, 4, dtype=np.float32)
    lon = np.linspace(0.0, 300.0, 6, dtype=np.float32)

    edge_index, edge_attr, _, edge_feature_names = g2d.build_static_graph(
        lat,
        lon,
        nlat=4,
        nlon=6,
        add_distances_edge=True,
        edge_feature_mode="receiver_local",
    )

    assert tuple(edge_index.shape) == (2, 24 * 8)
    assert tuple(edge_attr.shape) == (24 * 8, 4)
    assert edge_feature_names == [
        "local_distance",
        "local_receiver_x",
        "local_receiver_y",
        "local_receiver_z",
    ]
    assert torch.all(edge_attr[:, 0] >= 0.0)
    assert torch.all(edge_attr[:, 0] <= 1.0 + 1e-6)


def test_process_gnn2d_records_receiver_local_edge_mode(tmp_path: Path, monkeypatch):
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

    monkeypatch.setattr(g2d, "load_isca_result_data", lambda exp_folder_name, file_name: fake_ds)

    processed_dir = tmp_path / "gnn2d_dataset"
    manifest, split_shapes = g2d.process_gnn2d(
        raw_cfg={"exp_folder_name": "exp", "file_name": "file.nc"},
        proc_cfg={
            "name": "gnn2d",
            "dataset_type": "SimpleDatasetGNN2D",
            "params": {
                "vars": ["temp", "ucomp", "vcomp"],
                "level_dim": "pfull",
                "lat_dim": "lat",
                "lon_dim": "lon",
                "time_dim": "time",
                "time_chunk": 3,
                "time_start": 0,
                "max_timesteps": None,
                "neighbour_connectivity": 8,
                "add_coords_node": "trig4",
                "add_distances_edge": True,
                "edge_feature_mode": "receiver_local",
                "shard_pairs": 2,
            },
        },
        out_cfg={"dataset_name": "gnn2d_dataset"},
        split_cfg=[0.5, 0.25, 0.25],
        split_ranges_fn=_split_ranges,
        stats_path=None,
        processed_dir=processed_dir,
    )

    assert manifest["processor"] == "gnn2d"
    assert split_shapes["train"][1:] == [9, 6]
    assert manifest["graph"]["edge_feature_mode"] == "receiver_local"
    assert manifest["graph"]["edge_feature_names"] == [
        "local_distance",
        "local_receiver_x",
        "local_receiver_y",
        "local_receiver_z",
    ]

    shard = torch.load(processed_dir / manifest["split_shards"]["train"][0]["path"], weights_only=False)
    assert tuple(shard.edge_attr.shape)[1] == 4
