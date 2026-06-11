from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import IterableDataset, get_worker_info

from isca_emulation_v2.data.utils import dump_yaml_mapping, load_isca_result_data


NEIGHBOR_OFFSETS_8: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _wrap_to_pi(delta_lon_rad: np.ndarray | float) -> np.ndarray | float:
    return (delta_lon_rad + np.pi) % (2.0 * np.pi) - np.pi


def _haversine_central_angle(
    lat_i_rad: np.ndarray | float,
    lon_i_rad: np.ndarray | float,
    lat_j_rad: np.ndarray | float,
    lon_j_rad: np.ndarray | float,
) -> np.ndarray | float:
    d_lat = lat_j_rad - lat_i_rad
    d_lon = _wrap_to_pi(lon_j_rad - lon_i_rad)
    sin_dlat = np.sin(0.5 * d_lat)
    sin_dlon = np.sin(0.5 * d_lon)
    a = sin_dlat * sin_dlat + np.cos(lat_i_rad) * np.cos(lat_j_rad) * sin_dlon * sin_dlon
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _latlon_to_xyz(lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    cos_lat = np.cos(lat_rad)
    return np.stack(
        [
            cos_lat * np.cos(lon_rad),
            cos_lat * np.sin(lon_rad),
            np.sin(lat_rad),
        ],
        axis=-1,
    )


def _safe_normalize(vectors: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe = vectors / np.clip(norms, 1e-12, None)
    if np.all(norms > 1e-12):
        return safe
    return np.where(norms > 1e-12, safe, fallback)


def _edge_features_legacy_local(
    sender_lat_rad: np.ndarray,
    sender_lon_rad: np.ndarray,
    receiver_lat_rad: np.ndarray,
    receiver_lon_rad: np.ndarray,
) -> np.ndarray:
    delta_lon = _wrap_to_pi(receiver_lon_rad - sender_lon_rad)
    delta_lat = receiver_lat_rad - sender_lat_rad
    dx = delta_lon * np.cos(sender_lat_rad)
    dy = delta_lat
    dist = _haversine_central_angle(sender_lat_rad, sender_lon_rad, receiver_lat_rad, receiver_lon_rad)
    return np.stack([dx, dy, dist], axis=-1).astype(np.float32)


def _edge_features_receiver_local(sender_xyz: np.ndarray, receiver_xyz: np.ndarray) -> np.ndarray:
    up = _safe_normalize(receiver_xyz, fallback=np.zeros_like(receiver_xyz))
    reference = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64), (up.shape[0], 1))
    east = np.cross(reference, up)

    degenerate = np.linalg.norm(east, axis=-1, keepdims=True) <= 1e-12
    if np.any(degenerate):
        alt_reference = np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64), (up.shape[0], 1))
        east = np.where(degenerate, np.cross(alt_reference, up), east)

    east = _safe_normalize(east, fallback=np.zeros_like(east))
    north = _safe_normalize(np.cross(up, east), fallback=np.zeros_like(up))

    relative = sender_xyz - receiver_xyz
    relative_local = np.stack(
        [
            np.sum(relative * up, axis=-1),
            np.sum(relative * east, axis=-1),
            np.sum(relative * north, axis=-1),
        ],
        axis=-1,
    )
    edge_distance = np.linalg.norm(relative_local, axis=-1, keepdims=True)
    normalization = float(np.clip(edge_distance.max(initial=0.0), 1e-12, None))
    return np.concatenate(
        [
            (edge_distance / normalization).astype(np.float32),
            (relative_local / normalization).astype(np.float32),
        ],
        axis=-1,
    )


def _build_edge_features(
    sender_lat_rad: np.ndarray,
    sender_lon_rad: np.ndarray,
    receiver_lat_rad: np.ndarray,
    receiver_lon_rad: np.ndarray,
    sender_xyz: np.ndarray,
    receiver_xyz: np.ndarray,
    *,
    edge_feature_mode: str,
) -> tuple[np.ndarray, list[str]]:
    edge_feature_mode = str(edge_feature_mode).lower()
    if edge_feature_mode == "legacy_dxdy":
        return _edge_features_legacy_local(
            sender_lat_rad,
            sender_lon_rad,
            receiver_lat_rad,
            receiver_lon_rad,
        ), ["dx", "dy", "distance"]
    if edge_feature_mode == "receiver_local":
        return _edge_features_receiver_local(sender_xyz, receiver_xyz), [
            "local_distance",
            "local_receiver_x",
            "local_receiver_y",
            "local_receiver_z",
        ]
    raise ValueError("edge_feature_mode must be one of ['legacy_dxdy', 'receiver_local'].")


def _wrap_latlon_spherical(lat_idx: int, lon_idx: int, nlat: int, nlon: int) -> tuple[int, int]:
    # Longitude is periodic.
    lon_wrapped = lon_idx % nlon
    lat_wrapped = lat_idx

    # Latitude crossing is handled on a sphere by moving to the opposite meridian
    # while staying on the corresponding polar ring. This keeps pole-crossing
    # neighbours on the same extreme latitude index instead of reflecting one row down.
    pole_lon_shift = nlon // 2
    while lat_wrapped < 0 or lat_wrapped >= nlat:
        if lat_wrapped < 0:
            lat_wrapped = -lat_wrapped - 1
            lon_wrapped = (lon_wrapped + pole_lon_shift) % nlon
        else:
            lat_wrapped = 2 * nlat - 1 - lat_wrapped
            lon_wrapped = (lon_wrapped + pole_lon_shift) % nlon

    return lat_wrapped, lon_wrapped


def build_static_graph(
    lat: np.ndarray,
    lon: np.ndarray,
    nlat: int,
    nlon: int,
    *,
    add_distances_edge: bool,
    edge_feature_mode: str = "legacy_dxdy",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    # Input coordinates are degrees in ISCA output; convert once to radians.
    lat_rad = np.deg2rad(np.asarray(lat, dtype=np.float64).reshape(-1))
    lon_rad = np.deg2rad(np.asarray(lon, dtype=np.float64).reshape(-1))

    node_lat = np.repeat(lat_rad, nlon)
    node_lon = np.tile(lon_rad, nlat)

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_sender_lat: list[float] = []
    edge_sender_lon: list[float] = []
    edge_receiver_lat: list[float] = []
    edge_receiver_lon: list[float] = []

    for lat_i in range(nlat):
        for lon_i in range(nlon):
            src = lat_i * nlon + lon_i
            lat_src = node_lat[src]
            lon_src = node_lon[src]

            # 8-neighbour stencil with spherical wrapping in both latitude and longitude.
            for d_lat, d_lon in NEIGHBOR_OFFSETS_8:
                lat_j, lon_j = _wrap_latlon_spherical(
                    lat_i + d_lat,
                    lon_i + d_lon,
                    nlat,
                    nlon,
                )
                dst = lat_j * nlon + lon_j

                edge_src.append(src)
                edge_dst.append(dst)

                if add_distances_edge:
                    lat_dst = node_lat[dst]
                    lon_dst = node_lon[dst]
                    edge_sender_lat.append(float(lat_src))
                    edge_sender_lon.append(float(lon_src))
                    edge_receiver_lat.append(float(lat_dst))
                    edge_receiver_lon.append(float(lon_dst))

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    if add_distances_edge:
        sender_lat_rad = np.asarray(edge_sender_lat, dtype=np.float64)
        sender_lon_rad = np.asarray(edge_sender_lon, dtype=np.float64)
        receiver_lat_rad = np.asarray(edge_receiver_lat, dtype=np.float64)
        receiver_lon_rad = np.asarray(edge_receiver_lon, dtype=np.float64)
        sender_xyz = _latlon_to_xyz(sender_lat_rad, sender_lon_rad)
        receiver_xyz = _latlon_to_xyz(receiver_lat_rad, receiver_lon_rad)
        edge_attr_np, edge_feature_names = _build_edge_features(
            sender_lat_rad,
            sender_lon_rad,
            receiver_lat_rad,
            receiver_lon_rad,
            sender_xyz,
            receiver_xyz,
            edge_feature_mode=edge_feature_mode,
        )
        edge_attr = torch.as_tensor(edge_attr_np, dtype=torch.float32)
    else:
        # Keep edge_attr present for serialization and batching, but empty by request.
        edge_attr = torch.empty((edge_index.shape[1], 0), dtype=torch.float32)
        edge_feature_names = []

    lat_grid = np.repeat(lat_rad[:, None], nlon, axis=1)
    lon_grid = np.repeat(lon_rad[None, :], nlat, axis=0)
    node_static_np = np.stack(
        [np.sin(lat_grid), np.cos(lat_grid), np.sin(lon_grid), np.cos(lon_grid)],
        axis=-1,
    ).reshape(nlat * nlon, 4)
    node_static = torch.as_tensor(node_static_np, dtype=torch.float32)

    return edge_index, edge_attr, node_static, edge_feature_names


def xarray_to_tensor(
    ds,
    vars: list[str],
    *,
    level_dim: str,
    lat_dim: str,
    lon_dim: str,
    time_dim: str,
    load: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Convert selected variables to node-major dynamic features [T, N, F_dyn] for graph models."""

    # Ordering follows the same convention as CNN pipelines: variable-major, level-major.
    blocks: list[np.ndarray] = []
    for var_name in vars:
        da = ds[var_name].transpose(time_dim, level_dim, lat_dim, lon_dim)
        if load:
            da = da.load()
        blocks.append(np.asarray(da.to_numpy(), dtype=np.float32))

    dyn = np.concatenate(blocks, axis=1)  # [T, F_dyn, lat, lon]
    t_size, f_dyn, nlat, nlon = dyn.shape
    x_dyn = np.transpose(dyn, (0, 2, 3, 1)).reshape(t_size, nlat * nlon, f_dyn)

    meta = {
        "selected_levels": np.asarray(ds[level_dim].values, dtype=np.float32).reshape(-1).tolist(),
        "dynamic_feature_order": "variable-major, then level-major",
        "num_dynamic_features": int(f_dyn),
    }
    return torch.from_numpy(np.ascontiguousarray(x_dyn)), meta


def _load_external_stats(path_like: str | Path) -> dict[str, torch.Tensor]:
    stats = torch.load(Path(path_like), map_location="cpu", weights_only=True)
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
    std = torch.as_tensor(stats["std"], dtype=torch.float32)
    return {"mean": mean, "std": std.clamp_min(1e-6)}


def inverse_scale_gnn2d(scaled_data: np.ndarray, cfg) -> np.ndarray:
    """Unscale the GNN2D output by applying the inverse of the standardization."""
    path_to_stats = os.path.join("data", "processed", cfg.data.output.dataset_name, "stats.pt")
    stats = torch.load(path_to_stats, map_location="cpu", weights_only=False)
    mean = stats["mean"].reshape(-1).cpu().detach().numpy()
    std = stats["std"].reshape(-1).cpu().detach().numpy()
    return scaled_data * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def _select_node_coords(node_static: torch.Tensor, add_coords_node: str | None) -> torch.Tensor | None:
    if add_coords_node in {None, "none", "None", False}:
        return None
    if add_coords_node == "trig2":
        return node_static[:, :2]
    if add_coords_node == "trig4":
        return node_static[:, :4]
    raise ValueError("add_coords_node must be one of [None, 'trig2', 'trig4'].")


class SimpleDatasetGNN2D(IterableDataset):
    """Stream contiguous timestep pairs from graph shards as pre-batched tensors."""

    def __init__(
        self,
        shard_paths: list[str],
        shard_lengths: list[int],
        mean: torch.Tensor,
        std: torch.Tensor,
        batch_size: int,
        add_coords_node: str | None = "trig4",
        shuffle_shards: bool = False,
        shuffle_in_shard: bool = False,
        cache_in_memory: bool = False,
    ):
        self.shard_paths = list(shard_paths)
        self.shard_lengths = [int(v) for v in shard_lengths]
        self.pairs_per_shard = [v - 1 for v in self.shard_lengths]
        self.batch_size = int(batch_size)
        self.total_pairs = int(sum(self.pairs_per_shard))
        self.total_batches = int(sum((n + self.batch_size - 1) // self.batch_size for n in self.pairs_per_shard))
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_in_shard = bool(shuffle_in_shard)
        self.cache_in_memory = bool(cache_in_memory)
        self._cached_shards = None

        self._mean_vec = mean.reshape(-1).to(torch.float32)
        self._std_vec = std.reshape(-1).to(torch.float32).clamp_min(1e-6)
        self.add_coords_node = add_coords_node

        first_shard = self._load_graph_shard(0, mmap=False)
        self.edge_index = first_shard.edge_index.clone().to(torch.long)
        self.edge_attr = first_shard.edge_attr.clone().to(torch.float32)
        self.node_static = first_shard.node_static.clone().to(torch.float32)
        self.node_coords = _select_node_coords(self.node_static, self.add_coords_node)

        if self.cache_in_memory:
            self._cached_shards = [self._prepare_shard(shard_idx) for shard_idx in range(len(self.shard_paths))]

    def _load_graph_shard(self, shard_idx: int, *, mmap: bool) -> Data:
        return torch.load(self.shard_paths[shard_idx], weights_only=False, mmap=mmap)

    def _append_coords(self, x_batch: torch.Tensor) -> torch.Tensor:
        if self.node_coords is None:
            return x_batch
        coords = self.node_coords.unsqueeze(0).expand(x_batch.shape[0], -1, -1)
        if coords.device != x_batch.device or coords.dtype != x_batch.dtype:
            coords = coords.to(device=x_batch.device, dtype=x_batch.dtype)
        return torch.cat([x_batch, coords], dim=-1)

    def _prepare_shard(self, shard_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard = self._load_graph_shard(shard_idx, mmap=False)
        x_dyn = shard.x_dyn.to(torch.float32)
        mean = self._mean_vec.view(1, 1, -1)
        std = self._std_vec.view(1, 1, -1)
        x_state = (x_dyn[:-1] - mean) / std
        y_state = (x_dyn[1:] - mean) / std
        if self.node_coords is not None:
            x_state = self._append_coords(x_state)
        return x_state, y_state

    def _iter_shard_batches(self, shard_idx: int):
        shard = self._load_graph_shard(shard_idx, mmap=True)
        x_dyn = shard.x_dyn
        num_pairs = int(x_dyn.shape[0]) - 1
        if num_pairs <= 0:
            return

        mean = self._mean_vec.view(1, 1, -1)
        std = self._std_vec.view(1, 1, -1)

        if self.shuffle_in_shard:
            sample_order = torch.randperm(num_pairs)
            for i in range(0, num_pairs, self.batch_size):
                idx = sample_order[i : i + self.batch_size]
                x_batch = (x_dyn[idx].to(torch.float32) - mean) / std
                y_batch = (x_dyn[idx + 1].to(torch.float32) - mean) / std
                if self.node_coords is not None:
                    x_batch = self._append_coords(x_batch)
                yield x_batch, y_batch
        else:
            for i in range(0, num_pairs, self.batch_size):
                j = min(num_pairs, i + self.batch_size)
                x_batch = (x_dyn[i:j].to(torch.float32) - mean) / std
                y_batch = (x_dyn[i + 1 : j + 1].to(torch.float32) - mean) / std
                if self.node_coords is not None:
                    x_batch = self._append_coords(x_batch)
                yield x_batch, y_batch

    def __len__(self) -> int:
        return self.total_batches

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            shard_idxs = list(range(len(self.shard_paths)))
        else:
            shard_idxs = list(range(worker.id, len(self.shard_paths), worker.num_workers))

        if self.shuffle_shards:
            shard_order = torch.randperm(len(shard_idxs)).tolist()
            shard_idxs = [shard_idxs[i] for i in shard_order]

        for shard_idx in shard_idxs:
            if self.cache_in_memory:
                x_state, y_state = self._cached_shards[shard_idx]
                if self.shuffle_in_shard:
                    sample_order = torch.randperm(x_state.shape[0])
                    x_state = x_state[sample_order]
                    y_state = y_state[sample_order]
                for i in range(0, x_state.shape[0], self.batch_size):
                    j = min(x_state.shape[0], i + self.batch_size)
                    yield x_state[i:j], y_state[i:j]
            else:
                yield from self._iter_shard_batches(shard_idx)


def process_gnn2d(
    *,
    raw_cfg: dict[str, Any],
    proc_cfg: dict[str, Any],
    out_cfg: dict[str, Any],
    split_cfg: list[float] | bool,
    split_ranges_fn: Callable[[int, list[float] | bool], dict[str, tuple[int, int]]],
    stats_path: str | None,
    processed_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    proc_params = proc_cfg["params"]

    exp_folder_name = raw_cfg["exp_folder_name"]
    file_name = raw_cfg["file_name"]
    variables = list(proc_params["vars"])
    level_dim = proc_params["level_dim"]
    lat_dim = proc_params["lat_dim"]
    lon_dim = proc_params["lon_dim"]
    time_dim = proc_params["time_dim"]
    time_chunk = int(proc_params.get("time_chunk", 240))
    time_start = int(proc_params.get("time_start", 0))
    max_timesteps = proc_params.get("max_timesteps")
    neighbour_connectivity = proc_params['neighbour_connectivity']
    if neighbour_connectivity != 8:
        raise ValueError(
            f"Only neighbour_connectivity=8 is currently supported for gnn2d "
            f"(got {neighbour_connectivity})."
        )
    add_distances_edge = bool(proc_params.get("add_distances_edge", True))
    edge_feature_mode = str(proc_params.get("edge_feature_mode", "legacy_dxdy"))
    shard_pairs = int(proc_params.get("shard_pairs", time_chunk))

    processed_dir.mkdir(parents=True, exist_ok=False)

    ds = load_isca_result_data(exp_folder_name=exp_folder_name, file_name=file_name)
    total_time_all = int(ds.sizes[time_dim])
    end_idx = total_time_all if max_timesteps is None else min(total_time_all, time_start + int(max_timesteps))
    ds = ds.isel({time_dim: slice(time_start, end_idx)})
    total_time = int(ds.sizes[time_dim])

    lat_vals = np.asarray(ds[lat_dim].values, dtype=np.float32).reshape(-1)
    lon_vals = np.asarray(ds[lon_dim].values, dtype=np.float32).reshape(-1)
    nlat = int(lat_vals.shape[0])
    nlon = int(lon_vals.shape[0])
    num_nodes = nlat * nlon

    edge_index, edge_attr, node_static, edge_feature_names = build_static_graph(
        lat_vals,
        lon_vals,
        nlat,
        nlon,
        add_distances_edge=add_distances_edge,
        edge_feature_mode=edge_feature_mode,
    )

    split_ranges = split_ranges_fn(total_time, split_cfg)
    split_names = list(split_ranges.keys())
    stats_source = "all" if "all" in split_names else "train"
    split_buffers: dict[str, torch.Tensor | None] = {name: None for name in split_names}
    split_shards: dict[str, list[dict[str, Any]]] = {name: [] for name in split_names}
    split_next_idx: dict[str, int] = {name: 0 for name in split_names}
    split_total_timesteps: dict[str, int] = {name: 0 for name in split_names}

    stats_sum = None
    stats_sumsq = None
    stats_count = None
    feature_shape: list[int] | None = None

    def append_split(name: str, piece: torch.Tensor) -> None:
        nonlocal stats_sum, stats_sumsq, stats_count
        split_total_timesteps[name] += int(piece.shape[0])

        if stats_path is None and name == stats_source:
            piece64 = piece.to(torch.float64)
            reduce_dims = (0, 1)
            finite = torch.isfinite(piece64)
            finite_piece = torch.where(finite, piece64, torch.zeros_like(piece64))
            chunk_sum = finite_piece.sum(dim=reduce_dims, keepdim=True)
            chunk_sumsq = (finite_piece * finite_piece).sum(dim=reduce_dims, keepdim=True)
            chunk_count = finite.to(torch.float64).sum(dim=reduce_dims, keepdim=True)
            if stats_sum is None:
                stats_sum = torch.zeros_like(chunk_sum)
                stats_sumsq = torch.zeros_like(chunk_sumsq)
                stats_count = torch.zeros_like(chunk_count)
            stats_sum += chunk_sum
            stats_sumsq += chunk_sumsq
            stats_count += chunk_count

        if split_buffers[name] is None:
            split_buffers[name] = piece
        else:
            split_buffers[name] = torch.cat([split_buffers[name], piece], dim=0)

        # Keep same sharding semantics as CNN: write [shard_pairs + 1] timesteps and overlap by one.
        while split_buffers[name].shape[0] >= shard_pairs + 1:
            shard_tensor = split_buffers[name][: shard_pairs + 1].clone()
            shard_name = f"{name}_{split_next_idx[name]:03d}.pt"
            shard_data = Data(
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_static=node_static,
                x_dyn=shard_tensor,
            )
            torch.save(shard_data, processed_dir / shard_name)
            split_shards[name].append({"path": shard_name, "shape": list(shard_tensor.shape)})
            split_next_idx[name] += 1
            split_buffers[name] = split_buffers[name][shard_pairs:]

    feature_meta: dict[str, Any] | None = None
    for start in range(0, total_time, time_chunk):
        end = min(total_time, start + time_chunk)
        ds_chunk = ds.isel({time_dim: slice(start, end)})
        x_chunk, meta = xarray_to_tensor(
            ds_chunk,
            vars=variables,
            level_dim=level_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            time_dim=time_dim,
            load=True,
        )
        if feature_shape is None:
            feature_shape = [int(v) for v in x_chunk.shape[1:]]
            feature_meta = meta

        for name, (split_start, split_end) in split_ranges.items():
            local_start = max(start, split_start)
            local_end = min(end, split_end)
            if local_end > local_start:
                chunk_start = local_start - start
                chunk_end = local_end - start
                append_split(name, x_chunk[chunk_start:chunk_end])

    for name in split_names:
        if split_buffers[name] is not None and split_buffers[name].shape[0] >= 2:
            shard_tensor = split_buffers[name].clone()
            shard_name = f"{name}_{split_next_idx[name]:03d}.pt"
            shard_data = Data(
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_static=node_static,
                x_dyn=shard_tensor,
            )
            torch.save(shard_data, processed_dir / shard_name)
            split_shards[name].append({"path": shard_name, "shape": list(shard_tensor.shape)})

    if stats_path is not None:
        stats = _load_external_stats(stats_path)
        stats_split = "external"
    else:
        valid_count = stats_count.clamp_min(1.0)
        mean64 = stats_sum / valid_count
        var64 = stats_sumsq / valid_count - mean64 * mean64
        var64 = torch.where(stats_count > 0, var64, torch.ones_like(var64))
        mean = mean64.to(torch.float32)
        std = torch.sqrt(var64.clamp_min(1e-12)).to(torch.float32).clamp_min(1e-6)
        # Store in channel-style shape to align with existing CNN stats artifacts.
        f_dyn = int(feature_shape[-1]) if feature_shape is not None else int(mean.shape[-1])
        stats = {"mean": mean.reshape(1, f_dyn, 1, 1), "std": std.reshape(1, f_dyn, 1, 1)}
        stats_split = stats_source

    torch.save(stats, processed_dir / "stats.pt")
    split_shapes = {
        name: [split_total_timesteps[name], *(feature_shape or [])]
        for name in split_names
    }

    manifest = {
        "dataset_name": out_cfg["dataset_name"],
        "data": {
            "raw": raw_cfg,
            "processing": proc_cfg,
            "split": split_cfg,
            "output": out_cfg,
            "stats_path": str(stats_path) if stats_path is not None else None,
        },
        "processor": "gnn2d",
        "split_config": split_cfg,
        "split_names": split_names,
        "split_shapes": split_shapes,
        "split_shards": split_shards,
        "stats_split": stats_split,
        "stats_path": str(stats_path) if stats_path is not None else None,
        "has_stats": True,
        "grid": {
            "lat": lat_vals.tolist(),
            "lon": lon_vals.tolist(),
            "level": np.asarray(ds[level_dim].values, dtype=np.float32).reshape(-1).tolist(),
        },
        "graph": {
            "connectivity": neighbour_connectivity,
            "bidirected": True,
            "num_nodes": num_nodes,
            "num_edges": int(edge_index.shape[1]),
            "edge_attr_dim": int(edge_attr.shape[1]),
            "edge_feature_mode": edge_feature_mode if add_distances_edge else None,
            "edge_feature_names": list(edge_feature_names),
            "node_static_dim": int(node_static.shape[1]),
        },
        "dynamic_features": {
            "vars": variables,
            "level_dim": level_dim,
            "selected_levels": (feature_meta or {}).get("selected_levels", []),
            "dynamic_feature_order": (feature_meta or {}).get("dynamic_feature_order", "variable-major, then level-major"),
            "num_dynamic_features": int(feature_shape[-1]) if feature_shape is not None else 0,
            "add_coords_node": proc_params.get("add_coords_node", "trig4"),
            "add_distances_edge": add_distances_edge,
            "edge_feature_mode": edge_feature_mode if add_distances_edge else None,
        },
        "artifacts": {
            "split_shards": split_shards,
            "stats": "stats.pt",
        },
    }
    dump_yaml_mapping(
        {
            "format_version": 1,
            **manifest,
        },
        processed_dir / "manifest.yaml",
    )

    return manifest, split_shapes


def process_gnn(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Backwards-compatible alias for older config names."""
    return process_gnn2d(**kwargs)
