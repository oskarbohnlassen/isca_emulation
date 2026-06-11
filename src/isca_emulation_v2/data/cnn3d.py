import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig
from torch.utils.data import IterableDataset, get_worker_info

from isca_emulation_v2.data.utils import build_cnn3d_coord_channels, dump_yaml_mapping, load_isca_result_data


def process_cnn3d(
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

    vars = proc_params["vars"]
    level_dim = proc_params["level_dim"]
    lat_dim = proc_params["lat_dim"]
    lon_dim = proc_params["lon_dim"]
    time_dim = proc_params["time_dim"]

    time_chunk = int(proc_params.get("time_chunk", 240))
    time_start = int(proc_params.get("time_start", 0))
    max_timesteps = proc_params.get("max_timesteps")
    shard_pairs = int(proc_params.get("shard_pairs", time_chunk))

    processed_dir.mkdir(parents=True, exist_ok=False)
    ds = load_isca_result_data(exp_folder_name=exp_folder_name, file_name=file_name)
    total_time_all = int(ds.sizes[time_dim])
    end_idx = total_time_all if max_timesteps is None else min(total_time_all, time_start + int(max_timesteps))
    ds = ds.isel({time_dim: slice(time_start, end_idx)})
    total_time = int(ds.sizes[time_dim])
    ranges = split_ranges_fn(total_time, split_cfg)
    split_names = list(ranges.keys())
    stats_source = "all" if "all" in split_names else "train"

    lat_vals = np.asarray(ds[lat_dim].values, dtype=np.float32).reshape(-1)
    lon_vals = np.asarray(ds[lon_dim].values, dtype=np.float32).reshape(-1)
    level_vals = np.asarray(ds[level_dim].values, dtype=np.float32).reshape(-1)
    processor_meta = {
        "grid": {
            "lat": lat_vals.tolist(),
            "lon": lon_vals.tolist(),
            "level": level_vals.tolist(),
        },
    }
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
            # Keep (channel, level) statistics separate to match cnn2d var-level scaling semantics.
            reduce_dims = (0, 3, 4)
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

        while split_buffers[name].shape[0] >= shard_pairs + 1:
            shard_tensor = split_buffers[name][: shard_pairs + 1].clone()
            shard_name = f"{name}_{split_next_idx[name]:03d}.pt"
            torch.save(shard_tensor, processed_dir / shard_name)
            split_shards[name].append({"path": shard_name, "shape": list(shard_tensor.shape)})
            split_next_idx[name] += 1
            split_buffers[name] = split_buffers[name][shard_pairs:]

    for start in range(0, total_time, time_chunk):
        end = min(total_time, start + time_chunk)
        ds_chunk = ds.isel({time_dim: slice(start, end)})
        x_chunk, _ = xarray_to_tensor(
            ds_chunk,
            vars=vars,
            level_dim=level_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            time_dim=time_dim,
            load=True,
        )
        if feature_shape is None:
            feature_shape = [int(v) for v in x_chunk.shape[1:]]

        for name, (split_start, split_end) in ranges.items():
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
            torch.save(shard_tensor, processed_dir / shard_name)
            split_shards[name].append({"path": shard_name, "shape": list(shard_tensor.shape)})

    if stats_path is not None:
        stats = _load_stats_from_path(stats_path)
        stats_split = "external"
    else:
        valid_count = stats_count.clamp_min(1.0)
        mean64 = stats_sum / valid_count
        var64 = stats_sumsq / valid_count - mean64 * mean64
        var64 = torch.where(stats_count > 0, var64, torch.ones_like(var64))
        mean = mean64.to(torch.float32)
        std = torch.sqrt(var64.clamp_min(1e-12)).to(torch.float32).clamp_min(1e-6)
        stats = {"mean": mean, "std": std}
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
        "processor": "cnn3d",
        "split_config": split_cfg,
        "split_names": split_names,
        "split_shapes": split_shapes,
        "split_shards": split_shards,
        "stats_split": stats_split,
        "stats_path": str(stats_path) if stats_path is not None else None,
        "has_stats": True,
        "grid": processor_meta["grid"],
    }
    dump_yaml_mapping(
        {
            "format_version": 1,
            **manifest,
        },
        processed_dir / "manifest.yaml",
    )

    return manifest, split_shapes


def _load_stats_from_path(path_like: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path_like)
    stats = torch.load(path, map_location="cpu", weights_only=True)
    return {"mean": stats["mean"], "std": stats["std"]}


def xarray_to_tensor(
    ds: "xr.Dataset",
    vars: list[str],
    *,
    level_dim: str,
    lat_dim: str,
    lon_dim: str,
    time_dim: str,
    load: bool = True,
) -> tuple[torch.Tensor, "xr.DataArray"]:
    """Convert selected variables to [T, C, L, H, W] where C is the number of variables."""
    blocks = []
    for var_name in vars:
        da = ds[var_name].transpose(time_dim, level_dim, lat_dim, lon_dim)
        if load:
            da = da.load()
        blocks.append(da)

    stacked = xr.concat(blocks, dim=xr.IndexVariable("channel", list(vars))).transpose(
        time_dim,
        "channel",
        level_dim,
        lat_dim,
        lon_dim,
    )

    arr = stacked.to_numpy().astype(np.float32)
    return torch.from_numpy(arr), stacked


def inverse_scale_cnn3d(scaled_data: np.ndarray, cfg: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    """Unscale the CNN3D output by applying the inverse of the standardization."""
    path_to_stats = os.path.join("data", "processed", cfg.data.output.dataset_name, "stats.pt")
    stats = torch.load(path_to_stats)
    mean = stats["mean"][0].cpu().detach().numpy()
    std = stats["std"][0].cpu().detach().numpy()

    unscaled_data = scaled_data * std + mean
    return unscaled_data


class SimpleDatasetCNN3D(IterableDataset):
    """Stream contiguous timestep pairs from time shards as pre-batched tensors."""

    def __init__(
        self,
        shard_paths: list[str],
        shard_lengths: list[int],
        mean: torch.Tensor,
        std: torch.Tensor,
        batch_size: int,
        lonlat_values: dict[str, Any] | None = None,
        add_coords: str | None = None,
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
        self.mean = mean
        self.std = std
        self.coords = build_cnn3d_coord_channels(level_lonlat_values=lonlat_values, add_coords=add_coords)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_in_shard = bool(shuffle_in_shard)
        self.cache_in_memory = bool(cache_in_memory)
        self._mean0 = self.mean[0]
        self._std0 = self.std[0]
        self._cached_shards = None

        if self.cache_in_memory:
            self._cached_shards = [self._prepare_shard(shard_idx) for shard_idx in range(len(self.shard_paths))]

    def _load_shard(self, shard_idx: int, *, mmap: bool) -> torch.Tensor:
        return torch.load(self.shard_paths[shard_idx], weights_only=False, mmap=mmap)

    def _prepare_shard(self, shard_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard = self._load_shard(shard_idx, mmap=False)
        x_state = (shard[:-1] - self._mean0) / self._std0
        y_state = (shard[1:] - self._mean0) / self._std0
        return x_state, y_state

    def _append_coords(self, x_batch: torch.Tensor) -> torch.Tensor:
        if self.coords is None:
            return x_batch
        bsz, x_ch, depth, height, width = x_batch.shape
        coord_ch = int(self.coords.shape[0])
        out = torch.empty(
            (bsz, x_ch + coord_ch, depth, height, width),
            dtype=x_batch.dtype,
            device=x_batch.device,
        )
        out[:, :x_ch] = x_batch
        coords_expanded = self.coords.unsqueeze(0).expand(bsz, -1, -1, -1, -1)
        if coords_expanded.device != x_batch.device or coords_expanded.dtype != x_batch.dtype:
            coords_expanded = coords_expanded.to(device=x_batch.device, dtype=x_batch.dtype)
        out[:, x_ch:] = coords_expanded
        return out

    def _iter_shard_batches(self, shard_idx: int):
        shard = self._load_shard(shard_idx, mmap=True)
        num_pairs = int(shard.shape[0]) - 1
        if num_pairs <= 0:
            return

        if self.shuffle_in_shard:
            sample_order = torch.randperm(num_pairs)
            for i in range(0, num_pairs, self.batch_size):
                idx = sample_order[i : i + self.batch_size]
                x_batch = (shard[idx] - self._mean0) / self._std0
                y_batch = (shard[idx + 1] - self._mean0) / self._std0
                if self.coords is not None:
                    coords = self.coords.unsqueeze(0).expand(x_batch.shape[0], -1, -1, -1, -1)
                    x_batch = torch.cat([x_batch, coords], dim=1)
                yield x_batch, y_batch
        else:
            for i in range(0, num_pairs, self.batch_size):
                j = min(num_pairs, i + self.batch_size)
                x_batch = (shard[i:j] - self._mean0) / self._std0
                y_batch = (shard[i + 1 : j + 1] - self._mean0) / self._std0
                if self.coords is not None:
                    coords = self.coords.unsqueeze(0).expand(x_batch.shape[0], -1, -1, -1, -1)
                    x_batch = torch.cat([x_batch, coords], dim=1)
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
                    x_iter = x_state[sample_order]
                    y_iter = y_state[sample_order]
                else:
                    x_iter = x_state
                    y_iter = y_state

                for i in range(0, x_iter.shape[0], self.batch_size):
                    j = i + self.batch_size
                    x_batch = x_iter[i:j]
                    y_batch = y_iter[i:j]
                    if self.coords is not None:
                        x_batch = self._append_coords(x_batch)
                    yield x_batch, y_batch
            else:
                yield from self._iter_shard_batches(shard_idx)
