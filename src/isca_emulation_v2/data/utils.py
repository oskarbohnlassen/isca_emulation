from torch.utils.data import Dataset
import os.path as osp
import pickle
import torch
import json
import os

from typing import Any
import yaml
from hydra import initialize, compose
from omegaconf import DictConfig
import typer
import numpy as np
import numpy as np
import torch
import xarray as xr
from pathlib import Path
from datetime import datetime, timezone

RAW_DATA_ROOT = Path("data/raw")
PROCESSED_DATA_ROOT = Path("data/processed")

def _to_yaml_safe(value: Any) -> Any:
    """Recursively convert objects to YAML-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, xr.DataArray):
        return {
            "type": "xarray.DataArray",
            "name": str(value.name),
            "dims": [str(dim) for dim in value.dims],
            "shape": [int(v) for v in value.shape],
        }

    if isinstance(value, xr.Dataset):
        return {
            "type": "xarray.Dataset",
            "dims": {str(k): int(v) for k, v in value.dims.items()},
            "data_vars": [str(k) for k in value.data_vars.keys()],
        }

    if isinstance(value, dict):
        return {str(k): _to_yaml_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_yaml_safe(v) for v in value]

    return str(value)


def dump_yaml_mapping(payload: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _to_yaml_safe(payload)
    with p.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(safe_payload, handle, sort_keys=False)
    return p

def _load_stats_from_path(path_like: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")
    stats = torch.load(path, map_location="cpu", weights_only=True)
    if "mean" not in stats or "std" not in stats:
        raise ValueError(f"Stats file '{path}' must contain mean and std")
    return {"mean": stats["mean"], "std": stats["std"]}

def save_processed_artifact(
    *,
    processed_dir: str | Path,
    split_tensors: dict[str, torch.Tensor],
    stats: dict[str, torch.Tensor] | None,
    manifest: dict[str, Any],
) -> Path:
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=False)

    expected = [out / f"{name}.pt" for name in split_tensors.keys()]
    expected.append(out / "manifest.yaml")
    if stats is not None:
        expected.append(out / "stats.pt")

    if any(p.exists() for p in expected):
        raise FileExistsError(f"Processed output already exists in '{out}'.")

    for split_name, tensor in split_tensors.items():
        torch.save(tensor, out / f"{split_name}.pt")

    if stats is not None:
        torch.save(stats, out / "stats.pt")

    manifest_payload = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **manifest,
    }
    dump_yaml_mapping(manifest_payload, out / "manifest.yaml")
    return out

def make_splits(
    Xt: torch.Tensor,
    split_cfg: list[float] | bool,
) -> dict[str, torch.Tensor]:
    if split_cfg is False:
        return {"all": Xt}
    if not isinstance(split_cfg, list):
        raise ValueError("split must be either false or [train, val, test]")
    return split_by_fractions(Xt, split_cfg)


def split_by_fractions(
    Xt: torch.Tensor,
    fractions: list[float],
) -> dict[str, torch.Tensor]:
    """Split [T, ...] into train/val/test using three fractions."""
    if Xt.shape[0] < 1:
        raise ValueError("Input tensor has zero timesteps")
    if len(fractions) != 3:
        raise ValueError("split list must contain exactly [train, val, test] fractions")

    vals = [float(v) for v in fractions]
    if any(v <= 0.0 for v in vals):
        raise ValueError("All split fractions must be > 0")
    if abs(sum(vals) - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to 1.0")

    T = Xt.shape[0]
    n_train = int(T * vals[0])
    n_val = int(T * vals[1])
    n_test = T - n_train - n_val

    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("train/val/test splits must each contain at least one timestep")

    return {
        "train": Xt[:n_train],
        "val": Xt[n_train : n_train + n_val],
        "test": Xt[n_train + n_val :],
    }

def compute_channel_stats(Xt_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = Xt_train.mean(dim=(0, 2, 3), keepdim=True)
    std = Xt_train.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    return mean, std


def _minmax_normalize(values: torch.Tensor) -> torch.Tensor:
    v_min = values.min()
    v_max = values.max()
    return (values - v_min) / (v_max - v_min).clamp_min(1e-6)


def build_cnn2d_coord_channels(
    lonlat_values: dict[str, Any] | None,
    add_coords: str | None,
) -> torch.Tensor | None:
    if lonlat_values is None or add_coords in {None, "None", "none", False}:
        return None

    # Convert geophysical coordinates from degrees to radians before trig encoding.
    lat_rad = torch.deg2rad(torch.as_tensor(lonlat_values["lat"], dtype=torch.float32))
    lon_rad = torch.deg2rad(torch.as_tensor(lonlat_values["lon"], dtype=torch.float32))
    lat_grid, lon_grid = torch.meshgrid(lat_rad, lon_rad, indexing="ij")

    if add_coords == "trig2":
        return torch.stack([torch.sin(lat_grid), torch.cos(lat_grid)], dim=0)
    if add_coords == "trig4":
        return torch.stack(
            [torch.sin(lat_grid), torch.cos(lat_grid), torch.sin(lon_grid), torch.cos(lon_grid)],
            dim=0,
        )
    raise ValueError("add_coords must be one of [None, 'trig2', 'trig4'].")


def build_cnn3d_coord_channels(
    level_lonlat_values: dict[str, Any] | None,
    add_coords: str | None,
) -> torch.Tensor | None:
    if level_lonlat_values is None or add_coords in {None, "None", "none", False}:
        return None

    level = torch.as_tensor(level_lonlat_values["level"], dtype=torch.float32).reshape(-1)
    # Convert angular coordinates to radians for trig encoding.
    lat_rad = torch.deg2rad(torch.as_tensor(level_lonlat_values["lat"], dtype=torch.float32).reshape(-1))
    lon_rad = torch.deg2rad(torch.as_tensor(level_lonlat_values["lon"], dtype=torch.float32).reshape(-1))
    lat_grid, lon_grid = torch.meshgrid(lat_rad, lon_rad, indexing="ij")
    lat_3d = lat_grid.unsqueeze(0).expand(level.shape[0], -1, -1)
    lon_3d = lon_grid.unsqueeze(0).expand(level.shape[0], -1, -1)
    # Pressure level is not angular: use monotonic normalized channels (linear and log-pressure).
    if bool((level <= 0).any()):
        safe_level = level - level.min() + 1e-6
    else:
        safe_level = level
    level_norm = _minmax_normalize(safe_level)
    level_log_norm = _minmax_normalize(torch.log(safe_level))
    level_norm_3d = level_norm.view(-1, 1, 1).expand(-1, lat_rad.shape[0], lon_rad.shape[0])
    level_log_norm_3d = level_log_norm.view(-1, 1, 1).expand(-1, lat_rad.shape[0], lon_rad.shape[0])

    if add_coords == "trig2":
        return torch.stack(
            [torch.sin(lat_3d), torch.cos(lat_3d), level_norm_3d, level_log_norm_3d],
            dim=0,
        )
    if add_coords == "trig4":
        return torch.stack(
            [
                torch.sin(lat_3d),
                torch.cos(lat_3d),
                torch.sin(lon_3d),
                torch.cos(lon_3d),
                level_norm_3d,
                level_log_norm_3d,
            ],
            dim=0,
        )
    raise ValueError("add_coords must be one of [None, 'trig2', 'trig4'].")


def load_isca_result_data(
    exp_folder_name: str,
    file_name: str,
):
    """Load all runXXXX files for one experiment from data/raw by default."""
    import xarray as xr

    exp_dir = RAW_DATA_ROOT / exp_folder_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment folder not found: {exp_dir}")

    run_dirs = sorted(
        d
        for d in exp_dir.iterdir()
        if d.is_dir() and d.name.startswith("run")
    )
    files = [str(run_dir / file_name) for run_dir in run_dirs]
    if not files:
        raise FileNotFoundError(
            f"No files matched under '{exp_dir}' for file name '{file_name}'"
        )

    return xr.open_mfdataset(files, decode_times=False, data_vars='all')
