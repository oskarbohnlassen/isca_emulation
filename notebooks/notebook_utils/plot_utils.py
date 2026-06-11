from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
import xarray as xr



VARIABLE_TO_IDX = {"temp": 0, "ucomp": 1, "vcomp": 2}
VARIABLE_TO_METRIC = {
    "temp": "Temperature (K)",
    "ucomp": "Zonal Wind (m/s)",
    "vcomp": "Meridional Wind (m/s)",
}

DEFAULT_ARCHITECTURE_ALIASES = {
    "cnn2d": ["simplecnn2d", "cnn2d"],
    "cnn2d_unet": ["unetcnn2d", "cnn2dunet", "unet_cnn2d"],
    "cnn3d": ["simplecnn3d", "cnn3d"],
    "cnn3d_unet": ["unetcnn3d", "cnn3dunet", "unet_cnn3d"],
    "transformer_with_cnn2_tokens_global_attention": [
        "transformer2dglobalattentionforecaster",
        "transformer2dglobalattention",
        "transformer_with_cnn2_tokens_global_attention",
    ],
    "transformer_with_cnn2_tokens_swin_attention": [
        "transformer2dswinforecaster",
        "transformer2dswinattentionforecaster",
        "transformer_with_cnn2_tokens_swin_attention",
    ],
    "transformer_with_cnn3_tokens_global_attention": [
        "transformer3dglobalattentionforecaster",
        "transformer3dglobalattention",
        "transformer_with_cnn3_tokens_global_attention",
    ],
    "transformer_with_cnn3_tokens_swin_attention": [
        "transformer3dswinforecaster",
        "transformer3dswinattentionforecaster",
        "transformer_with_cnn3_tokens_swin_attention",
    ],
    "gnn_based_encoder_regulargrid_mpnn_decoder": ["simplegnn2d", "gnn", "mpnn"],
    "mesh_gnn_gridtomesh_mesh_and_meshtogrid": ["mesh_gnn2d", "mesh_gnn"],
}


def _validate_variable(variable: str) -> int:
    if variable not in VARIABLE_TO_IDX:
        raise ValueError(f"variable must be one of {list(VARIABLE_TO_IDX)}, got {variable}")
    return VARIABLE_TO_IDX[variable]


def _entries_from_results(results_obj: Any, name: str) -> list[dict[str, Any]]:
    """Strict parser for result collections.

    Accepted formats:
    - list/tuple of (label, result_dict)
    - dict[label] = result_dict
    where result_dict must contain y_true_unscaled and y_pred_unscaled.
    """
    if isinstance(results_obj, Mapping):
        iterable = list(results_obj.items())
    elif isinstance(results_obj, (list, tuple)):
        iterable = list(results_obj)
    else:
        raise TypeError(f"{name} must be a list/tuple or dict of (label, result_dict) entries")

    entries: list[dict[str, Any]] = []
    for item in iterable:
        if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], Mapping)):
            raise TypeError(
                f"{name} entries must be (label: str, result_dict: dict). Got: {type(item)}"
            )
        label, result = item
        if "y_true_unscaled" not in result or "y_pred_unscaled" not in result:
            raise ValueError(f"{name}[{label!r}] is missing y_true_unscaled/y_pred_unscaled")

        entries.append(
            {
                "label": label,
                "model_type": str(result.get("evaluation_info", {}).get("model_type", "")),
                "result": dict(result),
                "y_true": np.asarray(result["y_true_unscaled"]),
                "y_pred": np.asarray(result["y_pred_unscaled"]),
            }
        )

    if not entries:
        raise ValueError(f"{name} is empty")

    return entries


def _to_var_level(arr: np.ndarray, n_levels: int, n_lats: int) -> np.ndarray:
    """Return tensor in [N, V, L, H, W] with strict checks.

    5D expected shape: [N, V, n_levels, n_lats, W]
    4D expected shape: [N, C, n_lats, W] with C == 3 * n_levels
    3D expected shape: [N, num_nodes, F] with num_nodes == n_lats * W and F == 3 * n_levels
    """
    arr = np.asarray(arr)

    if arr.ndim == 5:
        if arr.shape[2] != n_levels:
            raise ValueError(
                f"Pressure-level mismatch: tensor has L={arr.shape[2]}, pressure_levels has length {n_levels}."
            )
        if arr.shape[3] != n_lats:
            raise ValueError(
                f"Latitude mismatch: tensor has H={arr.shape[3]}, lat_levels has length {n_lats}."
            )
        if arr.shape[1] < 3:
            raise ValueError(f"Expected at least 3 variables in axis=1, got {arr.shape[1]}.")
        return arr

    expected_v = len(VARIABLE_TO_IDX)
    if arr.ndim == 4:
        n, c, h, w = arr.shape
        if h != n_lats:
            raise ValueError(f"Latitude mismatch: tensor has H={h}, lat_levels has length {n_lats}.")
        expected_c = expected_v * n_levels
        if c != expected_c:
            raise ValueError(
                f"For 4D tensors [N,C,H,W], expected C == {expected_v}*len(pressure_levels) == {expected_c}, got C={c}."
            )
        return arr.reshape(n, expected_v, n_levels, h, w)

    if arr.ndim == 3:
        n, num_nodes, num_features = arr.shape
        if num_nodes % n_lats != 0:
            raise ValueError(
                f"For 3D tensors [N,num_nodes,F], expected num_nodes to be divisible by n_lats={n_lats}, got {num_nodes}."
            )
        expected_f = expected_v * n_levels
        if num_features != expected_f:
            raise ValueError(
                f"For 3D tensors [N,num_nodes,F], expected F == {expected_v}*len(pressure_levels) == {expected_f}, got F={num_features}."
            )
        n_lons = num_nodes // n_lats
        arr_grid = arr.reshape(n, n_lats, n_lons, num_features).transpose(0, 3, 1, 2)
        return arr_grid.reshape(n, expected_v, n_levels, n_lats, n_lons)

    raise ValueError(f"Expected 3D, 4D or 5D tensor, got shape {arr.shape}")


def _canonicalize_result_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pressure_levels: Sequence[float],
    lat_levels: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    n_levels = int(np.asarray(pressure_levels).shape[0])
    n_lats = int(np.asarray(lat_levels).shape[0])
    y_true_vl = _to_var_level(y_true, n_levels=n_levels, n_lats=n_lats)
    y_pred_vl = _to_var_level(y_pred, n_levels=n_levels, n_lats=n_lats)
    if y_true_vl.shape != y_pred_vl.shape:
        raise ValueError(
            "Canonicalized y_true and y_pred shapes do not match: "
            f"{tuple(y_true_vl.shape)} vs {tuple(y_pred_vl.shape)}"
        )
    return y_true_vl, y_pred_vl


def _mae_profile(y_true: np.ndarray, y_pred: np.ndarray, var_idx: int, n_levels: int) -> np.ndarray:
    y_true_vl = _to_var_level(y_true, n_levels=n_levels, n_lats=y_true.shape[-2] if y_true.ndim == 5 else y_true.shape[-2])
    y_pred_vl = _to_var_level(y_pred, n_levels=n_levels, n_lats=y_pred.shape[-2] if y_pred.ndim == 5 else y_pred.shape[-2])
    return np.abs(y_true_vl - y_pred_vl)[:, var_idx, :, :, :].mean(axis=(0, 2, 3))


def _persistence_profile_from_y_true(y_true: np.ndarray, var_idx: int, n_levels: int, n_lats: int) -> np.ndarray:
    y_true_vl = _to_var_level(y_true, n_levels=n_levels, n_lats=n_lats)
    if y_true_vl.shape[0] < 2:
        raise ValueError("Need at least 2 timesteps to compute persistence profile")
    return np.abs(y_true_vl[1:] - y_true_vl[:-1])[:, var_idx, :, :, :].mean(axis=(0, 2, 3))


def _mae_map(y_true: np.ndarray, y_pred: np.ndarray, var_idx: int, n_levels: int, n_lats: int) -> np.ndarray:
    y_true_vl = _to_var_level(y_true, n_levels=n_levels, n_lats=n_lats)
    y_pred_vl = _to_var_level(y_pred, n_levels=n_levels, n_lats=n_lats)
    return np.abs(y_pred_vl - y_true_vl)[:, var_idx, :, :, :].mean(axis=(0, 3))


def _persistence_map_from_y_true(y_true: np.ndarray, var_idx: int, n_levels: int, n_lats: int) -> np.ndarray:
    y_true_vl = _to_var_level(y_true, n_levels=n_levels, n_lats=n_lats)
    if y_true_vl.shape[0] < 2:
        raise ValueError("Need at least 2 timesteps to compute persistence map")
    return np.abs(y_true_vl[1:] - y_true_vl[:-1])[:, var_idx, :, :, :].mean(axis=(0, 3))


def _aggregate_map(
    arr: np.ndarray,
    pressure_levels: np.ndarray,
    lat_levels: np.ndarray,
    n_pressure_bins: int,
    n_lat_bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_levels = int(pressure_levels.shape[0])
    n_lats = int(lat_levels.shape[0])
    arr = np.asarray(arr)
    if arr.shape != (n_levels, n_lats):
        raise ValueError(f"Expected map shape {(n_levels, n_lats)}, got {arr.shape}")

    if n_pressure_bins == n_levels and n_lat_bins == n_lats:
        return arr, pressure_levels, lat_levels

    p_groups = np.array_split(np.arange(n_levels), n_pressure_bins)
    lat_groups = np.array_split(np.arange(n_lats), n_lat_bins)

    arr_agg = np.empty((n_pressure_bins, n_lat_bins), dtype=float)
    for i, p_idx in enumerate(p_groups):
        for j, l_idx in enumerate(lat_groups):
            arr_agg[i, j] = np.nanmean(arr[np.ix_(p_idx, l_idx)])

    p_agg = np.array([np.nanmean(pressure_levels[g]) for g in p_groups])
    lat_agg = np.array([np.nanmean(lat_levels[g]) for g in lat_groups])
    return arr_agg, p_agg, lat_agg


def load_grid_from_processed_dataset(
    processed_dataset_path_to_get_grid: Path | str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pressure, latitude and longitude arrays from manifest.yaml."""
    dataset_path = Path(processed_dataset_path_to_get_grid)
    manifest_path = dataset_path / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Could not find manifest.yaml at {manifest_path}")

    manifest = yaml.safe_load(manifest_path.read_text())
    if "grid" not in manifest:
        raise KeyError(f"manifest.yaml at {manifest_path} has no 'grid' section")

    grid = manifest["grid"]
    missing = [k for k in ("level", "lat", "lon") if k not in grid]
    if missing:
        raise KeyError(f"manifest.yaml missing grid keys: {missing}")

    pressure_levels = np.asarray(grid["level"])
    lat_levels = np.asarray(grid["lat"])
    lon_levels = np.asarray(grid["lon"])
    return pressure_levels, lat_levels, lon_levels


def _count_parameters_from_checkpoint_file(ckpt_path: Path) -> Optional[int]:
    """Count tensor elements from a checkpoint state dict."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Fallback for torch versions that do not support weights_only.
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return None

    if isinstance(ckpt, Mapping) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    if not isinstance(state_dict, Mapping):
        return None

    num_parameters = 0
    found_tensor = False
    for value in state_dict.values():
        if torch.is_tensor(value):
            num_parameters += int(value.numel())
            found_tensor = True

    if not found_tensor:
        return None
    return int(num_parameters)


def _count_parameters_from_artifact_checkpoint(
    artifact_path: Path | str,
) -> Tuple[Optional[int], Optional[Path]]:
    """Load a checkpoint from an artifact directory and count parameters."""
    artifact_path = Path(artifact_path)
    candidates = sorted(
        [p for p in artifact_path.glob("*.pt") if p.name != "evaluation_results.pt"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for ckpt_path in candidates:
        num = _count_parameters_from_checkpoint_file(ckpt_path)
        if num is not None:
            return int(num), ckpt_path
    return None, None


def load_evaluation_results(artifact_path: Path | str, *, verbose: bool = True) -> dict[str, Any]:
    artifact_path = Path(artifact_path)
    results = torch.load(artifact_path / "evaluation_results.pt", map_location="cpu", weights_only=False)

    y_true = np.asarray(results["y_true_unscaled"])
    y_pred = np.asarray(results["y_pred_unscaled"])
    pers_true = y_true[1:]
    pers_pred = y_true[:-1]

    mae_model = float(np.nanmean(np.abs(y_pred - y_true)))
    mae_persistence = float(np.nanmean(np.abs(pers_pred - pers_true)))
    num_parameters, checkpoint_path = _count_parameters_from_artifact_checkpoint(artifact_path)

    if verbose:
        params_str = f", params={num_parameters:,}" if num_parameters is not None else ", params=NA"
        print(
            f"[{artifact_path.name}] y_true shape={y_true.shape}, "
            f"MAE={mae_model:.6f}, persistence_MAE={mae_persistence:.6f}{params_str}"
        )

    return {
        "artifact_path": artifact_path,
        "result": results,
        "y_true": y_true,
        "y_pred": y_pred,
        "pers_true": pers_true,
        "pers_pred": pers_pred,
        "mae_model": mae_model,
        "mae_persistence": mae_persistence,
        "num_parameters": num_parameters,
        "num_parameters_source": str(checkpoint_path) if checkpoint_path is not None else None,
        "evaluation_info": results.get("evaluation_info", {}),
        "metrics": results.get("metrics", {}),
    }


def load_artifact_collections(
    no_ssw_artifacts: Mapping[str, Path | str],
    with_ssw_artifacts: Mapping[str, Path | str],
    *,
    processed_dataset_path_to_get_grid: Path | str,
    verbose: bool = True,
) -> dict[str, Any]:
    """Load artifact collections and return model payloads + persistence + grid arrays.

    Relative artifact paths are resolved under repo-root `results/`.
    Returned result tensors are canonicalized to [N, V, L, H, W].
    """
    repo_root = Path(__file__).resolve().parents[2]
    results_root = repo_root / "results"

    def _resolve_artifact_path(path_like: Path | str) -> Path:
        candidate = Path(path_like)
        if candidate.is_absolute():
            return candidate
        if candidate.parts and candidate.parts[0] == "results":
            return repo_root / candidate
        return results_root / candidate

    pressure_levels, lat_levels, lon_levels = load_grid_from_processed_dataset(
        processed_dataset_path_to_get_grid
    )

    def _load_collection(artifacts: Mapping[str, Path | str]):
        results_models = []
        model_payloads = {}
        persistence = {}

        for label, artifact_path in artifacts.items():
            item = load_evaluation_results(_resolve_artifact_path(artifact_path), verbose=verbose)
            y_true_vl, y_pred_vl = _canonicalize_result_arrays(
                item["y_true"],
                item["y_pred"],
                pressure_levels,
                lat_levels,
            )

            result_with_meta = dict(item["result"])
            result_with_meta["y_true_unscaled"] = y_true_vl
            result_with_meta["y_pred_unscaled"] = y_pred_vl
            result_with_meta["num_parameters"] = item["num_parameters"]
            result_with_meta["num_parameters_source"] = item["num_parameters_source"]

            item = {
                **item,
                "result": result_with_meta,
                "y_true": y_true_vl,
                "y_pred": y_pred_vl,
                "pers_true": y_true_vl[1:],
                "pers_pred": y_true_vl[:-1],
            }

            results_models.append((str(label), result_with_meta))
            model_payloads[str(label)] = item
            persistence[str(label)] = {
                "pers_true": item["pers_true"],
                "pers_pred": item["pers_pred"],
                "mae": item["mae_persistence"],
            }

        return results_models, model_payloads, persistence

    results_no_ssw_models, model_payloads_no_ssw, persistence_no_ssw = _load_collection(no_ssw_artifacts)
    results_with_ssw_models, model_payloads_with_ssw, persistence_with_ssw = _load_collection(with_ssw_artifacts)

    return {
        "results_no_ssw_models": results_no_ssw_models,
        "results_with_ssw_models": results_with_ssw_models,
        "model_payloads_no_ssw": model_payloads_no_ssw,
        "model_payloads_with_ssw": model_payloads_with_ssw,
        "persistence_no_ssw": persistence_no_ssw,
        "persistence_with_ssw": persistence_with_ssw,
        "pressure_levels": pressure_levels,
        "lat_levels": lat_levels,
        "lon_levels": lon_levels,
        "processed_dataset_path_to_get_grid": str(Path(processed_dataset_path_to_get_grid)),
    }


def _metadata_parameter_estimator(label: str, result_dict: Mapping[str, Any]) -> float:
    """Read parameter count from result metadata only."""
    del label
    eval_info = result_dict.get("evaluation_info", {}) if isinstance(result_dict, Mapping) else {}
    metrics = result_dict.get("metrics", {}) if isinstance(result_dict, Mapping) else {}
    candidates = [
        result_dict.get("num_parameters") if isinstance(result_dict, Mapping) else None,
        result_dict.get("n_params") if isinstance(result_dict, Mapping) else None,
        result_dict.get("parameter_count") if isinstance(result_dict, Mapping) else None,
        eval_info.get("num_parameters") if isinstance(eval_info, Mapping) else None,
        eval_info.get("n_params") if isinstance(eval_info, Mapping) else None,
        eval_info.get("parameter_count") if isinstance(eval_info, Mapping) else None,
        metrics.get("num_parameters") if isinstance(metrics, Mapping) else None,
        metrics.get("n_params") if isinstance(metrics, Mapping) else None,
        metrics.get("parameter_count") if isinstance(metrics, Mapping) else None,
    ]
    for value in candidates:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float("nan")


def _build_mae_summary_dataframe(
    entries: Sequence[dict[str, Any]],
    pressure_levels: np.ndarray,
    strat_max_pressure_hpa: float,
    *,
    parameter_estimator: Optional[Callable[[str, Mapping[str, Any]], Optional[float]]] = None,
    extra_columns: Optional[Mapping[str, Callable[[Mapping[str, Any]], Any]]] = None,
    include_persistence_row: bool = True,
    persistence_label: str = "Persistence",
) -> pd.DataFrame:
    """Internal helper to build a MAE summary table for one result collection."""
    n_levels = int(pressure_levels.shape[0])
    strat_mask = np.asarray(pressure_levels) <= float(strat_max_pressure_hpa)
    if not np.any(strat_mask):
        raise ValueError(
            f"No pressure levels found at/above stratosphere threshold "
            f"(pressure <= {strat_max_pressure_hpa})."
        )

    rows = {}
    for e in entries:
        y_true = np.asarray(e["y_true"])
        y_pred = np.asarray(e["y_pred"])
        if y_true.ndim not in (3, 4, 5):
            raise ValueError(f"{e['label']!r}: Expected y_true to be 3D, 4D or 5D, got {y_true.shape}")
        if y_pred.ndim not in (3, 4, 5):
            raise ValueError(f"{e['label']!r}: Expected y_pred to be 3D, 4D or 5D, got {y_pred.shape}")
        n_lats = int(y_true.shape[1] if y_true.ndim == 3 else y_true.shape[-2])

        y_true_vl = _to_var_level(y_true, n_levels=n_levels, n_lats=n_lats)
        y_pred_vl = _to_var_level(y_pred, n_levels=n_levels, n_lats=n_lats)
        abs_err = np.abs(y_pred_vl - y_true_vl)
        abs_err_strat = abs_err[:, :, strat_mask, :, :]

        estimator = parameter_estimator or _metadata_parameter_estimator
        num_params_value = estimator(e["label"], e["result"])
        try:
            num_params = float(num_params_value) if num_params_value is not None else float("nan")
        except (TypeError, ValueError):
            num_params = float("nan")

        row = {
            "MAE": float(np.nanmean(abs_err)),
            "MAE_strat": float(np.nanmean(abs_err_strat)),
            "MAE_temp": float(np.nanmean(abs_err[:, VARIABLE_TO_IDX["temp"], :, :, :])),
            "MAE_ucomp": float(np.nanmean(abs_err[:, VARIABLE_TO_IDX["ucomp"], :, :, :])),
            "MAE_vcomp": float(np.nanmean(abs_err[:, VARIABLE_TO_IDX["vcomp"], :, :, :])),
            "MAE_strat_temp": float(np.nanmean(abs_err_strat[:, VARIABLE_TO_IDX["temp"], :, :, :])),
            "MAE_strat_ucomp": float(np.nanmean(abs_err_strat[:, VARIABLE_TO_IDX["ucomp"], :, :, :])),
            "MAE_strat_vcomp": float(np.nanmean(abs_err_strat[:, VARIABLE_TO_IDX["vcomp"], :, :, :])),
            "num_parameters": num_params,
        }

        if extra_columns:
            context = {
                "entry": e,
                "y_true_vl": y_true_vl,
                "y_pred_vl": y_pred_vl,
                "abs_err": abs_err,
                "abs_err_strat": abs_err_strat,
                "pressure_levels": pressure_levels,
                "strat_mask": strat_mask,
                "strat_max_pressure_hpa": float(strat_max_pressure_hpa),
            }
            for col_name, col_fn in extra_columns.items():
                row[col_name] = col_fn(context)

        rows[e["label"]] = row

    if include_persistence_row:
        first = entries[0]
        y_true_first = np.asarray(first["y_true"])
        n_lats_first = int(y_true_first.shape[1] if y_true_first.ndim == 3 else y_true_first.shape[-2])
        y_true_first_vl = _to_var_level(y_true_first, n_levels=n_levels, n_lats=n_lats_first)
        if y_true_first_vl.shape[0] < 2:
            raise ValueError("Need at least 2 timesteps to compute persistence MAE row")

        pers_abs_err = np.abs(y_true_first_vl[1:] - y_true_first_vl[:-1])
        pers_abs_err_strat = pers_abs_err[:, :, strat_mask, :, :]
        pers_row = {
            "MAE": float(np.nanmean(pers_abs_err)),
            "MAE_strat": float(np.nanmean(pers_abs_err_strat)),
            "MAE_temp": float(np.nanmean(pers_abs_err[:, VARIABLE_TO_IDX["temp"], :, :, :])),
            "MAE_ucomp": float(np.nanmean(pers_abs_err[:, VARIABLE_TO_IDX["ucomp"], :, :, :])),
            "MAE_vcomp": float(np.nanmean(pers_abs_err[:, VARIABLE_TO_IDX["vcomp"], :, :, :])),
            "MAE_strat_temp": float(np.nanmean(pers_abs_err_strat[:, VARIABLE_TO_IDX["temp"], :, :, :])),
            "MAE_strat_ucomp": float(np.nanmean(pers_abs_err_strat[:, VARIABLE_TO_IDX["ucomp"], :, :, :])),
            "MAE_strat_vcomp": float(np.nanmean(pers_abs_err_strat[:, VARIABLE_TO_IDX["vcomp"], :, :, :])),
            "num_parameters": float("nan"),
        }
        if extra_columns:
            for col_name in extra_columns.keys():
                pers_row[col_name] = float("nan")
        rows[str(persistence_label)] = pers_row

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "model"
    base_cols = [
        "MAE",
        "MAE_strat",
        "MAE_temp",
        "MAE_ucomp",
        "MAE_vcomp",
        "MAE_strat_temp",
        "MAE_strat_ucomp",
        "MAE_strat_vcomp",
        "num_parameters",
    ]
    extra_cols = list(extra_columns.keys()) if extra_columns else []
    df = df[base_cols + extra_cols]
    return df


def build_mae_summary_dataframes_from_results(
    results_no_ssw: Any,
    results_with_ssw: Any,
    pressure_levels: Sequence[float],
    *,
    strat_max_pressure_hpa: float,
    parameter_estimator: Optional[Callable[[str, Mapping[str, Any]], Optional[float]]] = None,
    extra_columns: Optional[Mapping[str, Callable[[Mapping[str, Any]], Any]]] = None,
    include_persistence_row: bool = True,
    persistence_label: str = "Persistence",
    round_digits: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return MAE summary DataFrames for no-SSW and with-SSW model collections.

    Notes:
    - `MAE_strat*` uses pressure levels with `pressure <= strat_max_pressure_hpa`.
    - `num_parameters` is read from loaded metadata (no heuristic fallback).
    - Adds one persistence baseline row per DataFrame by default.
    - `extra_columns` accepts callables for easy notebook-time extension without changing core logic.
    """
    pressure_levels = np.asarray(pressure_levels, dtype=float)
    entries_no = _entries_from_results(results_no_ssw, "results_no_ssw")
    entries_with = _entries_from_results(results_with_ssw, "results_with_ssw")

    df_no = _build_mae_summary_dataframe(
        entries_no,
        pressure_levels,
        strat_max_pressure_hpa,
        parameter_estimator=parameter_estimator,
        extra_columns=extra_columns,
        include_persistence_row=include_persistence_row,
        persistence_label=persistence_label,
    )
    df_with = _build_mae_summary_dataframe(
        entries_with,
        pressure_levels,
        strat_max_pressure_hpa,
        parameter_estimator=parameter_estimator,
        extra_columns=extra_columns,
        include_persistence_row=include_persistence_row,
        persistence_label=persistence_label,
    )

    if round_digits is not None:
        df_no = df_no.round(round_digits)
        df_with = df_with.round(round_digits)

    return df_no, df_with


def _build_mae_pressure_profiles_data_from_entries(
    entries_no: Sequence[Mapping[str, Any]],
    entries_with: Sequence[Mapping[str, Any]],
    pressure_levels: np.ndarray,
    *,
    variable: str,
) -> dict[str, Any]:
    var_idx = _validate_variable(variable)
    n_levels = int(pressure_levels.shape[0])

    profiles_no = {
        e["label"]: _mae_profile(e["y_true"], e["y_pred"], var_idx, n_levels)
        for e in entries_no
    }
    profiles_with = {
        e["label"]: _mae_profile(e["y_true"], e["y_pred"], var_idx, n_levels)
        for e in entries_with
    }

    n_lats_no = int(entries_no[0]["y_true"].shape[-2])
    n_lats_with = int(entries_with[0]["y_true"].shape[-2])
    persistence_no = _persistence_profile_from_y_true(entries_no[0]["y_true"], var_idx, n_levels, n_lats_no)
    persistence_with = _persistence_profile_from_y_true(entries_with[0]["y_true"], var_idx, n_levels, n_lats_with)

    all_profiles = list(profiles_no.values()) + [persistence_no] + list(profiles_with.values()) + [persistence_with]
    xmax = float(max(np.nanmax(p) for p in all_profiles) * 1.05)

    ordered_labels: list[str] = []
    seen_labels: set[str] = set()
    for label in list(profiles_no.keys()) + list(profiles_with.keys()):
        if label not in seen_labels:
            seen_labels.add(label)
            ordered_labels.append(label)

    n_labels = len(ordered_labels)
    if n_labels <= 20:
        palette = plt.cm.Dark2(np.linspace(0.0, 1.0, max(1, n_labels)))
    else:
        palette = plt.cm.gist_rainbow(np.linspace(0.0, 1.0, n_labels, endpoint=False))
    color_by_label = {label: palette[i] for i, label in enumerate(ordered_labels)}

    legend_source_labels = list(profiles_no.keys()) if len(profiles_no) >= len(profiles_with) else list(profiles_with.keys())
    return {
        "pressure_levels": pressure_levels,
        "profiles_no_ssw": profiles_no,
        "profiles_with_ssw": profiles_with,
        "persistence_no_ssw": persistence_no,
        "persistence_with_ssw": persistence_with,
        "ordered_labels": ordered_labels,
        "legend_source_labels": legend_source_labels,
        "color_by_label": color_by_label,
        "persistence_color": "black",
        "xmax": xmax,
        "variable": variable,
        "metric_label": VARIABLE_TO_METRIC[variable],
    }


def build_mae_pressure_profiles_data_from_results(
    results_no_ssw: Any,
    results_with_ssw: Any,
    pressure_levels: Sequence[float],
    *,
    variable: str = "temp",
) -> dict[str, Any]:
    pressure_levels = np.asarray(pressure_levels)
    entries_no = _entries_from_results(results_no_ssw, "results_no_ssw")
    entries_with = _entries_from_results(results_with_ssw, "results_with_ssw")
    return _build_mae_pressure_profiles_data_from_entries(
        entries_no,
        entries_with,
        pressure_levels,
        variable=variable,
    )


def _plot_mae_pressure_profiles_pair_from_data(
    ax_no: Any,
    ax_with: Any,
    plot_data: Mapping[str, Any],
    *,
    legend_handles: Optional[dict[str, Any]] = None,
    line_alpha: float = 0.8,
    invert_yaxis: bool = True,
) -> Tuple[dict[str, Any], Any]:
    if legend_handles is None:
        legend_handles = {}

    pressure_levels = np.asarray(plot_data["pressure_levels"])
    profiles_no = plot_data["profiles_no_ssw"]
    profiles_with = plot_data["profiles_with_ssw"]
    persistence_no = np.asarray(plot_data["persistence_no_ssw"])
    persistence_with = np.asarray(plot_data["persistence_with_ssw"])
    color_by_label = plot_data["color_by_label"]
    persistence_color = plot_data["persistence_color"]

    for label, prof in profiles_no.items():
        line, = ax_no.plot(prof, pressure_levels, color=color_by_label[label], lw=2, label=label, alpha=line_alpha)
        legend_handles.setdefault(label, line)
    persistence_line, = ax_no.plot(
        persistence_no, pressure_levels, color=persistence_color, lw=2.5, ls="--", label="Persistence"
    )

    for label, prof in profiles_with.items():
        line, = ax_with.plot(prof, pressure_levels, color=color_by_label[label], lw=2, label=label, alpha=line_alpha)
        legend_handles.setdefault(label, line)
    ax_with.plot(persistence_with, pressure_levels, color=persistence_color, lw=2.5, ls="--", label="Persistence")

    for ax in (ax_no, ax_with):
        ax.set_yscale("log")
        ax.set_xlim(0, float(plot_data["xmax"]))
        ax.grid(True, ls=":", alpha=0.35)
    if invert_yaxis:
        ax_no.invert_yaxis()

    return legend_handles, persistence_line


def plot_mae_pressure_profiles_from_results(
    results_no_ssw: Any,
    results_with_ssw: Any,
    pressure_levels: Sequence[float],
    *,
    variable: str = "temp",
    panel_titles: Tuple[str, str] = ("No SSW", "With SSW"),
):
    plot_data = build_mae_pressure_profiles_data_from_results(
        results_no_ssw,
        results_with_ssw,
        pressure_levels,
        variable=variable,
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharex=True, sharey=True, constrained_layout=False)
    legend_handles, persistence_line = _plot_mae_pressure_profiles_pair_from_data(
        axes[0],
        axes[1],
        plot_data,
    )

    axes[0].set_title(panel_titles[0], fontsize=10)
    axes[1].set_title(panel_titles[1], fontsize=10)
    axes[0].set_ylabel("Pressure (hPa)")
    xlabel = f"Mean Absolute Error for {plot_data['metric_label']}"
    axes[0].set_xlabel(xlabel)
    axes[1].set_xlabel(xlabel)

    legend_labels = list(plot_data["legend_source_labels"]) + ["Persistence"]
    legend_lines = [legend_handles[label] for label in plot_data["legend_source_labels"]] + [persistence_line]

    fig.legend(
        legend_lines,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.075, 0.02, 0.92, 0.08),  # x, y, width, height in figure coords
        mode="expand",
        ncol=min(4, len(legend_labels)),
        fontsize=10,
        frameon=True,
    )
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    plt.show()

    return fig, axes, dict(plot_data)


def plot_mae_pressure_profiles_from_results_all(
    results_no_ssw: Any,
    results_with_ssw: Any,
    pressure_levels: Sequence[float],
    *,
    variables: Sequence[str] = ("temp", "ucomp", "vcomp"),
    row_titles: Tuple[str, str] = ("No SSW", "With SSW"),
    col_titles: Optional[Sequence[str]] = None,
    legend_label_map: Optional[Mapping[str, str]] = None,
    background_poster: bool = False,
):
    if len(variables) == 0:
        raise ValueError("variables must contain at least 1 entry")
    if len(row_titles) != 2:
        raise ValueError("row_titles must contain exactly 2 entries")

    for variable in variables:
        _validate_variable(variable)

    if col_titles is None:
        col_titles = tuple(VARIABLE_TO_METRIC[v] for v in variables)
    elif len(col_titles) != len(variables):
        raise ValueError("col_titles must contain one entry per variable")
    if legend_label_map is None:
        legend_label_map = {}

    pressure_levels = np.asarray(pressure_levels)
    entries_no = _entries_from_results(results_no_ssw, "results_no_ssw")
    entries_with = _entries_from_results(results_with_ssw, "results_with_ssw")
    n_cols = len(variables)

    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(13 * n_cols / 3, 4),
        sharex="col",
        sharey=True,
        constrained_layout=False,
        squeeze=False,
    )

    if background_poster:
        background_rgb = np.array([47, 62, 234]) / 255
        tint = 0.9
        boxcolor = tuple((1 - tint) * background_rgb + tint * np.ones(3))
        fig.patch.set_facecolor(boxcolor)   # full figure background
        for ax in axes.flat:
            ax.set_facecolor(boxcolor)          # axes / plotting area background

    legend_handles: dict[str, Any] = {}
    persistence_line = None
    payload_by_variable: dict[str, dict[str, Any]] = {}
    for col, variable in enumerate(variables):
        plot_data = _build_mae_pressure_profiles_data_from_entries(
            entries_no,
            entries_with,
            pressure_levels,
            variable=variable,
        )
        payload_by_variable[variable] = plot_data

        legend_handles, this_persistence_line = _plot_mae_pressure_profiles_pair_from_data(
            axes[0, col],
            axes[1, col],
            plot_data,
            legend_handles=legend_handles,
            invert_yaxis=not (n_cols % 2 == 0 and col == n_cols - 1),
        )
        if persistence_line is None:
            persistence_line = this_persistence_line

        axes[0, col].set_title(col_titles[col], fontsize=10)
        axes[1, col].set_xlabel(f"Mean Absolute Error for {plot_data['metric_label']}")

    axes[0, 0].set_ylabel(f"{row_titles[0]}\nPressure (hPa)")
    axes[1, 0].set_ylabel(f"{row_titles[1]}\nPressure (hPa)")

    legend_source_labels: list[str] = []
    for variable in variables:
        for label in payload_by_variable[variable]["legend_source_labels"]:
            if label not in legend_source_labels:
                legend_source_labels.append(label)

    if persistence_line is None:
        raise RuntimeError("Failed to create persistence legend handle")

    legend_lines = [legend_handles[label] for label in legend_source_labels] + [persistence_line]
    legend_labels = [legend_label_map.get(label, label) for label in legend_source_labels]
    legend_labels.append(legend_label_map.get("Persistence", "Persistence"))
    legend = fig.legend(
        legend_lines,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.06, -0.02, 0.92, 0.08),  # x, y, width, height in figure coords
        mode="expand",
        ncol=min(4, len(legend_labels)),
        fontsize=9,
        frameon=True,
    )
    if background_poster:
        legend.get_frame().set_facecolor(boxcolor)
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_alpha(1.0)
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    plt.yticks([1000, 100, 10, 1, 0.1, 0.01])
    plt.show()

    payload = {
        "by_variable": payload_by_variable,
        "meta": {
            "variables": variables,
            "row_titles": row_titles,
            "col_titles": col_titles,
            "legend_label_map": dict(legend_label_map),
        },
    }
    return fig, axes, payload


def plot_model_mae_across_pressure_latitude(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pressure_levels: Sequence[float],
    lat_levels: Sequence[float],
    *,
    variable: str = "ucomp",
    panel_title: str = "Model",
    vmin_vmax: Optional[Tuple[float, float]] = None,
    cmap: str = "magma",
):
    """Single-model MAE pressure/latitude map (no persistence panel)."""
    var_idx = _validate_variable(variable)
    pressure_levels = np.asarray(pressure_levels)
    lat_levels = np.asarray(lat_levels)

    n_levels = int(pressure_levels.shape[0])
    n_lats = int(lat_levels.shape[0])

    mae_map = _mae_map(y_true, y_pred, var_idx, n_levels, n_lats)

    if vmin_vmax is None:
        vmin, vmax = float(np.nanmin(mae_map)), float(np.nanmax(mae_map))
    else:
        vmin, vmax = vmin_vmax

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=140, constrained_layout=True)
    mappable = ax.pcolormesh(
        lat_levels,
        pressure_levels,
        mae_map,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title(f"{panel_title} - {variable}")
    ax.grid(True, ls=":", alpha=0.3)

    cbar = fig.colorbar(mappable, ax=ax, orientation="horizontal", pad=0.01, fraction=0.08, aspect=40)
    cbar.set_label(f"Mean absolute error for {VARIABLE_TO_METRIC[variable]}")
    plt.show()

    payload = {
        "mae_map": mae_map,
        "meta": {"variable": variable, "vmin": float(vmin), "vmax": float(vmax)},
    }
    return fig, ax, payload


def plot_persistence_mae_across_pressure_latitude_2x3_from_results(
    persistence_no_ssw: Mapping[str, Mapping[str, Any]],
    persistence_with_ssw: Mapping[str, Mapping[str, Any]],
    pressure_levels: Sequence[float],
    lat_levels: Sequence[float],
    *,
    no_ssw_model_key: str,
    with_ssw_model_key: str,
    variables: Tuple[str, str, str] = ("temp", "ucomp", "vcomp"),
    row_titles: Tuple[str, str] = ("No SSW", "With SSW"),
    col_titles: Tuple[str, str, str] = ("Temp", "Ucomp", "Vcomp"),
    metric: str = "mae",
    vmin_vmax: Optional[Any] = None,
    cmap: str = "Reds",
    yticks = None,
):
    """2x3 persistence error grid with one horizontal colorbar per column.

    metric:
    - "mae": mean absolute persistence error
    - "std": standard deviation of absolute persistence error magnitudes
    """
    if no_ssw_model_key not in persistence_no_ssw:
        raise KeyError(f"{no_ssw_model_key!r} not found in persistence_no_ssw")
    if with_ssw_model_key not in persistence_with_ssw:
        raise KeyError(f"{with_ssw_model_key!r} not found in persistence_with_ssw")

    if len(variables) != 3 or len(col_titles) != 3:
        raise ValueError("variables and col_titles must each contain exactly 3 entries")

    for v in variables:
        _validate_variable(v)

    metric = str(metric).lower()
    metric_labels = {
        "mae": "MAE",
        "std": "STD",
    }
    if metric not in metric_labels:
        raise ValueError(f"metric must be one of {list(metric_labels)}, got {metric!r}")

    pressure_levels = np.asarray(pressure_levels)
    lat_levels = np.asarray(lat_levels)
    n_levels = int(pressure_levels.shape[0])
    n_lats = int(lat_levels.shape[0])

    pers_true_no = np.asarray(persistence_no_ssw[no_ssw_model_key]["pers_true"])
    pers_pred_no = np.asarray(persistence_no_ssw[no_ssw_model_key]["pers_pred"])
    pers_true_with = np.asarray(persistence_with_ssw[with_ssw_model_key]["pers_true"])
    pers_pred_with = np.asarray(persistence_with_ssw[with_ssw_model_key]["pers_pred"])

    pers_true_no_vl = _to_var_level(pers_true_no, n_levels=n_levels, n_lats=n_lats)
    pers_pred_no_vl = _to_var_level(pers_pred_no, n_levels=n_levels, n_lats=n_lats)
    pers_true_with_vl = _to_var_level(pers_true_with, n_levels=n_levels, n_lats=n_lats)
    pers_pred_with_vl = _to_var_level(pers_pred_with, n_levels=n_levels, n_lats=n_lats)

    pers_err_no_vl = pers_pred_no_vl - pers_true_no_vl
    pers_err_with_vl = pers_pred_with_vl - pers_true_with_vl

    maps = [[None for _ in range(3)] for _ in range(2)]
    for col, var in enumerate(variables):
        idx = VARIABLE_TO_IDX[var]
        if metric == "mae":
            maps[0][col] = np.nanmean(np.abs(pers_err_no_vl[:, idx, :, :, :]), axis=(0, 3))
            maps[1][col] = np.nanmean(np.abs(pers_err_with_vl[:, idx, :, :, :]), axis=(0, 3))
        else:  # metric == "std"
            maps[0][col] = np.nanstd(np.abs(pers_err_no_vl[:, idx, :, :, :]), axis=(0, 3))
            maps[1][col] = np.nanstd(np.abs(pers_err_with_vl[:, idx, :, :, :]), axis=(0, 3))

    col_scales = {}
    for col, var in enumerate(variables):
        vals = np.concatenate([maps[0][col].ravel(), maps[1][col].ravel()])
        if vmin_vmax is None:
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        elif isinstance(vmin_vmax, Mapping):
            vmin, vmax = vmin_vmax[var]
        else:
            vmin, vmax = vmin_vmax
        col_scales[var] = (float(vmin), float(vmax))

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(13, 4),
        dpi=140,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    mappables_by_col = [None, None, None]
    for row in range(2):
        for col, var in enumerate(variables):
            ax = axes[row, col]
            arr = maps[row][col]
            vmin, vmax = col_scales[var]
            m = ax.pcolormesh(
                lat_levels,
                pressure_levels,
                arr,
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if mappables_by_col[col] is None:
                mappables_by_col[col] = m

            ax.set_yscale("log")
            ax.grid(True, ls=":", alpha=0.3)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            if row == 1:
                ax.set_xlabel("Latitude")

    axes[0, 0].set_ylabel(f"{row_titles[0]}\nPressure (hPa)")
    axes[1, 0].set_ylabel(f"{row_titles[1]}\nPressure (hPa)")
    axes[0, 0].invert_yaxis()

    cbars = []
    for col in range(3):
        cbar = fig.colorbar(
            mappables_by_col[col],
            ax=axes[:, col],
            orientation="horizontal",
            pad=0.01,
            fraction=0.06,
            aspect=30,
        )
        cbar.set_label(f"{col_titles[col]} persistence {metric_labels[metric]}", fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        cbars.append(cbar)

    if yticks is not None:
        for row in range(2):
            axes[row, 0].set_yticks(yticks)

    plt.show()

    payload = {
        "maps": {
            row_titles[0]: {variables[c]: maps[0][c] for c in range(3)},
            row_titles[1]: {variables[c]: maps[1][c] for c in range(3)},
        },
        "meta": {
            "no_ssw_source_label": no_ssw_model_key,
            "with_ssw_source_label": with_ssw_model_key,
            "variables": variables,
            "col_titles": col_titles,
            "metric": metric,
            "col_scales": col_scales,
        },
        "colorbars": cbars,
    }
    return fig, axes, payload


def plot_mae_across_pressure_latitude_2x3_from_results(
    results_no_ssw: Any,
    results_with_ssw: Any,
    pressure_levels: Sequence[float],
    lat_levels: Sequence[float],
    *,
    no_ssw_selection: Tuple[str, str, str],
    with_ssw_selection: Tuple[str, str, str],
    variable: str = "ucomp",
    row_titles: Tuple[str, str] = ("No SSW", "With SSW"),
    col_titles: Optional[Tuple[str, str, str]] = None,
    n_pressure_bins: Optional[int] = None,
    n_lat_bins: Optional[int] = None,
    vmin_vmax: Optional[Any] = None,
    cmap: str = "magma",
    include_persistence: bool = True,
    persistence_col_title: str = "Persistence",
    yticks=None,
):
    """MAE pressure-latitude grid with one shared bottom colorbar.

    By default (`include_persistence=True`), renders a 2x4 layout:
    [Persistence, Choice 1, Choice 2, Choice 3].
    If `include_persistence=False`, renders a 2x3 layout with only model choices.
    """
    if len(no_ssw_selection) != 3:
        raise ValueError("no_ssw_selection must contain exactly 3 entries")
    if len(with_ssw_selection) != 3:
        raise ValueError("with_ssw_selection must contain exactly 3 entries")

    if col_titles is None:
        if tuple(no_ssw_selection) == tuple(with_ssw_selection):
            col_titles = tuple(no_ssw_selection)
        else:
            col_titles = ("Choice 1", "Choice 2", "Choice 3")
    elif len(col_titles) != 3:
        raise ValueError("col_titles must contain exactly 3 entries")

    var_idx = _validate_variable(variable)
    pressure_levels = np.asarray(pressure_levels)
    lat_levels = np.asarray(lat_levels)

    n_levels = int(pressure_levels.shape[0])
    n_lats = int(lat_levels.shape[0])

    if n_pressure_bins is None:
        n_pressure_bins = n_levels
    if n_lat_bins is None:
        n_lat_bins = n_lats

    n_pressure_bins = int(n_pressure_bins)
    n_lat_bins = int(n_lat_bins)
    if n_pressure_bins < 1 or n_pressure_bins > n_levels:
        raise ValueError(f"n_pressure_bins must be between 1 and {n_levels}, got {n_pressure_bins}")
    if n_lat_bins < 1 or n_lat_bins > n_lats:
        raise ValueError(f"n_lat_bins must be between 1 and {n_lats}, got {n_lat_bins}")

    entries_no = _entries_from_results(results_no_ssw, "results_no_ssw")
    entries_with = _entries_from_results(results_with_ssw, "results_with_ssw")
    entries_no_by_label = {e["label"]: e for e in entries_no}
    entries_with_by_label = {e["label"]: e for e in entries_with}

    def _entry_from_label(index: Mapping[str, dict[str, Any]], label: str, source_name: str) -> dict[str, Any]:
        if label not in index:
            available = ", ".join(sorted(index))
            raise KeyError(f"{label!r} not found in {source_name}. Available labels: [{available}]")
        return index[label]

    maps = [[None for _ in range(3)] for _ in range(2)]
    p_plot = pressure_levels
    lat_plot = lat_levels

    for col in range(3):
        e_no = _entry_from_label(entries_no_by_label, no_ssw_selection[col], "results_no_ssw")
        e_with = _entry_from_label(entries_with_by_label, with_ssw_selection[col], "results_with_ssw")

        mae_no_raw = _mae_map(e_no["y_true"], e_no["y_pred"], var_idx, n_levels, n_lats)
        mae_with_raw = _mae_map(e_with["y_true"], e_with["y_pred"], var_idx, n_levels, n_lats)

        mae_no, p_plot, lat_plot = _aggregate_map(
            mae_no_raw, pressure_levels, lat_levels, n_pressure_bins, n_lat_bins
        )
        mae_with, _, _ = _aggregate_map(
            mae_with_raw, pressure_levels, lat_levels, n_pressure_bins, n_lat_bins
        )

        maps[0][col] = mae_no
        maps[1][col] = mae_with

    pers_no_raw = _persistence_map_from_y_true(entries_no[0]["y_true"], var_idx, n_levels, n_lats)
    pers_with_raw = _persistence_map_from_y_true(entries_with[0]["y_true"], var_idx, n_levels, n_lats)
    pers_no, _, _ = _aggregate_map(
        pers_no_raw, pressure_levels, lat_levels, n_pressure_bins, n_lat_bins
    )
    pers_with, _, _ = _aggregate_map(
        pers_with_raw, pressure_levels, lat_levels, n_pressure_bins, n_lat_bins
    )
    persistence_vals = np.concatenate([pers_no.ravel(), pers_with.ravel()])

    if vmin_vmax is None:
        vmin, vmax = float(np.nanmin(persistence_vals)), float(np.nanmax(persistence_vals))
    elif isinstance(vmin_vmax, Mapping):
        if "all" in vmin_vmax:
            vmin, vmax = vmin_vmax["all"]
        else:
            # Backward-compatible handling if per-column ranges are provided:
            # use the global min/max envelope across those column ranges.
            ranges = []
            for col in range(3):
                if col_titles[col] in vmin_vmax:
                    ranges.append(tuple(vmin_vmax[col_titles[col]]))
                elif col in vmin_vmax:
                    ranges.append(tuple(vmin_vmax[col]))
            if len(ranges) == 3:
                vmin = float(min(r[0] for r in ranges))
                vmax = float(max(r[1] for r in ranges))
            else:
                raise KeyError(
                    "For a shared colorbar, provide vmin_vmax as (vmin, vmax), "
                    "or mapping with key 'all', or per-column ranges for all 3 columns."
                )
    else:
        vmin, vmax = vmin_vmax
    shared_scale = (float(vmin), float(vmax))

    if include_persistence:
        ncols = 4
        panel_titles = (str(persistence_col_title),) + tuple(col_titles)
        panel_maps = [
            [pers_no, maps[0][0], maps[0][1], maps[0][2]],
            [pers_with, maps[1][0], maps[1][1], maps[1][2]],
        ]
        fig_w = 12
    else:
        ncols = 3
        panel_titles = tuple(col_titles)
        panel_maps = maps
        fig_w = 9

    fig, axes = plt.subplots(
        nrows=2,
        ncols=ncols,
        figsize=(fig_w, 4),
        dpi=140,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    mappable = None
    for row in range(2):
        for col in range(ncols):
            ax = axes[row, col]
            arr = panel_maps[row][col]
            m = ax.pcolormesh(
                lat_plot,
                p_plot,
                arr,
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if mappable is None:
                mappable = m

            ax.set_yscale("log")
            ax.grid(True, ls=":", alpha=0.3)
            if row == 0:
                ax.set_title(panel_titles[col], fontsize=10)
            if row == 1:
                ax.set_xlabel("Latitude")

    axes[0, 0].set_ylabel(f"{row_titles[0]}\nPressure (hPa)")
    axes[1, 0].set_ylabel(f"{row_titles[1]}\nPressure (hPa)")
    axes[0, 0].invert_yaxis()

    cbar = fig.colorbar(
        mappable,
        ax=axes,
        orientation="horizontal",
        pad=0.06,
        fraction=0.05,
        aspect=50,
    )
    cbar.set_label(f"Mean absolute error for {VARIABLE_TO_METRIC[variable]}", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if yticks is not None:
        for row in range(2):
            axes[row, 0].set_yticks(yticks)

    plt.show()

    payload = {
        "maps": {
            row_titles[0]: {col_titles[c]: maps[0][c] for c in range(3)},
            row_titles[1]: {col_titles[c]: maps[1][c] for c in range(3)},
        },
        "meta": {
            "variable": variable,
            "no_ssw_selection": no_ssw_selection,
            "with_ssw_selection": with_ssw_selection,
            "col_titles": col_titles,
            "n_pressure_bins": n_pressure_bins,
            "n_lat_bins": n_lat_bins,
            "shared_scale": shared_scale,
            "scale_source": "persistence",
            "include_persistence": bool(include_persistence),
            "panel_titles": panel_titles,
        },
        "persistence": {"no_ssw": pers_no, "with_ssw": pers_with},
        "colorbar": cbar,
        "colorbars": [cbar],
    }
    return fig, axes, payload


def plot_mae_pressure_latitude_grid_from_results(
    results_no_ssw: Any,
    results_with_ssw: Any,
    pressure_levels: Sequence[float],
    lat_levels: Sequence[float],
    *,
    architecture_names: Sequence[str],
    variable: str = "ucomp",
    panel_titles: Tuple[str, str] = ("No SSW", "With SSW"),
    architecture_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    n_pressure_bins: Optional[int] = None,
    n_lat_bins: Optional[int] = None,
    vmin_vmax: Optional[Tuple[float, float]] = None,
    cmap: str = "magma",
):
    """Fixed architecture-row MAE pressure-latitude grid with persistence in top row."""
    if not architecture_names:
        raise ValueError("architecture_names must be provided and non-empty")

    var_idx = _validate_variable(variable)
    pressure_levels = np.asarray(pressure_levels)
    lat_levels = np.asarray(lat_levels)

    n_levels = int(pressure_levels.shape[0])
    n_lats = int(lat_levels.shape[0])

    if n_pressure_bins is None:
        n_pressure_bins = n_levels
    if n_lat_bins is None:
        n_lat_bins = n_lats

    n_pressure_bins = int(n_pressure_bins)
    n_lat_bins = int(n_lat_bins)
    if n_pressure_bins < 1 or n_pressure_bins > n_levels:
        raise ValueError(f"n_pressure_bins must be between 1 and {n_levels}, got {n_pressure_bins}")
    if n_lat_bins < 1 or n_lat_bins > n_lats:
        raise ValueError(f"n_lat_bins must be between 1 and {n_lats}, got {n_lat_bins}")

    aliases = dict(DEFAULT_ARCHITECTURE_ALIASES)
    if architecture_aliases:
        for k, v in architecture_aliases.items():
            aliases[k] = list(v)

    entries_no = _entries_from_results(results_no_ssw, "results_no_ssw")
    entries_with = _entries_from_results(results_with_ssw, "results_with_ssw")

    def _canon(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    def _patterns(arch: str) -> list[str]:
        arch_canon = _canon(arch)
        pats = [arch]
        for alias_key, alias_values in aliases.items():
            if _canon(alias_key) == arch_canon:
                pats.append(alias_key)
                pats.extend(alias_values)
        return [_canon(p) for p in pats if _canon(p)]

    def _passes_constraints(arch: str, cand: str) -> bool:
        arch_canon = _canon(arch)
        if arch_canon in {"cnn2d", "cnn3d"} and "unet" in cand:
            return False
        if arch_canon in {"cnn2dunet", "cnn3dunet"} and "unet" not in cand:
            return False
        if "transformerwithcnn2" in arch_canon and "2d" not in cand:
            return False
        if "transformerwithcnn3" in arch_canon and "3d" not in cand:
            return False
        if arch_canon.endswith("globalattention") and "global" not in cand:
            return False
        if arch_canon.endswith("swinattention") and "swin" not in cand:
            return False
        if arch_canon in {"meshgnn2d", "meshgnn", "meshgnngridtomeshmeshandmeshtogrid", "GNN-MeshGrid"} and "mesh" not in cand:
            return False
        if arch_canon in {"simplegnn2d", "gnn", "gnnbasedencoderregulargridmpnndecoder", "GNN-RegularGrid"} and "gnn" not in cand:
            return False
        return True

    def _score(arch: str, candidates: Sequence[str]) -> int:
        pats = _patterns(arch)
        best = -1
        for cand in candidates:
            if not _passes_constraints(arch, cand):
                continue
            score = -1
            for pat in pats:
                if cand == pat:
                    score = max(score, 120)
                elif pat in cand:
                    score = max(score, 100)
                elif cand in pat:
                    score = max(score, 70)
            best = max(best, score)
        return best

    def _build_models(entries: Sequence[dict[str, Any]]):
        models = []
        p_out = pressure_levels
        lat_out = lat_levels
        for e in entries:
            mae = _mae_map(e["y_true"], e["y_pred"], var_idx, n_levels, n_lats)
            mae, p_out, lat_out = _aggregate_map(mae, pressure_levels, lat_levels, n_pressure_bins, n_lat_bins)
            candidates = [_canon(e["label"]), _canon(e["model_type"]), _canon(f"{e['label']}_{e['model_type']}")]
            models.append({"label": e["label"], "map": mae, "candidates": [c for c in candidates if c]})
        return models, p_out, lat_out

    def _match_rows(models: Sequence[dict[str, Any]], archs: Sequence[str]):
        used = set()
        rows = []
        for arch in archs:
            best_i = None
            best_s = -1
            for i, m in enumerate(models):
                if i in used:
                    continue
                s = _score(arch, m["candidates"])
                if s > best_s:
                    best_s = s
                    best_i = i
            if best_i is not None and best_s >= 70:
                used.add(best_i)
                rows.append((arch, models[best_i]))
            else:
                rows.append((arch, None))
        return rows

    pers_no_raw = _persistence_map_from_y_true(entries_no[0]["y_true"], var_idx, n_levels, n_lats)
    pers_with_raw = _persistence_map_from_y_true(entries_with[0]["y_true"], var_idx, n_levels, n_lats)

    pers_no, p_plot, lat_plot = _aggregate_map(
        pers_no_raw, pressure_levels, lat_levels, n_pressure_bins, n_lat_bins
    )
    pers_with, _, _ = _aggregate_map(
        pers_with_raw, pressure_levels, lat_levels, n_pressure_bins, n_lat_bins
    )

    models_no, p_plot, lat_plot = _build_models(entries_no)
    models_with, _, _ = _build_models(entries_with)

    rows_no = _match_rows(models_no, architecture_names)
    rows_with = _match_rows(models_with, architecture_names)

    all_vals = [pers_no.ravel(), pers_with.ravel()]
    for _, m in rows_no + rows_with:
        if m is not None:
            all_vals.append(m["map"].ravel())
    all_vals = np.concatenate(all_vals)

    if vmin_vmax is None:
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    else:
        vmin, vmax = vmin_vmax

    n_arch_rows = len(architecture_names)
    n_rows = n_arch_rows + 1
    fig_h = max(7.0, 2.2 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(10, fig_h),
        dpi=140,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    mappable = axes[0, 0].pcolormesh(
        lat_plot, p_plot, pers_no, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
    )
    axes[0, 0].set_title(f"{panel_titles[0]} - Persistence", fontsize=9)

    axes[0, 1].pcolormesh(
        lat_plot, p_plot, pers_with, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
    )
    axes[0, 1].set_title(f"{panel_titles[1]} - Persistence", fontsize=9)

    for row, arch in enumerate(architecture_names, start=1):
        ax_l = axes[row, 0]
        _, m_l = rows_no[row - 1]
        if m_l is not None:
            ax_l.pcolormesh(lat_plot, p_plot, m_l["map"], shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax_l.set_title(f"{panel_titles[0]} - [{m_l['label']}]", fontsize=9)
        else:
            ax_l.axis("off")
            ax_l.text(
                0.5,
                0.5,
                f"{panel_titles[0]} - {arch}\\n(Not loaded)",
                ha="center",
                va="center",
                transform=ax_l.transAxes,
                fontsize=4,
            )

        ax_r = axes[row, 1]
        _, m_r = rows_with[row - 1]
        if m_r is not None:
            ax_r.pcolormesh(lat_plot, p_plot, m_r["map"], shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax_r.set_title(f"{panel_titles[1]} - [{m_r['label']}]", fontsize=9)
        else:
            ax_r.axis("off")
            ax_r.text(
                0.5,
                0.5,
                f"{panel_titles[1]} - {arch}\\n(Not loaded)",
                ha="center",
                va="center",
                transform=ax_r.transAxes,
                fontsize=4,
            )

    for row in range(n_rows):
        for col in range(2):
            ax = axes[row, col]
            if not ax.has_data():
                continue
            ax.set_yscale("log")
            ax.grid(True, ls=":", alpha=0.3)

    for row in range(n_rows):
        if axes[row, 0].has_data():
            axes[row, 0].set_ylabel("Pressure (hPa)")

    for col in range(2):
        axes[-1, col].set_xlabel("Latitude")

    axes[0, 0].invert_yaxis()

    cbar = fig.colorbar(mappable, ax=axes, orientation="horizontal", pad=0.02, fraction=0.04, aspect=50)
    cbar.set_label(f"Mean absolute error for {VARIABLE_TO_METRIC[variable]}")

    plt.show()

    payload = {
        "rows_no_ssw": rows_no,
        "rows_with_ssw": rows_with,
        "persistence": {"no_ssw": pers_no, "with_ssw": pers_with},
        "meta": {
            "architecture_names": list(architecture_names),
            "n_rows": n_rows,
            "n_pressure_bins": n_pressure_bins,
            "n_lat_bins": n_lat_bins,
            "vmin": float(vmin),
            "vmax": float(vmax),
        },
    }
    return fig, axes, payload


def compute_ep_flux_and_pv_from_data(
    u,
    v,
    temp,
    pressure_levels,
    lat_levels,
    lon_levels,
    climate,
    lat_min=0,
    lat_max=90,
    pres_min=0.1,
    pres_max=1000,
    ):

    coords = {
        "time": np.arange(temp.shape[0]),
        "pres": pressure_levels,
        "lat": lat_levels,
        "lon": lon_levels,
    }

    u_xr = xr.DataArray(u, dims=("time", "pres", "lat", "lon"), coords=coords)
    v_xr = xr.DataArray(v, dims=("time", "pres", "lat", "lon"), coords=coords)
    t_xr = xr.DataArray(temp, dims=("time", "pres", "lat", "lon"), coords=coords)

    ep1_xr, ep2_xr, div1_xr, div2_xr = climate.ComputeEPfluxDivXr(
        u_xr,
        v_xr,
        t_xr,
        lon="lon",
        lat="lat",
        pres="pres",
        time="time",
    )

    pv_xr = climate.PotentialVorticity(
        u_xr,
        v_xr,
        t_xr,
        lon="lon",
        lat="lat",
        pres="pres",
    )

    lat_vals = np.asarray(lat_levels)
    pres_vals = np.asarray(pressure_levels)

    lat_slice = slice(lat_min, lat_max) if lat_vals[0] <= lat_vals[-1] else slice(lat_max, lat_min)
    pres_slice = slice(pres_min, pres_max) if pres_vals[0] <= pres_vals[-1] else slice(pres_max, pres_min)

    ep1_plot = ep1_xr.mean("time").sel(lat=lat_slice, pres=pres_slice).transpose("pres", "lat")
    ep2_plot = ep2_xr.mean("time").sel(lat=lat_slice, pres=pres_slice).transpose("pres", "lat")
    div_plot = (div1_xr + div2_xr).mean("time").sel(lat=lat_slice, pres=pres_slice).transpose("pres", "lat")
    ubar_plot = u_xr.mean(("time", "lon")).sel(lat=lat_slice, pres=pres_slice).transpose("pres", "lat")
    pv_plot = pv_xr.mean(("time", "lon")).sel(lat=lat_slice, pres=pres_slice).transpose("pres", "lat")

    return ep1_plot, ep2_plot, div_plot, ubar_plot, pv_plot

def _resolve_div_levels(
    div_levels: Optional[Sequence[float]],
    vmin_vmax: Optional[Tuple[float, float]],
) -> tuple[np.ndarray, Optional[float], Optional[float]]:
    if vmin_vmax is not None:
        if len(vmin_vmax) != 2:
            raise ValueError("vmin_vmax must be a 2-item tuple/list: (vmin, vmax)")
        vmin, vmax = float(vmin_vmax[0]), float(vmin_vmax[1])
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError("vmin_vmax values must be finite numbers")
        if vmin >= vmax:
            raise ValueError(f"Expected vmin < vmax, got vmin={vmin}, vmax={vmax}")
    else:
        vmin = vmax = None

    if div_levels is None:
        div_levels = np.arange(-12, 12, 1)
    div_levels_arr = np.asarray(div_levels, dtype=float)
    if div_levels_arr.ndim != 1 or div_levels_arr.size < 2:
        raise ValueError("div_levels must be a 1D array-like with at least 2 values")
    if not np.all(np.isfinite(div_levels_arr)):
        raise ValueError("div_levels must contain only finite numbers")
    if not np.all(np.diff(div_levels_arr) > 0):
        raise ValueError("div_levels must be strictly increasing")

    if vmin is not None and vmax is not None:
        inner_levels = div_levels_arr[(div_levels_arr > vmin) & (div_levels_arr < vmax)]
        div_levels_plot = np.unique(np.concatenate(([vmin], inner_levels, [vmax])))
        if div_levels_plot.size < 2:
            raise ValueError(
                "Could not construct contour levels from div_levels and vmin_vmax; "
                "provide a wider vmin_vmax range."
            )
    else:
        div_levels_plot = div_levels_arr

    return div_levels_plot, vmin, vmax


def ep_flux_panel(
    fig,
    ax,
    ep1_plot,
    ep2_plot,
    div_plot,
    ubar_plot,
    pv_plot,
    climate,
    *,
    lat_min=0,
    lat_max=90,
    div_levels=None,
    u_levels=None,
    arrow_scale=4e16,
    arrow_width=0.002,
    pv_level=2e-6,
    cmap="PuOr_r",
    add_divergence=True,
    add_wind=True,
    add_tropopause=True,
    vmin_vmax: Optional[Tuple[float, float]] = None,
    wind_color="darkgray",
    wind_linewidth=0.8,
    tropopause_color="black",
    tropopause_linewidth=1.5,
    tropopause_linestyle="--",
    x_label="Latitude",
    y_label="Pressure [hPa]",
    title: Optional[str] = None,
    add_ep_flux_arrows: bool = True,
):
    div_levels_plot, _, _ = _resolve_div_levels(div_levels, vmin_vmax)
    if u_levels is None:
        u_levels = np.arange(-80, 81, 10)

    cf = None
    if add_divergence:
        cf = ax.contourf(
            div_plot["lat"],
            div_plot["pres"],
            div_plot,
            levels=div_levels_plot,
            cmap=cmap,
            extend="both",
        )

    if add_ep_flux_arrows:
        climate.PlotEPfluxArrows(
            ep1_plot["lat"],
            ep1_plot["pres"],
            ep1_plot,
            ep2_plot,
            fig,
            ax,
            yscale="log",
            invert_y=True,
            scale=arrow_scale,
            quiv_args={"width": arrow_width},
        )

    wind_cs = None
    if add_wind:
        wind_cs = ax.contour(
            ubar_plot["lat"],
            ubar_plot["pres"],
            ubar_plot,
            levels=u_levels,
            colors=wind_color,
            linewidths=wind_linewidth,
        )

    tropopause_cs = None
    if add_tropopause:
        tropopause_cs = ax.contour(
            pv_plot["lat"],
            pv_plot["pres"],
            np.abs(pv_plot),
            levels=[pv_level],
            colors=tropopause_color,
            linewidths=tropopause_linewidth,
            linestyles=tropopause_linestyle,
        )

    ax.set_xlim(lat_min, lat_max)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    return {
        "ep1": ep1_plot,
        "ep2": ep2_plot,
        "div": div_plot,
        "ubar": ubar_plot,
        "pv": pv_plot,
        "cf": cf,
        "wind_cs": wind_cs,
        "tropopause_cs": tropopause_cs,
    }


def plot_ep_flux_panel(
    u,
    v,
    temp,
    pressure_levels,
    lat_levels,
    lon_levels,
    climate,
    *,
    lat_min=0,
    lat_max=90,
    pres_min=0.1,
    pres_max=1000,
    div_levels=None,
    u_levels=None,
    arrow_scale=4e16,
    arrow_width=0.002,
    pv_level=2e-6,
    cmap="PuOr_r",
    figsize=(5, 6),
    add_divergence=True,
    add_wind=True,
    add_tropopause=True,
    colorbar=True,
    colorbar_label=r"Divergence (m s$^{-1}$ day$^{-1}$)",
    vmin_vmax: Optional[Tuple[float, float]] = None,
    wind_color="darkgray",
    wind_linewidth=0.8,
    tropopause_color="black",
    tropopause_linewidth=1.5,
    tropopause_linestyle="--",
    x_label="Latitude",
    y_label="Pressure [hPa]",
    show=True,
):
    ep1_plot, ep2_plot, div_plot, ubar_plot, pv_plot = compute_ep_flux_and_pv_from_data(
        u,
        v,
        temp,
        pressure_levels,
        lat_levels,
        lon_levels,
        climate,
        lat_min=lat_min,
        lat_max=lat_max,
        pres_min=pres_min,
        pres_max=pres_max,
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=130, constrained_layout=True)
    out = ep_flux_panel(
        fig,
        ax,
        ep1_plot,
        ep2_plot,
        div_plot,
        ubar_plot,
        pv_plot,
        climate,
        lat_min=lat_min,
        lat_max=lat_max,
        div_levels=div_levels,
        u_levels=u_levels,
        arrow_scale=arrow_scale,
        arrow_width=arrow_width,
        pv_level=pv_level,
        cmap=cmap,
        add_divergence=add_divergence,
        add_wind=add_wind,
        add_tropopause=add_tropopause,
        vmin_vmax=vmin_vmax,
        wind_color=wind_color,
        wind_linewidth=wind_linewidth,
        tropopause_color=tropopause_color,
        tropopause_linewidth=tropopause_linewidth,
        tropopause_linestyle=tropopause_linestyle,
        x_label=x_label,
        y_label=y_label,
    )

    if colorbar and out["cf"] is not None:
        cbar = fig.colorbar(out["cf"], ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)

  #  ax.invert_yaxis()
    if show:
        plt.show()

    return fig, ax, out


def plot_ep_flux_panel_row_with_shared_colorbar(
    truth_fields: Mapping[str, Any],
    model_fields: Sequence[Mapping[str, Any]],
    pressure_levels: Sequence[float],
    lat_levels: Sequence[float],
    lon_levels: Sequence[float],
    climate,
    *,
    lat_min=0,
    lat_max=90,
    pres_min=0.1,
    pres_max=1000,
    div_levels=None,
    u_levels=None,
    arrow_scale=4e16,
    arrow_width=0.002,
    pv_level=2e-6,
    cmap="PuOr_r",
    figsize: Optional[Tuple[float, float]] = None,
    add_divergence=True,
    add_wind=True,
    add_tropopause=True,
    colorbar=True,
    colorbar_label=r"divergence (ms$^{-1}$day$^{-1}$)",
    vmin_vmax: Optional[Tuple[float, float]] = None,
    wind_color="darkgray",
    wind_linewidth=0.8,
    tropopause_color="black",
    tropopause_linewidth=1.5,
    tropopause_linestyle="--",
    x_label="Latitude",
    y_label="Pressure [hPa]",
    truth_title="True",
    persistence_title: Optional[str] = None,
    plot_divergence_difference_below: bool = False,
    divergence_difference_label: str = r"divergence (ms$^{-1}$day$^{-1}$)",
    model_titles_ep_flux: Optional[Sequence[str]] = None,
    model_titles_divergence_error: Optional[Sequence[str]] = None,
    show=True,
):
    if not model_fields:
        raise ValueError("model_fields must be a non-empty list/sequence")

    def _extract_uvt(payload: Mapping[str, Any], payload_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def _pick(keys: Sequence[str], comp_name: str) -> np.ndarray:
            for key in keys:
                if key in payload:
                    return np.asarray(payload[key])
            raise KeyError(f"{payload_name} missing {comp_name} key. Tried {list(keys)}")

        u_arr = _pick(("u", "u_true", "u_pred"), "u")
        v_arr = _pick(("v", "v_true", "v_pred"), "v")
        t_arr = _pick(("t", "temp", "t_true", "temp_true", "t_pred", "temp_pred"), "t")
        if u_arr.shape != v_arr.shape or u_arr.shape != t_arr.shape:
            raise ValueError(
                f"{payload_name} requires matching u/v/t shapes, got {u_arr.shape}, {v_arr.shape}, {t_arr.shape}"
            )
        if u_arr.ndim != 4:
            raise ValueError(
                f"{payload_name} expects 4D u/v/t arrays [time, pressure, lat, lon], got {u_arr.shape}"
            )
        return u_arr, v_arr, t_arr

    u_true, v_true, t_true = _extract_uvt(truth_fields, "truth_fields")
    if persistence_title is not None:
        # Backward compatibility for existing notebook calls that passed this kwarg.
        truth_title = persistence_title

    if model_titles_ep_flux is not None and len(model_titles_ep_flux) != len(model_fields):
        raise ValueError(
            "model_titles_ep_flux must have the same length as model_fields. "
            f"Got {len(model_titles_ep_flux)} and {len(model_fields)}."
        )
    if model_titles_divergence_error is not None and len(model_titles_divergence_error) != len(model_fields):
        raise ValueError(
            "model_titles_divergence_error must have the same length as model_fields. "
            f"Got {len(model_titles_divergence_error)} and {len(model_fields)}."
        )

    panel_inputs: list[dict[str, Any]] = [
        {"label": str(truth_title), "u": u_true, "v": v_true, "t": t_true}
    ]
    for i, payload in enumerate(model_fields):
        default_label = str(payload.get("label", f"Model {i + 1}"))
        ep_label = str(model_titles_ep_flux[i]) if model_titles_ep_flux is not None else default_label
        div_label = (
            str(model_titles_divergence_error[i])
            if model_titles_divergence_error is not None
            else default_label
        )
        panel_inputs.append(
            {
                **payload,
                "label": ep_label,
                "difference_label": div_label,
            }
        )

    n_panels = len(panel_inputs)
    if figsize is None:
        if plot_divergence_difference_below:
            figsize = (max(7.0, 3.2 * n_panels), 10.0)
        else:
            figsize = (max(7.0, 3.2 * n_panels), 5.0)

    n_rows = 2 if plot_divergence_difference_below else 1
    fig, axes = plt.subplots(
        n_rows,
        n_panels,
        figsize=figsize,
        dpi=140,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if n_rows == 1:
        axes_top = np.asarray([axes]) if n_panels == 1 else np.asarray(axes)
        axes_bottom = None
    else:
        axes_grid = np.asarray(axes)
        if n_panels == 1:
            axes_grid = axes_grid.reshape(n_rows, 1)
        axes_top = axes_grid[0]
        axes_bottom = axes_grid[1]

    panel_outputs: list[dict[str, Any]] = []
    first_cf = None
    for idx, (ax, panel_input) in enumerate(zip(axes_top, panel_inputs)):
        u_arr, v_arr, t_arr = _extract_uvt(panel_input, f"panel_inputs[{idx}]")
        ep1_plot, ep2_plot, div_plot, ubar_plot, pv_plot = compute_ep_flux_and_pv_from_data(
            u_arr,
            v_arr,
            t_arr,
            pressure_levels,
            lat_levels,
            lon_levels,
            climate,
            lat_min=lat_min,
            lat_max=lat_max,
            pres_min=pres_min,
            pres_max=pres_max,
        )
        out = ep_flux_panel(
            fig,
            ax,
            ep1_plot,
            ep2_plot,
            div_plot,
            ubar_plot,
            pv_plot,
            climate,
            lat_min=lat_min,
            lat_max=lat_max,
            div_levels=div_levels,
            u_levels=u_levels,
            arrow_scale=arrow_scale,
            arrow_width=arrow_width,
            pv_level=pv_level,
            cmap=cmap,
            add_divergence=add_divergence,
            add_wind=add_wind,
            add_tropopause=add_tropopause,
            vmin_vmax=vmin_vmax,
            wind_color=wind_color,
            wind_linewidth=wind_linewidth,
            tropopause_color=tropopause_color,
            tropopause_linewidth=tropopause_linewidth,
            tropopause_linestyle=tropopause_linestyle,
            x_label="" if plot_divergence_difference_below else x_label,
            y_label=f"Actual EP-flux\n{y_label}" if plot_divergence_difference_below and idx == 0 else y_label if idx == 0 else "",
            title=str(panel_input["label"]),
        )
        if first_cf is None and out["cf"] is not None:
            first_cf = out["cf"]
        panel_outputs.append(out)

    difference_outputs: list[dict[str, Any]] = []
    diff_first_cf = None
    if plot_divergence_difference_below and axes_bottom is not None:
        true_div = panel_outputs[0]["div"]
        for idx, ax in enumerate(axes_bottom):
            if idx == 0:
                div_diff = true_div - true_div
            else:
                div_diff = true_div - panel_outputs[idx]["div"]

            diff_out = ep_flux_panel(
                fig,
                ax,
                panel_outputs[idx]["ep1"],
                panel_outputs[idx]["ep2"],
                div_diff,
                panel_outputs[idx]["ubar"],
                panel_outputs[idx]["pv"],
                climate,
                lat_min=lat_min,
                lat_max=lat_max,
                div_levels=div_levels,
                u_levels=u_levels,
                arrow_scale=arrow_scale,
                arrow_width=arrow_width,
                pv_level=pv_level,
                cmap=cmap,
                add_divergence=add_divergence,
                add_wind=add_wind,
                add_tropopause=add_tropopause,
                vmin_vmax=vmin_vmax,
                wind_color=wind_color,
                wind_linewidth=wind_linewidth,
                tropopause_color=tropopause_color,
                tropopause_linewidth=tropopause_linewidth,
                tropopause_linestyle=tropopause_linestyle,
                x_label=x_label,
                y_label=f"Divergence error\n{y_label}" if idx == 0 else "",
                title=None,
                add_ep_flux_arrows=False,
            )
            if diff_first_cf is None and diff_out["cf"] is not None:
                diff_first_cf = diff_out["cf"]
            difference_outputs.append(diff_out)

    label_fontsize = plt.rcParams["axes.titlesize"]
    tick_fontsize = plt.rcParams["axes.titlesize"]
    axes_for_text = np.vstack([axes_top, axes_bottom]) if axes_bottom is not None else np.asarray([axes_top])
    for ax in axes_for_text.flat:
        ax.xaxis.label.set_size(label_fontsize)
        ax.yaxis.label.set_size(label_fontsize)
        ax.tick_params(axis="both", which="both", labelsize=tick_fontsize)

    cbar = None
    if colorbar and first_cf is not None and not plot_divergence_difference_below:
        cbar = fig.colorbar(
            first_cf,
            ax=axes_top,
            orientation="horizontal",
            pad=0.03,
            fraction=0.05,
            aspect=50,
        )
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=plt.rcParams["axes.titlesize"])

    diff_cbar = None
    if plot_divergence_difference_below and colorbar and diff_first_cf is not None and axes_bottom is not None:
        diff_cbar = fig.colorbar(
            diff_first_cf,
            ax=axes_bottom,
            orientation="horizontal",
            pad=0.03,
            fraction=0.05,
            aspect=50,
        )
        if divergence_difference_label:
            diff_cbar.set_label(divergence_difference_label, fontsize=plt.rcParams["axes.titlesize"])
    # if len(model_fields) % 2 != 0:
    #     axes_arr[-1].invert_yaxis()
    if show:
        plt.show()

    if plot_divergence_difference_below:
        axes_out = np.vstack([axes_top, axes_bottom]) if axes_bottom is not None else np.asarray([axes_top])
    else:
        axes_out = axes_top

    return fig, axes_out, {
        "panels": panel_outputs,
        "colorbar": cbar,
        "difference_panels": difference_outputs,
        "difference_colorbar": diff_cbar,
    }


def _extract_uvt_fields(payload: Mapping[str, Any], payload_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _pick(keys: Sequence[str], comp_name: str) -> np.ndarray:
        for key in keys:
            if key in payload:
                return np.asarray(payload[key])
        raise KeyError(f"{payload_name} missing {comp_name} key. Tried {list(keys)}")

    u_arr = _pick(("u", "u_true", "u_pred"), "u")
    v_arr = _pick(("v", "v_true", "v_pred"), "v")
    t_arr = _pick(("t", "temp", "t_true", "temp_true", "t_pred", "temp_pred"), "t")

    if u_arr.shape != v_arr.shape or u_arr.shape != t_arr.shape:
        raise ValueError(
            f"{payload_name} requires matching u/v/t shapes, got {u_arr.shape}, {v_arr.shape}, {t_arr.shape}"
        )
    if u_arr.ndim != 4:
        raise ValueError(
            f"{payload_name} expects 4D u/v/t arrays [time, pressure, lat, lon], got {u_arr.shape}"
        )
    return u_arr, v_arr, t_arr


def _compute_ep_flux_wave_components_from_xr(
    u_xr: xr.DataArray,
    v_xr: xr.DataArray,
    t_xr: xr.DataArray,
    climate,
    waves: Sequence[int],
    *,
    lat_min: float,
    lat_max: float,
    pres_min: float,
    pres_max: float,
) -> dict[str, dict[str, xr.DataArray]]:
    lat_vals = np.asarray(u_xr["lat"].values)
    pres_vals = np.asarray(u_xr["pres"].values)
    lat_slice = slice(lat_min, lat_max) if lat_vals[0] <= lat_vals[-1] else slice(lat_max, lat_min)
    pres_slice = slice(pres_min, pres_max) if pres_vals[0] <= pres_vals[-1] else slice(pres_max, pres_min)

    def _mean_slice(field: xr.DataArray) -> xr.DataArray:
        return field.mean("time").sel(lat=lat_slice, pres=pres_slice).transpose("pres", "lat")

    def _bundle_from_raw(
        ep1_raw: xr.DataArray,
        ep2_raw: xr.DataArray,
        div1_raw: xr.DataArray,
        div2_raw: xr.DataArray,
    ) -> dict[str, xr.DataArray]:
        return {
            "ep1": _mean_slice(ep1_raw),
            "ep2": _mean_slice(ep2_raw),
            "div": _mean_slice(div1_raw + div2_raw),
        }

    ep1_full, ep2_full, div1_full, div2_full = climate.ComputeEPfluxDivXr(
        u_xr,
        v_xr,
        t_xr,
        lon="lon",
        lat="lat",
        pres="pres",
        time="time",
        wave=0,
    )
    components: dict[str, dict[str, xr.DataArray]] = {
        "full": _bundle_from_raw(ep1_full, ep2_full, div1_full, div2_full)
    }

    wave_bundles: list[dict[str, xr.DataArray]] = []
    for wave in waves:
        ep1_w, ep2_w, div1_w, div2_w = climate.ComputeEPfluxDivXr(
            u_xr,
            v_xr,
            t_xr,
            lon="lon",
            lat="lat",
            pres="pres",
            time="time",
            wave=[int(wave)],
        )
        bundle = _bundle_from_raw(ep1_w, ep2_w, div1_w, div2_w)
        components[f"k={int(wave)}"] = bundle
        wave_bundles.append(bundle)

    if wave_bundles:
        ep1_large = wave_bundles[0]["ep1"].copy(deep=True)
        ep2_large = wave_bundles[0]["ep2"].copy(deep=True)
        div_large = wave_bundles[0]["div"].copy(deep=True)
        for bundle in wave_bundles[1:]:
            ep1_large = ep1_large + bundle["ep1"]
            ep2_large = ep2_large + bundle["ep2"]
            div_large = div_large + bundle["div"]
    else:
        ep1_large = 0.0 * components["full"]["ep1"]
        ep2_large = 0.0 * components["full"]["ep2"]
        div_large = 0.0 * components["full"]["div"]

    components["residual"] = {
        "ep1": components["full"]["ep1"] - ep1_large,
        "ep2": components["full"]["ep2"] - ep2_large,
        "div": components["full"]["div"] - div_large,
    }
    return components


def compute_ep_flux_wave_decomposition_from_data(
    u: np.ndarray,
    v: np.ndarray,
    temp: np.ndarray,
    pressure_levels: Sequence[float],
    lat_levels: Sequence[float],
    lon_levels: Sequence[float],
    climate,
    *,
    waves: Sequence[int] = (1, 2, 3),
    lat_min=0,
    lat_max=90,
    pres_min=0.1,
    pres_max=1000,
) -> dict[str, dict[str, xr.DataArray]]:
    coords = {
        "time": np.arange(temp.shape[0]),
        "pres": pressure_levels,
        "lat": lat_levels,
        "lon": lon_levels,
    }
    u_xr = xr.DataArray(u, dims=("time", "pres", "lat", "lon"), coords=coords)
    v_xr = xr.DataArray(v, dims=("time", "pres", "lat", "lon"), coords=coords)
    t_xr = xr.DataArray(temp, dims=("time", "pres", "lat", "lon"), coords=coords)

    return _compute_ep_flux_wave_components_from_xr(
        u_xr,
        v_xr,
        t_xr,
        climate,
        waves,
        lat_min=lat_min,
        lat_max=lat_max,
        pres_min=pres_min,
        pres_max=pres_max,
    )


def _subtract_wave_component_sets(
    lhs: Mapping[str, Mapping[str, xr.DataArray]],
    rhs: Mapping[str, Mapping[str, xr.DataArray]],
    component_keys: Sequence[str],
) -> dict[str, dict[str, xr.DataArray]]:
    out: dict[str, dict[str, xr.DataArray]] = {}
    for key in component_keys:
        out[key] = {
            "ep1": lhs[key]["ep1"] - rhs[key]["ep1"],
            "ep2": lhs[key]["ep2"] - rhs[key]["ep2"],
            "div": lhs[key]["div"] - rhs[key]["div"],
        }
    return out


def _ensure_log_pressure_axis(ax) -> None:
    ax.set_yscale("log")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


def plot_ep_flux_wave_decomposition_grid(
    truth_fields: Mapping[str, Any],
    model_fields: Sequence[Mapping[str, Any]],
    pressure_levels: Sequence[float],
    lat_levels: Sequence[float],
    lon_levels: Sequence[float],
    climate,
    *,
    waves: Sequence[int] = (1, 2, 3),
    residual_title: str = "Small waves (k>=4)",
    model_rows_as_error: bool = True,
    model_error_label_suffix: str = " (truth - pred)",
    lat_min=0,
    lat_max=90,
    pres_min=0.1,
    pres_max=1000,
    div_levels: Optional[Sequence[float]] = None,
    arrow_scale: float = 4e16,
    arrow_width: float = 0.002,
    cmap: str = "PuOr_r",
    figsize: Optional[Tuple[float, float]] = None,
    colorbar: bool = True,
    colorbar_label: str = r"Divergence (m s$^{-1}$ day$^{-1}$)",
    vmin_vmax: Optional[Tuple[float, float]] = None,
    truth_label: str = "True",
    model_row_titles: Optional[Sequence[str]] = None,
    x_label: str = "Latitude",
    y_label: str = "Pressure [hPa]",
    add_model_row_arrows: bool = True,
    show: bool = True,
):
    if len(waves) != len(set(int(w) for w in waves)):
        raise ValueError(f"waves must contain unique integers. Got {list(waves)}")
    if any(int(w) <= 0 for w in waves):
        raise ValueError(f"waves must be positive integers. Got {list(waves)}")

    waves_int = tuple(int(w) for w in waves)
    component_keys = ["full", *[f"k={w}" for w in waves_int], "residual"]
    column_titles = ["All waves (k=0)", *[f"Wave {w}" for w in waves_int], residual_title]

    u_true, v_true, t_true = _extract_uvt_fields(truth_fields, "truth_fields")

    if model_row_titles is not None and len(model_row_titles) != len(model_fields):
        raise ValueError(
            "model_row_titles must have the same length as model_fields. "
            f"Got {len(model_row_titles)} and {len(model_fields)}."
        )

    truth_components = compute_ep_flux_wave_decomposition_from_data(
        u_true,
        v_true,
        t_true,
        pressure_levels,
        lat_levels,
        lon_levels,
        climate,
        waves=waves_int,
        lat_min=lat_min,
        lat_max=lat_max,
        pres_min=pres_min,
        pres_max=pres_max,
    )

    row_specs: list[dict[str, Any]] = [
        {
            "label": str(truth_label),
            "components": truth_components,
            "show_arrows": True,
            "is_error": False,
        }
    ]

    for i, model_payload in enumerate(model_fields):
        default_model_label = str(model_payload.get("label", f"Model {i + 1}"))
        model_label = str(model_row_titles[i]) if model_row_titles is not None else default_model_label
        u_model, v_model, t_model = _extract_uvt_fields(model_payload, f"model_fields[{i}]")
        model_components = compute_ep_flux_wave_decomposition_from_data(
            u_model,
            v_model,
            t_model,
            pressure_levels,
            lat_levels,
            lon_levels,
            climate,
            waves=waves_int,
            lat_min=lat_min,
            lat_max=lat_max,
            pres_min=pres_min,
            pres_max=pres_max,
        )
        if model_rows_as_error:
            components = _subtract_wave_component_sets(truth_components, model_components, component_keys)
            row_label = model_label
        else:
            components = {key: dict(model_components[key]) for key in component_keys}
            row_label = model_label

        row_specs.append(
            {
                "label": row_label,
                "components": components,
                "show_arrows": bool(add_model_row_arrows),
                "is_error": bool(model_rows_as_error),
            }
        )

    n_rows = len(row_specs)
    n_cols = len(column_titles)
    if figsize is None:
        figsize = (max(12.0, 2.8 * n_cols), max(4.5, 2.8 * n_rows))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        dpi=140,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes)
    if n_rows == 1 and n_cols == 1:
        axes_arr = axes_arr.reshape(1, 1)
    elif n_rows == 1:
        axes_arr = axes_arr.reshape(1, n_cols)
    elif n_cols == 1:
        axes_arr = axes_arr.reshape(n_rows, 1)

    first_cf = None
    panel_outputs: list[list[dict[str, Any]]] = []
    div_levels_plot, _, _ = _resolve_div_levels(div_levels, vmin_vmax)

    for row_idx, row in enumerate(row_specs):
        row_outputs: list[dict[str, Any]] = []
        for col_idx, (component_key, title) in enumerate(zip(component_keys, column_titles)):
            ax = axes_arr[row_idx, col_idx]
            fields = row["components"][component_key]
            if row_idx == 0:
                row_y_label = f"Actual EP-flux\n{y_label}"
            elif row["is_error"]:
                row_y_label = f"Divergence error\n{row['label']}\n{y_label}"
            else:
                row_y_label = f"{row['label']}\n{y_label}"
            out = ep_flux_panel(
                fig,
                ax,
                fields["ep1"],
                fields["ep2"],
                fields["div"],
                fields["div"],
                fields["div"],
                climate,
                lat_min=lat_min,
                lat_max=lat_max,
                div_levels=div_levels_plot,
                arrow_scale=arrow_scale,
                arrow_width=arrow_width,
                cmap=cmap,
                add_divergence=True,
                add_wind=False,
                add_tropopause=False,
                vmin_vmax=vmin_vmax,
                x_label=x_label if row_idx == n_rows - 1 else "",
                y_label=row_y_label if col_idx == 0 else "",
                title=title if row_idx == 0 else None,
                add_ep_flux_arrows=bool(row["show_arrows"]),
            )
            _ensure_log_pressure_axis(ax)
            if first_cf is None and out["cf"] is not None:
                first_cf = out["cf"]
            row_outputs.append(out)
        panel_outputs.append(row_outputs)

    label_fontsize = plt.rcParams["axes.titlesize"]
    tick_fontsize = plt.rcParams["axes.titlesize"]
    for ax in axes_arr.flat:
        ax.xaxis.label.set_size(label_fontsize)
        ax.yaxis.label.set_size(label_fontsize)
        ax.tick_params(axis="both", which="both", labelsize=tick_fontsize)

    cbar = None
    if colorbar and first_cf is not None:
        cbar = fig.colorbar(
            first_cf,
            ax=axes_arr,
            orientation="horizontal",
            pad=0.03,
            fraction=0.05,
            aspect=50,
        )
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)

    if show:
        plt.show()

    return fig, axes_arr, {
        "row_specs": row_specs,
        "panel_outputs": panel_outputs,
        "colorbar": cbar,
        "component_keys": component_keys,
        "column_titles": column_titles,
    }

__all__ = [
    "load_grid_from_processed_dataset",
    "load_evaluation_results",
    "load_artifact_collections",
    "build_mae_summary_dataframes_from_results",
    "build_mae_pressure_profiles_data_from_results",
    "plot_mae_pressure_profiles_from_results",
    "plot_mae_pressure_profiles_from_results_all",
    "plot_model_mae_across_pressure_latitude",
    "plot_persistence_mae_across_pressure_latitude_2x3_from_results",
    "plot_mae_across_pressure_latitude_2x3_from_results",
    "plot_mae_pressure_latitude_grid_from_results",
    "compute_ep_flux_and_pv_from_data",
    "compute_ep_flux_wave_decomposition_from_data",
    "ep_flux_panel",
    "plot_ep_flux_panel",
    "plot_ep_flux_panel_row_with_shared_colorbar",
    "plot_ep_flux_wave_decomposition_grid",
]
