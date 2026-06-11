from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def _load_dynamic_tensor(shard_path: Path) -> torch.Tensor:
    shard_obj = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
    if torch.is_tensor(shard_obj):
        return shard_obj
    if isinstance(shard_obj, dict) and "x_dyn" in shard_obj:
        return shard_obj["x_dyn"]
    if hasattr(shard_obj, "x_dyn"):
        return shard_obj.x_dyn
    raise TypeError(f"Unsupported shard format at: {shard_path}")


def _reduce_dims_for_processor(processor: str) -> tuple[int, ...]:
    if processor == "cnn2d":
        return (0, 2, 3)
    if processor == "cnn3d":
        return (0, 3, 4)
    if processor in {"gnn2d", "mesh_gnn"}:
        return (0, 1)
    raise ValueError(f"Unsupported processor: {processor}")


def _reshape_stats_for_processor(processor: str, tensor: torch.Tensor) -> torch.Tensor:
    if processor in {"gnn2d", "mesh_gnn"}:
        return tensor.reshape(1, tensor.shape[-1], 1, 1)
    return tensor


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a - b).abs()
    if not torch.isfinite(diff).all():
        return float("inf")
    return float(diff.max().item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare legacy stats computation vs NaN-safe computation on processed shards. "
            "Use this to verify no-NaN datasets are unchanged."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/cnn2d_coordinates_trig4_1000_to_3500_postprocessed_no_ssw"),
        help="Path to processed dataset directory (contains manifest.yaml and stats.pt).",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=8,
        help="Timesteps per streaming chunk when reducing shard tensors.",
    )
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for comparisons.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    manifest_path = dataset_dir / "manifest.yaml"
    stats_path = dataset_dir / "stats.pt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats not found: {stats_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}

    processor = str(manifest["processor"])
    stats_split = str(manifest.get("stats_split", "train"))
    if stats_split == "external":
        raise ValueError("Dataset uses external stats; cannot validate internal stat computation.")

    shards = manifest["split_shards"][stats_split]
    if len(shards) == 0:
        raise ValueError(f"No shards found for stats split '{stats_split}'.")

    reduce_dims = _reduce_dims_for_processor(processor)

    legacy_sum = None
    legacy_sumsq = None
    legacy_count = 0

    safe_sum = None
    safe_sumsq = None
    safe_count = None

    for shard_info in shards:
        tensor = _load_dynamic_tensor(dataset_dir / shard_info["path"])
        time_dim = int(tensor.shape[0])
        for start in range(0, time_dim, args.time_chunk):
            end = min(time_dim, start + args.time_chunk)
            tensor_chunk = tensor[start:end].to(torch.float64)

            chunk_sum_legacy = tensor_chunk.sum(dim=reduce_dims, keepdim=True)
            chunk_sumsq_legacy = (tensor_chunk * tensor_chunk).sum(dim=reduce_dims, keepdim=True)
            reduced_size = 1
            for dim in reduce_dims:
                reduced_size *= int(tensor_chunk.shape[dim])
            legacy_count += reduced_size
            if legacy_sum is None:
                legacy_sum = torch.zeros_like(chunk_sum_legacy)
                legacy_sumsq = torch.zeros_like(chunk_sumsq_legacy)
            legacy_sum += chunk_sum_legacy
            legacy_sumsq += chunk_sumsq_legacy

            finite = torch.isfinite(tensor_chunk)
            finite_tensor = torch.where(finite, tensor_chunk, torch.zeros_like(tensor_chunk))
            chunk_sum_safe = finite_tensor.sum(dim=reduce_dims, keepdim=True)
            chunk_sumsq_safe = (finite_tensor * finite_tensor).sum(dim=reduce_dims, keepdim=True)
            chunk_count_safe = finite.to(torch.float64).sum(dim=reduce_dims, keepdim=True)
            if safe_sum is None:
                safe_sum = torch.zeros_like(chunk_sum_safe)
                safe_sumsq = torch.zeros_like(chunk_sumsq_safe)
                safe_count = torch.zeros_like(chunk_count_safe)
            safe_sum += chunk_sum_safe
            safe_sumsq += chunk_sumsq_safe
            safe_count += chunk_count_safe

    legacy_mean64 = legacy_sum / legacy_count
    legacy_var64 = legacy_sumsq / legacy_count - legacy_mean64 * legacy_mean64
    legacy_mean = legacy_mean64.to(torch.float32)
    legacy_std = torch.sqrt(legacy_var64.clamp_min(1e-12)).to(torch.float32).clamp_min(1e-6)

    safe_valid_count = safe_count.clamp_min(1.0)
    safe_mean64 = safe_sum / safe_valid_count
    safe_var64 = safe_sumsq / safe_valid_count - safe_mean64 * safe_mean64
    safe_var64 = torch.where(safe_count > 0, safe_var64, torch.ones_like(safe_var64))
    safe_mean = safe_mean64.to(torch.float32)
    safe_std = torch.sqrt(safe_var64.clamp_min(1e-12)).to(torch.float32).clamp_min(1e-6)

    legacy_mean = _reshape_stats_for_processor(processor, legacy_mean)
    legacy_std = _reshape_stats_for_processor(processor, legacy_std)
    safe_mean = _reshape_stats_for_processor(processor, safe_mean)
    safe_std = _reshape_stats_for_processor(processor, safe_std)

    saved_stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    saved_mean = torch.as_tensor(saved_stats["mean"], dtype=torch.float32)
    saved_std = torch.as_tensor(saved_stats["std"], dtype=torch.float32)

    diff_legacy_vs_safe_mean = _max_abs_diff(legacy_mean, safe_mean)
    diff_legacy_vs_safe_std = _max_abs_diff(legacy_std, safe_std)
    diff_safe_vs_saved_mean = _max_abs_diff(safe_mean, saved_mean)
    diff_safe_vs_saved_std = _max_abs_diff(safe_std, saved_std)

    print(f"dataset: {dataset_dir}")
    print(f"processor: {processor}")
    print(f"stats_split: {stats_split}")
    print(f"max|legacy_mean - safe_mean|: {diff_legacy_vs_safe_mean:.8e}")
    print(f"max|legacy_std  - safe_std |: {diff_legacy_vs_safe_std:.8e}")
    print(f"max|safe_mean   - saved_mean|: {diff_safe_vs_saved_mean:.8e}")
    print(f"max|safe_std    - saved_std |: {diff_safe_vs_saved_std:.8e}")

    if diff_legacy_vs_safe_mean > args.atol or diff_legacy_vs_safe_std > args.atol:
        raise SystemExit(
            "FAILED: NaN-safe stats differ from legacy stats above tolerance. "
            "For a no-NaN dataset these should match."
        )

    print("PASS: NaN-safe stats match legacy stats within tolerance.")


if __name__ == "__main__":
    main()
