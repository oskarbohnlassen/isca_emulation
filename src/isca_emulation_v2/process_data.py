from __future__ import annotations
import typer
from pathlib import Path
from typing import Any
from omegaconf import OmegaConf
from isca_emulation_v2.data.cnn2d import SimpleDatasetCNN2D, process_cnn2d
from isca_emulation_v2.data.cnn3d import SimpleDatasetCNN3D, process_cnn3d
from isca_emulation_v2.data.gnn2d import SimpleDatasetGNN2D, process_gnn2d
from isca_emulation_v2.data.mesh_gnn import SimpleDatasetMeshGNN, process_mesh_gnn


RAW_DATA_ROOT = Path("data/raw")
PROCESSED_DATA_ROOT = Path("data/processed")

def resolve_processed_dir(
    dataset_name: str,
) -> Path:
    return PROCESSED_DATA_ROOT / dataset_name

app = typer.Typer()

DATASET_NAMES = {
    "SimpleDatasetCNN2D": SimpleDatasetCNN2D,
    "SimpleDatasetCNN3D": SimpleDatasetCNN3D,
    "SimpleDatasetGNN2D": SimpleDatasetGNN2D,
    "SimpleDatasetMeshGNN": SimpleDatasetMeshGNN,
}

def _check_out_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Output path already exists: {path}")

def _split_ranges(total_time: int, split_cfg: list[float] | bool) -> dict[str, tuple[int, int]]:
    if split_cfg is False:
        return {"all": (0, total_time)}
    n_train = int(total_time * float(split_cfg[0]))
    n_val = int(total_time * float(split_cfg[1]))
    n_test = total_time - n_train - n_val
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_train + n_val + n_test),
    }

def prepare_from_config(data_cfg: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Run data preparation and write processed artifacts to disk.

    Returns:
        processed_dir, manifest, split_shapes
    """
    raw_cfg = OmegaConf.to_container(data_cfg["raw"], resolve=True)
    proc_cfg = OmegaConf.to_container(data_cfg["processing"], resolve=True)
    out_cfg = OmegaConf.to_container(data_cfg["output"], resolve=True)
    split_value = data_cfg["split"]
    split_cfg = False if split_value is False else list(split_value)
    stats_path = data_cfg.get("stats_path")

    processor_name = proc_cfg["name"]

    dataset_name = out_cfg["dataset_name"]
    processed_dir = resolve_processed_dir(dataset_name)

    _check_out_path(processed_dir)

    if processor_name == "cnn2d":
        manifest, split_shapes = process_cnn2d(
            raw_cfg=raw_cfg,
            proc_cfg=proc_cfg,
            out_cfg=out_cfg,
            split_cfg=split_cfg,
            split_ranges_fn=_split_ranges,
            stats_path=stats_path,
            processed_dir=processed_dir,
        )
    elif processor_name == "cnn3d":
        manifest, split_shapes = process_cnn3d(
            raw_cfg=raw_cfg,
            proc_cfg=proc_cfg,
            out_cfg=out_cfg,
            split_cfg=split_cfg,
            split_ranges_fn=_split_ranges,
            stats_path=stats_path,
            processed_dir=processed_dir,
        )
    elif processor_name in {"gnn2d", "gnn"}:
        manifest, split_shapes = process_gnn2d(
            raw_cfg=raw_cfg,
            proc_cfg=proc_cfg,
            out_cfg=out_cfg,
            split_cfg=split_cfg,
            split_ranges_fn=_split_ranges,
            stats_path=stats_path,
            processed_dir=processed_dir,
        )
    elif processor_name == "mesh_gnn":
        manifest, split_shapes = process_mesh_gnn(
            raw_cfg=raw_cfg,
            proc_cfg=proc_cfg,
            out_cfg=out_cfg,
            split_cfg=split_cfg,
            split_ranges_fn=_split_ranges,
            stats_path=stats_path,
            processed_dir=processed_dir,
        )
    else:
        raise NotImplementedError(f"Processor '{processor_name}' is not implemented.")

    return processed_dir, manifest, split_shapes

@app.command()
def main(
    data_cfg_name: str = typer.Argument(
        ...,
        help="YAML filename located in config_data (for example: default.yaml).",
    )
) -> None:
    """
    Main function to load and preprocess the data with OmegaConf.
    """

    cfg = OmegaConf.load(data_cfg_name)

    processed_dir, manifest, split_shapes = prepare_from_config(cfg)
    print(f"Data preparation complete. Processed data saved to: {processed_dir}")
    print(f"Manifest: {manifest}")
    print(f"Split shapes: {split_shapes}")
