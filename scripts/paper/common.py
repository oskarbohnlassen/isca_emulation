"""Portable paths, pinned input verification, and retained-checkpoint loading."""
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from .manifest import read_manifest
MODEL_ORDER = ["cnn2d", "cnn2d_unet", "cnn3d", "cnn3d_unet",
               "transformer2d_global_attention", "transformer2d_swin",
               "transformer3d_global_attention", "transformer3d_swin",
               "gnn2d_gatedgcn", "mesh_gnn2d"]
LABELS = ["CNN2D", "CNN2D U-Net", "CNN3D", "CNN3D U-Net", "Transformer2D GA", "Transformer2D Swin",
          "Transformer3D GA", "Transformer3D Swin", "Grid GatedGCN", "Mesh GNN"]
REGIMES = ("no_ssw", "with_ssw")
VARIABLES = ("temp", "ucomp", "vcomp")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@dataclass
class Context:
    root: Path
    output: Path
    verified: set = field(default_factory=set)
    benchmark: object = None

    def input(self, relative):
        """Require exactly the evaluated input, independent of the checkout location."""
        relative = str(relative)
        hashes = read_manifest("input_hashes.json")
        if relative not in hashes:
            raise ValueError(f"Input not listed in the manifest's input_hashes section: {relative}")
        path = self.root / relative
        if relative not in self.verified:
            if not path.is_file():
                raise FileNotFoundError(f"Missing reproduction input: {path}. See scripts/README.md.")
            if sha(path) != hashes[relative]:
                raise ValueError(f"Reproduction input differs from the final evaluated version: {path}")
            self.verified.add(relative)
        return path

    def json(self, relative):
        return json.loads(self.input(relative).read_text())


def mean_std(values):
    values = np.asarray(values)
    if len(values) != 5 or not np.isfinite(values).all():
        raise ValueError("Five finite, distinct retained-seed results required")
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def load_selected(context):
    if context.benchmark is not None:
        return context.benchmark
    manifest = read_manifest("selected_checkpoints.json")
    assert manifest["verified"] and len(manifest["runs"]) == 100
    groups = defaultdict(list)
    for row in manifest["runs"]:
        stem = f"{row['regime']}_{row['model']}_seed{row['seed']}_{row['run_id']}"
        folder = f"results/paper/inputs/benchmark/compact/{stem}"
        info = context.json(f"{folder}/summary.json")
        for key in ("run_id", "seed", "model", "regime", "artifact_name", "checkpoint_file", "checkpoint_digest", "config_sha256"):
            assert info["run"][key] == row[key], (stem, key)
        assert info["samples"] == 249
        with np.load(context.input(f"{folder}/reductions.npz"), allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        groups[row["regime"], row["model"]].append((row, info, arrays))
    assert set(groups) == {(r, m) for r in REGIMES for m in MODEL_ORDER}
    for key, rows in groups.items():
        assert len(rows) == len({r[0]["seed"] for r in rows}) == 5, key
    context.benchmark = manifest, groups
    return context.benchmark
