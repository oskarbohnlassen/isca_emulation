from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml


matplotlib.use("Agg")


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

from notebook_utils import plot_utils as pu


def _make_graph_result(*, n_samples: int = 4, n_levels: int = 2, n_lats: int = 2, n_lons: int = 3):
    n_vars = len(pu.VARIABLE_TO_IDX)
    n_features = n_vars * n_levels
    y_true = np.arange(n_samples * n_lats * n_lons * n_features, dtype=np.float32).reshape(
        n_samples,
        n_lats * n_lons,
        n_features,
    )
    y_pred = y_true + 1.0
    return y_true, y_pred


def test_to_var_level_accepts_graph_layout():
    y_true, _ = _make_graph_result()

    got = pu._to_var_level(y_true, n_levels=2, n_lats=2)

    expected = y_true.reshape(4, 2, 3, 6).transpose(0, 3, 1, 2).reshape(4, 3, 2, 2, 3)
    np.testing.assert_allclose(got, expected)


def test_load_artifact_collections_canonicalizes_graph_results(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    manifest = {
        "grid": {
            "level": [1000.0, 500.0],
            "lat": [-45.0, 45.0],
            "lon": [0.0, 120.0, 240.0],
        }
    }
    (processed_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest))

    y_true, y_pred = _make_graph_result()

    no_ssw_artifact = tmp_path / "artifact_no"
    no_ssw_artifact.mkdir()
    torch.save(
        {
            "y_true_unscaled": y_true,
            "y_pred_unscaled": y_pred,
            "evaluation_info": {"model_type": "SimpleGNN2D"},
            "metrics": {},
        },
        no_ssw_artifact / "evaluation_results.pt",
    )

    with_ssw_artifact = tmp_path / "artifact_with"
    with_ssw_artifact.mkdir()
    torch.save(
        {
            "y_true_unscaled": y_true,
            "y_pred_unscaled": y_pred,
            "evaluation_info": {"model_type": "MeshGNN2D"},
            "metrics": {},
        },
        with_ssw_artifact / "evaluation_results.pt",
    )

    bundle = pu.load_artifact_collections(
        {"gnn": no_ssw_artifact},
        {"mesh": with_ssw_artifact},
        processed_dataset_path_to_get_grid=processed_dir,
        verbose=False,
    )

    result_no = bundle["results_no_ssw_models"][0][1]
    payload_no = bundle["model_payloads_no_ssw"]["gnn"]
    persistence_with = bundle["persistence_with_ssw"]["mesh"]

    assert result_no["y_true_unscaled"].shape == (4, 3, 2, 2, 3)
    assert result_no["y_pred_unscaled"].shape == (4, 3, 2, 2, 3)
    assert payload_no["y_true"].shape == (4, 3, 2, 2, 3)
    assert payload_no["pers_true"].shape == (3, 3, 2, 2, 3)
    assert persistence_with["pers_pred"].shape == (3, 3, 2, 2, 3)

    df_no, df_with = pu.build_mae_summary_dataframes_from_results(
        bundle["results_no_ssw_models"],
        bundle["results_with_ssw_models"],
        bundle["pressure_levels"],
        strat_max_pressure_hpa=500.0,
    )

    assert "gnn" in df_no.index
    assert "mesh" in df_with.index


def test_build_mae_summary_dataframes_accepts_raw_graph_results():
    y_true, y_pred = _make_graph_result()
    pressure_levels = np.asarray([1000.0, 500.0], dtype=float)

    results_no = [
        ("gnn", {"y_true_unscaled": y_true, "y_pred_unscaled": y_pred, "evaluation_info": {"model_type": "SimpleGNN2D"}})
    ]
    results_with = [
        ("mesh", {"y_true_unscaled": y_true, "y_pred_unscaled": y_pred, "evaluation_info": {"model_type": "MeshGNN2D"}})
    ]

    df_no, df_with = pu.build_mae_summary_dataframes_from_results(
        results_no,
        results_with,
        pressure_levels,
        strat_max_pressure_hpa=500.0,
    )

    assert "gnn" in df_no.index
    assert "mesh" in df_with.index


def test_plot_mae_pressure_profiles_all_accepts_variable_selection_and_legend_map():
    import matplotlib.pyplot as plt

    y_true, y_pred = _make_graph_result()
    pressure_levels = np.asarray([1000.0, 500.0], dtype=float)
    results = [
        (
            "raw-cnn",
            {
                "y_true_unscaled": y_true,
                "y_pred_unscaled": y_pred,
                "evaluation_info": {"model_type": "SimpleGNN2D"},
            },
        )
    ]

    fig, axes, payload = pu.plot_mae_pressure_profiles_from_results_all(
        results,
        results,
        pressure_levels,
        variables=("temp", "ucomp"),
        legend_label_map={"raw-cnn": "CNN", "Persistence": "Previous step"},
    )

    assert axes.shape == (2, 2)
    assert tuple(payload["meta"]["variables"]) == ("temp", "ucomp")
    assert set(payload["by_variable"]) == {"temp", "ucomp"}
    assert axes[0, 0].yaxis_inverted()
    legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert legend_labels == ["CNN", "Previous step"]

    plt.close(fig)


def test_plot_mae_pressure_latitude_grid_matches_mesh_names():
    import matplotlib.pyplot as plt

    y_true, y_pred = _make_graph_result()
    pressure_levels = np.asarray([1000.0, 500.0], dtype=float)
    lat_levels = np.asarray([-45.0, 45.0], dtype=float)
    results = [
        (
            "mesh-row",
            {
                "y_true_unscaled": y_true,
                "y_pred_unscaled": y_pred,
                "evaluation_info": {"model_type": "MeshGNN2D"},
            },
        )
    ]

    fig_a, axes_a, payload_a = pu.plot_mae_pressure_latitude_grid_from_results(
        results,
        results,
        pressure_levels,
        lat_levels,
        architecture_names=("MeshGNN2D",),
    )
    assert payload_a["rows_no_ssw"][0][1] is not None
    plt.close(fig_a)

    fig_b, axes_b, payload_b = pu.plot_mae_pressure_latitude_grid_from_results(
        results,
        results,
        pressure_levels,
        lat_levels,
        architecture_names=("mesh_gnn_gridtomesh_mesh_and_meshtogrid",),
    )
    assert payload_b["rows_no_ssw"][0][1] is not None
    plt.close(fig_b)
