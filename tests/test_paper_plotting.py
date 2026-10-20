from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from paper import plotting as pu


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


def test_physical_mae_and_persistence_in_graph_layout():
    y_true, y_pred = _make_graph_result()
    for variable in range(3):
        mae = pu._mae_map(y_true, y_pred, variable, n_levels=2, n_lats=2)
        persistence = pu._persistence_map_from_y_true(y_true, variable, n_levels=2, n_lats=2)
        np.testing.assert_array_equal(mae, np.ones((2, 2)))
        np.testing.assert_array_equal(persistence, np.full((2, 2), 36.0))
