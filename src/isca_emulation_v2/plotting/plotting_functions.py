import os
import textwrap
from typing import Sequence, TypedDict
import logging
import numpy as np
import torch


def error_min_max_from_percentile(error_values: np.ndarray, error_percentile: float) -> tuple[float, float]:
    """Return symmetric error limits around zero using percentile clipping."""
    q = float(error_percentile)
    if not (0 < q <= 1):
        raise ValueError(f"error_percentile must be in (0, 1], got {q}")

    values = np.asarray(error_values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1e-12, 1e-12)

    if q == 1.0:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    else:
        lo_pct = (1.0 - q) * 100.0
        hi_pct = q * 100.0
        lo = float(np.nanpercentile(values, lo_pct))
        hi = float(np.nanpercentile(values, hi_pct))
        if lo > hi:
            lo, hi = hi, lo

    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))

    abs_lim = max(abs(lo), abs(hi))
    if not np.isfinite(abs_lim) or abs_lim == 0:
        abs_lim = 1e-12

    return -abs_lim, abs_lim


def _serve_panel_app(app) -> None:
    import panel as pn

    # Bokeh may emit noisy patch-drop warnings during rapid model replacement.
    # They are generally harmless for this app's update pattern.
    class _DropPatchWarningFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "Dropping a patch because it contains a previously known reference" not in record.getMessage()

    root_logger = logging.getLogger()
    if not getattr(root_logger, "_isca_drop_patch_filter_installed", False):
        root_logger.addFilter(_DropPatchWarningFilter())
        root_logger._isca_drop_patch_filter_installed = True

    port_env = os.environ.get("ISCA_PLOT_PORT")
    port = int(port_env) if port_env else 0

    ws_env = os.environ.get("BOKEH_ALLOW_WS_ORIGIN")
    if ws_env:
        websocket_origin = [item.strip() for item in ws_env.split(",") if item.strip()]
    else:
        websocket_origin = "*"

    pn.serve(
        app,
        port=port,
        address="127.0.0.1",
        websocket_origin=websocket_origin,
        show=False,
    )

def plot_isca_result(
    ds0,
    times: np.ndarray,
    pvals: np.ndarray,
    var_keys: list[str],
) -> None:
    import holoviews as hv
    import hvplot.xarray  # noqa: F401
    import panel as pn
    import xarray as xr

    clim_cache: dict[str, tuple["xr.DataArray", "xr.DataArray"]] = {}

    def get_da(var: str):
        if var == "wind_speed":
            u = ds0["ucomp"]
            v = ds0["vcomp"]
            return np.sqrt(u**2 + v**2).rename("wind_speed")
        return ds0[var]

    def ensure_clim(var: str) -> None:
        if var in clim_cache:
            return
        da = get_da(var)
        da_c = da.chunk({"time": -1}) if hasattr(da.data, "chunks") and "time" in da.dims else da
        mins = da_c.min(dim=("time", "lat", "lon")).compute() if hasattr(da_c.data, "compute") else da_c.min(dim=("time", "lat", "lon"))
        maxs = da_c.max(dim=("time", "lat", "lon")).compute() if hasattr(da_c.data, "compute") else da_c.max(dim=("time", "lat", "lon"))
        clim_cache[var] = (mins, maxs)

    ensure_clim(var_keys[0])

    def make_plot(var: str, time_idx: int, p_target: float):
        ensure_clim(var)
        mins, maxs = clim_cache[var]
        da = get_da(var).isel(time=time_idx).sel(pfull=p_target, method="nearest")
        p_used = float(da["pfull"].values)

        vmin = float(mins.sel(pfull=p_used, method="nearest").values)
        vmax = float(maxs.sel(pfull=p_used, method="nearest").values)

        return da.hvplot.quadmesh(
            x="lon",
            y="lat",
            cmap="coolwarm",
            clim=(vmin, vmax),
            colorbar=True,
            height=350,
            width=700,
            title=f"{var} | time={times[time_idx]} | pfull≈{p_used:.1f} | fixed per-level clim",
        )

    pn.extension()
    hv.extension("bokeh")

    var_w = pn.widgets.Select(name="variable", options=var_keys, value=var_keys[0])
    time_w = pn.widgets.IntSlider(name="time index", start=0, end=len(times) - 1, value=0)

    p_unique = np.sort(np.unique(pvals))
    p_step = float(np.min(np.abs(np.diff(p_unique)))) if len(p_unique) > 1 else 1.0
    p_start = float(pvals[np.argmin(np.abs(pvals - 1000.0))])

    p_w = pn.widgets.FloatSlider(
        name="pfull",
        start=float(pvals.min()),
        end=float(pvals.max()),
        step=p_step,
        value=p_start,
    )

    stream = hv.streams.Stream.define("State", var=str, time_idx=int, p_target=float)(
        var=var_w.value,
        time_idx=time_w.value,
        p_target=float(p_w.value),
    )
    dmap = hv.DynamicMap(
        lambda var, time_idx, p_target: make_plot(var, time_idx, p_target),
        streams=[stream],
    )
    zonal_mean_text = pn.pane.Markdown("", sizing_mode="stretch_width")

    def _format_zonal_mean_text(time_idx: int, p_target: float) -> str:
        da_u = ds0["ucomp"]
        if "time" in da_u.dims:
            da_u = da_u.isel(time=time_idx)
        if "pfull" in da_u.dims:
            da_u = da_u.sel(pfull=p_target, method="nearest")
            p_used = float(da_u["pfull"].values)
        else:
            p_used = None
        da_u = da_u.sel(lat=60.0, method="nearest")
        lat_used = float(da_u["lat"].values)
        u_mean = da_u.mean(dim="lon") if "lon" in da_u.dims else da_u
        u_val = float(u_mean.values)
        if p_used is None:
            return f"Mean zonal wind (`ucomp`) at lat≈{lat_used:.1f}: **{u_val:.3f} m/s**"
        return (
            f"Mean zonal wind (`ucomp`) at lat≈{lat_used:.1f}, "
            f"pfull≈{p_used:.1f}: **{u_val:.3f} m/s**"
        )

    def sync(_=None) -> None:
        stream.event(var=var_w.value, time_idx=time_w.value, p_target=float(p_w.value))
        zonal_mean_text.object = _format_zonal_mean_text(time_w.value, float(p_w.value))

    var_w.param.watch(sync, "value")
    time_w.param.watch(sync, "value_throttled")
    p_w.param.watch(sync, "value_throttled")

    sync()
    app = pn.Column(pn.Row(var_w, time_w, p_w), zonal_mean_text, dmap)
    _serve_panel_app(app)



def plot_isca_emulator_result(
    test_in: torch.Tensor | np.ndarray,
    test_true: torch.Tensor | np.ndarray,
    test_pred: torch.Tensor | np.ndarray,
    *,
    lat: Sequence[float],
    lon: Sequence[float],
    channel_index: list[tuple[str, float]],
    units_by_var: dict[str, str] | None = None,
) -> None:
    import holoviews as hv
    import hvplot.xarray  # noqa: F401
    import panel as pn

    pn.extension()
    hv.extension("bokeh")

    def _cpu_tensor(t: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(t, torch.Tensor):
            return t.detach().cpu()
        if isinstance(t, np.ndarray):
            return torch.from_numpy(t)
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(t)!r}")

    test_in = _cpu_tensor(test_in)
    test_true = _cpu_tensor(test_true)
    test_pred = _cpu_tensor(test_pred)

    if test_true.ndim not in {4, 5}:
        raise ValueError(f"Expected test_true to be rank 4 or 5, got shape {tuple(test_true.shape)}")
    if test_in.ndim != test_true.ndim or test_pred.ndim != test_true.ndim:
        raise ValueError(
            "Input/target/prediction tensors must have matching rank: "
            f"got test_in={tuple(test_in.shape)}, test_true={tuple(test_true.shape)}, test_pred={tuple(test_pred.shape)}"
        )

    n_samples = int(test_true.shape[0])
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    units_by_var = units_by_var or {}

    var_keys: list[str] = []
    for var_name, _ in channel_index:
        if var_name not in var_keys:
            var_keys.append(var_name)

    pvals = np.array([float(level) for _, level in channel_index], dtype=float)
    p_unique = np.sort(np.unique(pvals))
    var_w = pn.widgets.Select(name="variable", options=var_keys, value=var_keys[0])
    idx_w = pn.widgets.IntSlider(name="sample_idx", start=0, end=n_samples - 1, value=0)

    p_step = float(np.min(np.abs(np.diff(p_unique)))) if len(p_unique) > 1 else 1.0
    p_start = float(p_unique[np.argmin(np.abs(p_unique - 1000.0))])
    p_w = pn.widgets.FloatSlider(
        name="pfull",
        start=float(p_unique.min()),
        end=float(p_unique.max()),
        step=p_step,
        value=p_start,
    )

    is_cnn3d = test_true.ndim == 5
    if is_cnn3d:
        _n, n_channels, n_levels, _h, _w = test_true.shape
        if len(var_keys) != int(n_channels):
            raise ValueError(
                "CNN3D plotting expects unique variables to match channel axis: "
                f"len(var_keys)={len(var_keys)} vs channels={int(n_channels)}"
            )
        if len(p_unique) != int(n_levels):
            raise ValueError(
                "CNN3D plotting expects pfull values to match level axis: "
                f"len(unique pfull)={len(p_unique)} vs levels={int(n_levels)}"
            )

        var_to_channel = {var_name: idx for idx, var_name in enumerate(var_keys)}
        clim_min = test_true.amin(dim=(0, 3, 4)).numpy()
        clim_max = test_true.amax(dim=(0, 3, 4)).numpy()
        err_abs_max = (test_pred - test_true).abs().amax(dim=(0, 3, 4)).numpy()
        persistence_err_abs_max = (test_in[:, : test_true.shape[1]] - test_true).abs().amax(dim=(0, 3, 4)).numpy()
    else:
        _n, n_channels, _h, _w = test_true.shape
        if len(channel_index) != int(n_channels):
            raise ValueError(
                "CNN2D plotting expects channel_index to match channel axis: "
                f"len(channel_index)={len(channel_index)} vs channels={int(n_channels)}"
            )

        def channel_from_var_p(var: str, p_target: float) -> tuple[int, float]:
            idxs = [i for i, (v, _) in enumerate(channel_index) if v == var]
            if not idxs:
                raise ValueError(f"Variable '{var}' not found in channel index.")
            p_arr = np.array([float(channel_index[i][1]) for i in idxs], dtype=float)
            j = int(np.argmin(np.abs(p_arr - float(p_target))))
            return idxs[j], float(p_arr[j])

        clim_min = test_true.amin(dim=(0, 2, 3)).numpy()
        clim_max = test_true.amax(dim=(0, 2, 3)).numpy()
        err_abs_max = (test_pred - test_true).abs().amax(dim=(0, 2, 3)).numpy()
        persistence_err_abs_max = (test_in[:, : test_true.shape[1]] - test_true).abs().amax(dim=(0, 2, 3)).numpy()

    def da2d(arr2d: np.ndarray, name: str):
        import xarray as xr

        return xr.DataArray(
            arr2d,
            coords={"lat": lat, "lon": lon},
            dims=("lat", "lon"),
            name=name,
        )

    def quad(da, title: str, clim=None):
        return da.hvplot.quadmesh(
            x="lon",
            y="lat",
            cmap = "coolwarm",
            colorbar=True,
            height=320,
            width=350,
            clim=clim,
            title=title,
        )

    def make_panel(var: str, sample_idx: int, p_target: float):
        if is_cnn3d:
            c = var_to_channel[var]
            level_idx = int(np.argmin(np.abs(p_unique - float(p_target))))
            p_used = float(p_unique[level_idx])
            xin = test_in[sample_idx, c, level_idx].numpy()
            ytru = test_true[sample_idx, c, level_idx].numpy()
            ypre = test_pred[sample_idx, c, level_idx].numpy()
        else:
            c, p_used = channel_from_var_p(var, p_target)
            xin = test_in[sample_idx, c].numpy()
            ytru = test_true[sample_idx, c].numpy()
            ypre = test_pred[sample_idx, c].numpy()
        err = ypre - ytru
        persistence_err = xin - ytru
        mae_pred_err = float(np.mean(np.abs(err)))
        mae_persistence_err = float(np.mean(np.abs(persistence_err)))

        units = units_by_var.get(var, "")
        if is_cnn3d:
            vmin = float(clim_min[c, level_idx])
            vmax = float(clim_max[c, level_idx])
        else:
            vmin = float(clim_min[c])
            vmax = float(clim_max[c])
        clim = (vmin, vmax)

        if is_cnn3d:
            err_max = max(float(err_abs_max[c, level_idx]), float(persistence_err_abs_max[c, level_idx]))
        else:
            err_max = max(float(err_abs_max[c]), float(persistence_err_abs_max[c]))
        err_clim = (-err_max, err_max) if np.isfinite(err_max) and err_max > 0 else None

        p0 = quad(da2d(xin, "input"), f"Input (t) | p≈{p_used:.4g} {units}", clim=clim)
        p1 = quad(da2d(ytru, "truth"), "Truth (t+1)", clim=clim)
        p2 = quad(da2d(ypre, "pred"), "Prediction (t+1)", clim=clim)
        p3 = quad(
            da2d(err, "err"),
            f"Error (pred-truth) | MAE={mae_pred_err:.4g} {units}",
            clim=err_clim,
        )
        p4 = quad(
            da2d(persistence_err, "persistence_err"),
            f"Error (persistence-truth) | MAE={mae_persistence_err:.4g} {units}",
            clim=err_clim,
        )
        blank = hv.Curve([]).opts(width=350, height=320, xaxis=None, yaxis=None, show_frame=False, toolbar=None)
        return (p0 + p1 + p2 + p3 + blank + blank + blank + p4).cols(4)

    stream = hv.streams.Stream.define("State", var=str, sample_idx=int, p_target=float)(
        var=var_w.value,
        sample_idx=idx_w.value,
        p_target=float(p_w.value),
    )
    dmap = hv.DynamicMap(
        lambda var, sample_idx, p_target: make_panel(var, sample_idx, p_target),
        streams=[stream],
    )

    def sync(_=None) -> None:
        stream.event(var=var_w.value, sample_idx=idx_w.value, p_target=float(p_w.value))

    var_w.param.watch(sync, "value")
    idx_w.param.watch(sync, "value_throttled")
    p_w.param.watch(sync, "value_throttled")

    sync()
    app = pn.Column(pn.Row(var_w, idx_w, p_w), dmap)
    _serve_panel_app(app)


def plot_isca_emulator_result_multi(
    model_rows: list[
        tuple[
            str,
            torch.Tensor | np.ndarray,
            torch.Tensor | np.ndarray,
            torch.Tensor | np.ndarray,
        ]
    ],
    *,
    lat: Sequence[float],
    lon: Sequence[float],
    channel_index: list[tuple[str, float]],
    units_by_var: dict[str, str] | None = None,
    error_percentile: float = 1.0,
) -> None:
    import importlib.util
    import holoviews as hv
    import hvplot.xarray  # noqa: F401
    import panel as pn

    pn.extension()
    hv.extension("bokeh")

    if len(model_rows) == 0:
        raise ValueError("model_rows is empty.")

    def _cpu_tensor(t: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(t, torch.Tensor):
            return t.detach().cpu()
        if isinstance(t, np.ndarray):
            return torch.from_numpy(t)
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(t)!r}")

    class _ModelRow(TypedDict):
        name: str
        x_in: np.ndarray
        y_true: np.ndarray
        y_pred: np.ndarray

    rows: list[_ModelRow] = []
    for model_name, row_in, row_true, row_pred in model_rows:
        x_in = _cpu_tensor(row_in)
        y_true = _cpu_tensor(row_true)
        y_pred = _cpu_tensor(row_pred)
        x_in_np = x_in.numpy()
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()

        rows.append({"name": str(model_name), "x_in": x_in_np, "y_true": y_true_np, "y_pred": y_pred_np})

    reference_true = rows[0]["y_true"]
    n_samples = min(int(row["y_true"].shape[0]) for row in rows)
    is_cnn3d = reference_true.ndim == 5

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    units_by_var = units_by_var or {}

    var_keys: list[str] = []
    for var_name, _ in channel_index:
        if var_name not in var_keys:
            var_keys.append(var_name)

    pvals = np.array([float(level) for _, level in channel_index], dtype=float)
    p_unique = np.sort(np.unique(pvals))
    var_w = pn.widgets.Select(name="variable", options=var_keys, value=var_keys[0])
    idx_w = pn.widgets.IntSlider(name="sample_idx", start=0, end=n_samples - 1, value=0)

    p_start = float(p_unique[np.argmin(np.abs(p_unique - 1000.0))])
    p_w = pn.widgets.Select(
        name="pfull",
        options=[float(x) for x in p_unique],
        value=float(p_start),
    )

    if is_cnn3d:
        _n, n_channels, n_levels, _h, _w = reference_true.shape
        if len(var_keys) != int(n_channels):
            raise ValueError(
                "CNN3D plotting expects unique variables to match channel axis: "
                f"len(var_keys)={len(var_keys)} vs channels={int(n_channels)}"
            )
        if len(p_unique) != int(n_levels):
            raise ValueError(
                "CNN3D plotting expects pfull values to match level axis: "
                f"len(unique pfull)={len(p_unique)} vs levels={int(n_levels)}"
            )

        var_to_channel = {var_name: idx for idx, var_name in enumerate(var_keys)}
    else:
        _n, n_channels, _h, _w = reference_true.shape
        if len(channel_index) != int(n_channels):
            raise ValueError(
                "CNN2D plotting expects channel_index to match channel axis: "
                f"len(channel_index)={len(channel_index)} vs channels={int(n_channels)}"
            )

        def channel_from_var_p(var: str, p_target: float) -> tuple[int, float]:
            idxs = [i for i, (v, _) in enumerate(channel_index) if v == var]
            if not idxs:
                raise ValueError(f"Variable '{var}' not found in channel index.")
            p_arr = np.array([float(channel_index[i][1]) for i in idxs], dtype=float)
            j = int(np.argmin(np.abs(p_arr - float(p_target))))
            return idxs[j], float(p_arr[j])

    # SSW-style diagnostic: zonal-mean zonal wind at 60N near 10 hPa.
    u60_target_lat = 60.0
    u60_target_pfull = 10.0
    lat_idx_u60 = int(np.argmin(np.abs(lat - u60_target_lat)))
    lat_used_u60 = float(lat[lat_idx_u60])
    u_var = "ucomp"
    u_units = units_by_var.get(u_var, "m/s") or "m/s"
    has_ucomp = u_var in var_keys

    if has_ucomp:
        if is_cnn3d:
            u_channel_idx = var_to_channel[u_var]
            u_level_idx = int(np.argmin(np.abs(p_unique - u60_target_pfull)))
            u_p_used = float(p_unique[u_level_idx])
        else:
            u_channel_idx, u_p_used = channel_from_var_p(u_var, u60_target_pfull)
            u_level_idx = None
    else:
        u_channel_idx = None
        u_level_idx = None
        u_p_used = float("nan")

    # Precompute one color scale per (variable, pfull) pair.
    min_max_dict: dict[tuple[str, float], tuple[float, float]] = {}
    err_min_max_dict: dict[tuple[str, float], tuple[float, float]] = {}

    for var in var_keys:
        for p in p_unique:
            if is_cnn3d:
                c = var_to_channel[var]
                level_idx = int(np.argmin(np.abs(p_unique - float(p))))
                p_used = float(p_unique[level_idx])
                value_arrays = []
                error_arrays = []
                for row in rows:
                    x_arr = row["x_in"][:, c, level_idx]
                    y_arr = row["y_true"][:, c, level_idx]
                    y_pred_arr = row["y_pred"][:, c, level_idx]
                    value_arrays.extend([x_arr, y_arr, y_pred_arr])
                    error_arrays.extend([x_arr - y_arr, y_pred_arr - y_arr])
            else:
                c, p_used = channel_from_var_p(var, float(p))
                value_arrays = []
                error_arrays = []
                for row in rows:
                    x_arr = row["x_in"][:, c]
                    y_arr = row["y_true"][:, c]
                    y_pred_arr = row["y_pred"][:, c]
                    value_arrays.extend([x_arr, y_arr, y_pred_arr])
                    error_arrays.extend([x_arr - y_arr, y_pred_arr - y_arr])

            value_stack = np.stack(value_arrays, axis=0)
            error_stack = np.stack(error_arrays, axis=0)
            key = (var, float(p_used))
            min_max_dict[key] = (float(np.nanmin(value_stack)), float(np.nanmax(value_stack)))
            err_min_max_dict[key] = error_min_max_from_percentile(error_stack, error_percentile)

    coords = {"lat": lat, "lon": lon}
    _has_datashader = importlib.util.find_spec("datashader") is not None

    def da2d(arr2d: np.ndarray, name: str):
        import xarray as xr

        return xr.DataArray(
            arr2d,
            coords=coords,
            dims=("lat", "lon"),
            name=name,
        )

    def quad(da, title: str, clim=None):
        wrapped_title = textwrap.fill(
            title,
            width=52,
            break_long_words=False,
            break_on_hyphens=False,
        )
        quadmesh_kwargs = {
            "x": "lon",
            "y": "lat",
            "cmap": "coolwarm",
            "colorbar": True,
            "height": 280,
            "width": 500,
            "clim": clim,
            "title": wrapped_title,
        }
        if _has_datashader:
            quadmesh_kwargs["rasterize"] = True
        return da.hvplot.quadmesh(**quadmesh_kwargs)

    def _extract_row_arrays(
        row: _ModelRow,
        sample_idx: int,
        channel_idx: int,
        level_idx: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if is_cnn3d:
            if level_idx is None:
                raise ValueError("level_idx must be provided for CNN3D extraction.")
            xin_row = row["x_in"][sample_idx, channel_idx, level_idx]
            ytru_row = row["y_true"][sample_idx, channel_idx, level_idx]
            ypre_row = row["y_pred"][sample_idx, channel_idx, level_idx]
            return xin_row, ytru_row, ypre_row
        xin_row = row["x_in"][sample_idx, channel_idx]
        ytru_row = row["y_true"][sample_idx, channel_idx]
        ypre_row = row["y_pred"][sample_idx, channel_idx]
        return xin_row, ytru_row, ypre_row

    def _extract_u_field(arr: np.ndarray, sample_idx: int) -> np.ndarray:
        if u_channel_idx is None:
            raise ValueError("ucomp channel is not available.")
        if is_cnn3d:
            if u_level_idx is None:
                raise ValueError("u_level_idx must be provided for CNN3D extraction.")
            return arr[sample_idx, u_channel_idx, u_level_idx]
        return arr[sample_idx, u_channel_idx]

    def _u60_stats(arr2d: np.ndarray) -> tuple[float, float, float]:
        zonal_slice = arr2d[lat_idx_u60, :]
        return (
            float(np.nanmean(zonal_slice)),
            float(np.nanmin(zonal_slice)),
            float(np.nanmax(zonal_slice)),
        )

    def _format_u60_text(sample_idx: int) -> str:
        if not has_ucomp:
            return "U60N zonal-mean wind unavailable: `ucomp` is missing from the channel mapping."

        sample_idx = int(sample_idx)
        baseline_row = rows[0]
        input_mean, input_min, input_max = _u60_stats(_extract_u_field(baseline_row["x_in"], sample_idx))

        lines = [
            (
                f"**U60N zonal wind (`ucomp`)** | sample={sample_idx} | "
                f"lat≈{lat_used_u60:.1f} | pfull≈{u_p_used:.4g}:"
            ),
            "",
            (
                f"- Input (t): mean: **{input_mean:.3f} {u_units}**, "
                f"min: **{input_min:.3f} {u_units}**, max: **{input_max:.3f} {u_units}**"
            ),
        ]

        return "\n".join(lines)

    def make_panel(var: str, sample_idx: int, p_target: float, nonce: int = 0):
        baseline_row = rows[0]

        if is_cnn3d:
            c = var_to_channel[var]
            level_idx = int(np.argmin(np.abs(p_unique - float(p_target))))
            p_used = float(p_unique[level_idx])
        else:
            c, p_used = channel_from_var_p(var, p_target)
            level_idx = None

        xin, ytru, _ = _extract_row_arrays(baseline_row, sample_idx, c, level_idx)
        persistence_err = xin - ytru
        row_fields: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for row in rows:
            xin_row, ytru_row, ypre_row = _extract_row_arrays(row, sample_idx, c, level_idx)
            err_row = ypre_row - ytru_row
            row_fields.append((row["name"], xin_row, ypre_row, err_row))

        key = (var, float(p_used))
        if key not in min_max_dict:
            raise KeyError(f"Missing state color limits for {key}")
        if key not in err_min_max_dict:
            raise KeyError(f"Missing error color limits for {key}")

        vmin, vmax = min_max_dict[key]
        err_min, err_max = err_min_max_dict[key]
        clim = (vmin, vmax)
        err_clim = (err_min, err_max)
        units = units_by_var.get(var, "")
        frame_key = f"var={var}|p={float(p_used):.6g}|sample={sample_idx}|nonce={nonce}"

        def value_plot(arr2d: np.ndarray, title: str, panel_role: str, model_tag: str):
            return (
                quad(da2d(arr2d, "field"), title, clim=clim)
                .redim.range(field=clim)
                .opts(framewise=True)
                .relabel(f"{panel_role}|{model_tag}|{frame_key}")
            )

        def error_plot(arr2d: np.ndarray, title: str, panel_role: str, model_tag: str):
            return (
                quad(da2d(arr2d, "error"), title, clim=err_clim)
                .redim.range(error=err_clim)
                .opts(framewise=True)
                .relabel(f"{panel_role}|{model_tag}|{frame_key}")
            )

        plots = []
        persistence_mae = float(np.mean(np.abs(persistence_err)))
        persistence_title = f"Persistence (t - (t+1)) | MAE={persistence_mae:.4g} {units}".strip()
        plots.append(value_plot(xin, "Input (t)", "input", "baseline"))
        plots.append(value_plot(ytru, "Truth (t+1)", "truth", "baseline"))
        plots.append(error_plot(persistence_err, persistence_title, "persistence_error", "baseline"))

        for row_idx, (model_name, xin_row, ypre_row, err_row) in enumerate(row_fields):
            mae_pred_err = float(np.mean(np.abs(err_row)))
            model_tag = f"{model_name}|row={row_idx}"
            plots.append(value_plot(xin_row, f"{model_name} | Input (t)", "input", model_tag))
            plots.append(value_plot(ypre_row, f"{model_name} | Pred (t+1)", "pred", model_tag))
            plots.append(
                error_plot(
                    err_row,
                    f"{model_name} | Error (pred-truth) MAE={mae_pred_err:.4g}",
                    "pred_error",
                    model_tag,
                )
            )

        return hv.Layout(plots).cols(3).opts(framewise=True)

    render_counter = {"v": 0}

    def render_panel(var: str, sample_idx: int, p_target: float):
        render_counter["v"] += 1
        return make_panel(var, int(sample_idx), float(p_target), nonce=render_counter["v"])

    bound_layout = pn.bind(
        render_panel,
        var=var_w.param.value,
        sample_idx=idx_w.param.value_throttled,
        p_target=p_w.param.value,
    )

    u60_text = pn.pane.Markdown("", sizing_mode="stretch_width")

    def sync_u60_text(_=None) -> None:
        u60_text.object = _format_u60_text(int(idx_w.value))

    idx_w.param.watch(sync_u60_text, "value_throttled")
    sync_u60_text()

    app = pn.Column(pn.Row(var_w, idx_w, p_w), u60_text, bound_layout)
    _serve_panel_app(app)
