import numpy as np
import pytest


def _make_zero_wind_dataset():
    xr = pytest.importorskip("xarray")

    lat = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    pfull = np.array([10.0, 83.44])
    time = np.array([0.0, 1.0])
    shape = (len(time), len(pfull), len(lat), len(lon))

    ds = xr.Dataset(
        {
            "ucomp": (("time", "pfull", "lat", "lon"), np.zeros(shape)),
            "vcomp": (("time", "pfull", "lat", "lon"), np.zeros(shape)),
        },
        coords={"time": time, "pfull": pfull, "lat": lat, "lon": lon},
    )
    return ds, lat, lon


def test_compute_isca_absolute_vorticity_slice_zero_wind_matches_coriolis():
    pytest.importorskip("metpy")

    from isca_emulation_v2.plotting.plotting_functions import compute_isca_absolute_vorticity_slice

    ds, lat, lon = _make_zero_wind_dataset()

    avort = compute_isca_absolute_vorticity_slice(ds, time_index=1, pressure_hpa=80.0)

    earth_angular_velocity = 7.2921159e-5
    expected = 2.0 * earth_angular_velocity * np.sin(np.deg2rad(lat))
    expected = np.broadcast_to(expected[:, None], (len(lat), len(lon)))
    np.testing.assert_allclose(avort.to_numpy(), expected, atol=1e-10)
    assert avort.attrs["pressure_hpa"] == pytest.approx(83.44)
    assert avort.attrs["time_index"] == 1


def test_load_isca_zonal_mean_wind_series_uses_nearest_lat_pressure():
    from isca_emulation_v2.plotting.plotting_functions import _load_isca_zonal_mean_wind_series

    ds, _lat, _lon = _make_zero_wind_dataset()
    ds["ucomp"].loc[{"pfull": 10.0, "lat": 60.0}] = 3.0

    series, lat_used, p_used = _load_isca_zonal_mean_wind_series(
        ds,
        target_lat=59.0,
        target_pressure_hpa=11.0,
    )

    np.testing.assert_allclose(series, np.array([3.0, 3.0]))
    assert lat_used == pytest.approx(60.0)
    assert p_used == pytest.approx(10.0)


def test_plot_isca_absolute_vorticity_for_paper_uses_pressure_rows_and_case_columns(monkeypatch):
    pytest.importorskip("metpy")
    pytest.importorskip("cartopy")
    plt = pytest.importorskip("matplotlib.pyplot")

    from isca_emulation_v2.plotting.plotting_functions import plot_isca_absolute_vorticity_for_paper

    ds, _lat, _lon = _make_zero_wind_dataset()
    monkeypatch.setattr(plt, "show", lambda: None)

    fig, axes = plot_isca_absolute_vorticity_for_paper(
        ds,
        ds,
        time_index_no_ssw=0,
        time_indices_with_ssw=(0, 1),
        pressure_levels_hpa=(10.0, 80.0),
        min_lat=20.0,
    )

    assert axes.shape == (2, 3)
    assert axes[0, 0].get_title() == "a)"
    assert axes[0, 1].get_title() == "b)"
    assert axes[0, 2].get_title() == "c)"
    plt.close(fig)


def test_plot_isca_absolute_vorticity_for_paper_accepts_scalar_pressure(monkeypatch):
    pytest.importorskip("metpy")
    pytest.importorskip("cartopy")
    plt = pytest.importorskip("matplotlib.pyplot")

    from isca_emulation_v2.plotting.plotting_functions import plot_isca_absolute_vorticity_for_paper

    ds, _lat, _lon = _make_zero_wind_dataset()
    monkeypatch.setattr(plt, "show", lambda: None)

    fig, axes = plot_isca_absolute_vorticity_for_paper(
        ds,
        ds,
        time_index_no_ssw=0,
        time_indices_with_ssw=(0, 1),
        pressure_levels_hpa=10.0,
        min_lat=20.0,
    )

    assert axes.shape == (1, 3)
    plt.close(fig)
