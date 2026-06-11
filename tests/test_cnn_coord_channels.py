import torch

from isca_emulation_v2.data.utils import build_cnn2d_coord_channels, build_cnn3d_coord_channels


def test_build_cnn2d_coord_channels_uses_radians_for_trig4():
    coords = build_cnn2d_coord_channels(
        lonlat_values={"lat": [0.0, 90.0], "lon": [0.0, 180.0]},
        add_coords="trig4",
    )

    assert coords is not None
    assert tuple(coords.shape) == (4, 2, 2)

    expected_sin_lat = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    expected_cos_lat = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    expected_sin_lon = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    expected_cos_lon = torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float32)

    assert torch.allclose(coords[0], expected_sin_lat, atol=1e-6)
    assert torch.allclose(coords[1], expected_cos_lat, atol=1e-6)
    assert torch.allclose(coords[2], expected_sin_lon, atol=1e-6)
    assert torch.allclose(coords[3], expected_cos_lon, atol=1e-6)


def test_build_cnn3d_coord_channels_level_features_are_monotonic_and_bounded():
    coords = build_cnn3d_coord_channels(
        level_lonlat_values={
            "level": [1.0, 10.0, 100.0],
            "lat": [0.0, 45.0],
            "lon": [0.0, 90.0],
        },
        add_coords="trig4",
    )

    assert coords is not None
    assert tuple(coords.shape) == (6, 3, 2, 2)

    # Check that angular features are in radians-based trig space.
    expected_lat = torch.tensor(
        [[0.0, 0.0], [torch.sin(torch.deg2rad(torch.tensor(45.0))), torch.sin(torch.deg2rad(torch.tensor(45.0)))]],
        dtype=torch.float32,
    )
    expected_lon_cos = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(coords[0, 0], expected_lat, atol=1e-6)
    assert torch.allclose(coords[3, 0], expected_lon_cos, atol=1e-6)

    # Last two channels are normalized pressure features and should be monotonic across levels.
    level_norm = coords[4, :, 0, 0]
    level_log_norm = coords[5, :, 0, 0]

    assert torch.all(level_norm[1:] > level_norm[:-1])
    assert torch.all(level_log_norm[1:] > level_log_norm[:-1])
    assert 0.0 <= float(level_norm.min()) <= float(level_norm.max()) <= 1.0
    assert 0.0 <= float(level_log_norm.min()) <= float(level_log_norm.max()) <= 1.0
    assert torch.allclose(level_norm, torch.tensor([0.0, 0.09090909, 1.0]), atol=1e-6)
    assert torch.allclose(level_log_norm, torch.tensor([0.0, 0.5, 1.0]), atol=1e-6)


def test_build_cnn3d_coord_channels_handles_nonpositive_levels_without_nans():
    coords = build_cnn3d_coord_channels(
        level_lonlat_values={
            "level": [0.0, 0.0, 0.0],
            "lat": [0.0],
            "lon": [0.0],
        },
        add_coords="trig2",
    )

    assert coords is not None
    assert tuple(coords.shape) == (4, 3, 1, 1)
    assert torch.isfinite(coords).all()
    assert torch.allclose(coords[2, :, 0, 0], torch.zeros(3), atol=1e-6)
    assert torch.allclose(coords[3, :, 0, 0], torch.zeros(3), atol=1e-6)
