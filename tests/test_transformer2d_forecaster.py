import torch
from omegaconf import OmegaConf

from isca_emulation_v2.models.model import load_model
from isca_emulation_v2.models.transformer import (
    Transformer2DGlobalAttentionForecaster,
    Transformer2DSwinForecaster,
)


def test_transformer2d_forecaster_forward_shape_and_residual_contract():
    model = Transformer2DGlobalAttentionForecaster(
        channels=10,
        out_channels=6,
        grid_height=16,
        grid_width=24,
        hidden_dim=32,
        patch_size=(4, 4),
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
    )
    x = torch.randn(3, 10, 16, 24)

    y = model(x)

    assert y.shape == (3, 6, 16, 24)
    assert torch.isfinite(y).all()


def test_load_model_builds_transformer2d_forecaster_from_cfg():
    cfg = OmegaConf.create(
        {
            "model": {
                "model_type": "Transformer2DGlobalAttentionForecaster",
                "hidden_dim": 32,
                "patch_height": 4,
                "patch_width": 4,
                "num_layers": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "activation": "gelu",
                "dropout": 0.1,
                "attention_dropout": 0.0,
            },
            "data": {
                "in_channels": 10,
                "out_channels": 6,
                "grid_height": 16,
                "grid_width": 24,
            },
        }
    )

    model = load_model(cfg)

    assert isinstance(model, Transformer2DGlobalAttentionForecaster)


def test_transformer2d_forecaster_rejects_incompatible_patch_size():
    try:
        Transformer2DGlobalAttentionForecaster(
            channels=8,
            out_channels=4,
            grid_height=10,
            grid_width=18,
            hidden_dim=32,
            patch_size=(4, 4),
            num_layers=2,
            num_heads=4,
        )
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("Expected ValueError for incompatible patch size.")


def test_transformer2d_tokenizer_and_detokenizer_use_structured_patch_grids():
    model = Transformer2DGlobalAttentionForecaster(
        channels=10,
        out_channels=6,
        grid_height=16,
        grid_width=24,
        hidden_dim=32,
        patch_size=(4, 4),
        num_layers=2,
        num_heads=4,
    )
    x = torch.randn(2, 10, 16, 24)

    token_grid = model.tokenizer(x)
    assert token_grid.shape == (2, 4, 6, 32)

    token_sequence = model._flatten_tokens(token_grid)
    assert token_sequence.shape == (2, 24, 32)

    restored_grid = model._unflatten_tokens(token_sequence)
    assert restored_grid.shape == token_grid.shape

    y = model.detokenizer(restored_grid)
    assert y.shape == (2, 6, 16, 24)


def test_transformer2d_swin_forecaster_forward_shape_and_residual_contract():
    model = Transformer2DSwinForecaster(
        channels=10,
        out_channels=6,
        grid_height=16,
        grid_width=24,
        hidden_dim=32,
        patch_size=(4, 4),
        window_size=(2, 3),
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
    )
    x = torch.randn(3, 10, 16, 24)

    y = model(x)

    assert y.shape == (3, 6, 16, 24)
    assert torch.isfinite(y).all()


def test_load_model_builds_transformer2d_swin_forecaster_from_cfg():
    cfg = OmegaConf.create(
        {
            "model": {
                "model_type": "Transformer2DSwinForecaster",
                "hidden_dim": 32,
                "patch_height": 4,
                "patch_width": 4,
                "window_height": 2,
                "window_width": 3,
                "num_layers": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "activation": "gelu",
                "dropout": 0.1,
                "attention_dropout": 0.0,
            },
            "data": {
                "in_channels": 10,
                "out_channels": 6,
                "grid_height": 16,
                "grid_width": 24,
            },
        }
    )

    model = load_model(cfg)

    assert isinstance(model, Transformer2DSwinForecaster)
