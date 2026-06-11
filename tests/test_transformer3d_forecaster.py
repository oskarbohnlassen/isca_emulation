import torch
from omegaconf import OmegaConf

from isca_emulation_v2.models.model import load_model
from isca_emulation_v2.models.transformer import (
    Transformer3DGlobalAttentionForecaster,
    Transformer3DSwinForecaster,
)


def test_transformer3d_forecaster_forward_shape_and_residual_contract():
    model = Transformer3DGlobalAttentionForecaster(
        channels=6,
        out_channels=3,
        grid_depth=12,
        grid_height=8,
        grid_width=16,
        hidden_dim=24,
        patch_size=(3, 4, 4),
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
    )
    x = torch.randn(2, 6, 12, 8, 16)

    y = model(x)

    assert y.shape == (2, 3, 12, 8, 16)
    assert torch.isfinite(y).all()


def test_load_model_builds_transformer3d_forecaster_from_cfg():
    cfg = OmegaConf.create(
        {
            "model": {
                "model_type": "Transformer3DGlobalAttentionForecaster",
                "hidden_dim": 24,
                "patch_depth": 3,
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
                "in_channels": 6,
                "out_channels": 3,
                "grid_depth": 12,
                "grid_height": 8,
                "grid_width": 16,
            },
        }
    )

    model = load_model(cfg)

    assert isinstance(model, Transformer3DGlobalAttentionForecaster)


def test_transformer3d_forecaster_rejects_incompatible_patch_size():
    try:
        Transformer3DGlobalAttentionForecaster(
            channels=4,
            out_channels=2,
            grid_depth=10,
            grid_height=8,
            grid_width=16,
            hidden_dim=24,
            patch_size=(3, 4, 4),
            num_layers=2,
            num_heads=4,
        )
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("Expected ValueError for incompatible patch size.")


def test_transformer3d_tokenizer_and_detokenizer_use_structured_patch_grids():
    model = Transformer3DGlobalAttentionForecaster(
        channels=6,
        out_channels=3,
        grid_depth=12,
        grid_height=8,
        grid_width=16,
        hidden_dim=24,
        patch_size=(3, 4, 4),
        num_layers=2,
        num_heads=4,
    )
    x = torch.randn(2, 6, 12, 8, 16)

    token_grid = model.tokenizer(x)
    assert token_grid.shape == (2, 4, 2, 4, 24)

    token_sequence = model._flatten_tokens(token_grid)
    assert token_sequence.shape == (2, 32, 24)

    restored_grid = model._unflatten_tokens(token_sequence)
    assert restored_grid.shape == token_grid.shape

    y = model.detokenizer(restored_grid)
    assert y.shape == (2, 3, 12, 8, 16)


def test_transformer3d_swin_forecaster_forward_shape_and_residual_contract():
    model = Transformer3DSwinForecaster(
        channels=6,
        out_channels=3,
        grid_depth=12,
        grid_height=8,
        grid_width=16,
        hidden_dim=24,
        patch_size=(3, 4, 4),
        window_size=(2, 2, 2),
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
    )
    x = torch.randn(2, 6, 12, 8, 16)

    y = model(x)

    assert y.shape == (2, 3, 12, 8, 16)
    assert torch.isfinite(y).all()


def test_load_model_builds_transformer3d_swin_forecaster_from_cfg():
    cfg = OmegaConf.create(
        {
            "model": {
                "model_type": "Transformer3DSwinForecaster",
                "hidden_dim": 24,
                "patch_depth": 3,
                "patch_height": 4,
                "patch_width": 4,
                "window_depth": 2,
                "window_height": 2,
                "window_width": 2,
                "num_layers": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "activation": "gelu",
                "dropout": 0.1,
                "attention_dropout": 0.0,
            },
            "data": {
                "in_channels": 6,
                "out_channels": 3,
                "grid_depth": 12,
                "grid_height": 8,
                "grid_width": 16,
            },
        }
    )

    model = load_model(cfg)

    assert isinstance(model, Transformer3DSwinForecaster)
