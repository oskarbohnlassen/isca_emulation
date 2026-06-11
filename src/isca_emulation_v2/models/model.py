from omegaconf import OmegaConf

from isca_emulation_v2.models.cnn import (
    SimpleCNN2D,
    SimpleCNN3D,
    UnetCNN2D_Fixed,
    UnetCNN3D_Fixed,
)
from isca_emulation_v2.models.gnn import SimpleGNN2D
from isca_emulation_v2.models.mesh_gnn import MeshGNN2D
from isca_emulation_v2.models.transformer import (
    Transformer2DGlobalAttentionForecaster,
    Transformer2DSwinForecaster,
    Transformer3DGlobalAttentionForecaster,
    Transformer3DSwinForecaster,
)


def load_model(cfg) -> None:
    """Load a model architecture."""
    print("Model configuration:")
    print(OmegaConf.to_yaml(cfg.model))
    model_type = str(cfg.model.model_type)

    if model_type == "SimpleCNN2D":
        model = SimpleCNN2D(
            channels=int(cfg.data.in_channels),
            out_channels=int(cfg.data.out_channels),
            hidden=int(cfg.model.hidden_dim),
            kernel_size=int(cfg.model.kernel_size),
            padding_type=cfg.model.padding_type,
            latlon_padding=None,
            activation=str(cfg.model.activation),
            use_batch_norm=bool(cfg.model.use_batch_norm),
            num_layers=int(cfg.model.num_layers),
        )

    elif model_type == "SimpleCNN3D":
        model = SimpleCNN3D(
            channels=int(cfg.data.in_channels),
            out_channels=int(cfg.data.out_channels),
            hidden=int(cfg.model.hidden_dim),
            kernel_size=int(cfg.model.kernel_size),
            padding_type=cfg.model.padding_type,
            latlon_padding=None,
            activation=str(cfg.model.activation),
            use_batch_norm=bool(cfg.model.use_batch_norm),
            num_layers=int(cfg.model.num_layers),
        )

    elif model_type == "UnetCNN2D_Fixed":
        model = UnetCNN2D_Fixed(
            channels=int(cfg.data.in_channels),
            hidden_dim_base=int(cfg.model.hidden_dim_base),
            down_sample_method=str(cfg.model.down_sample_method),
            bottleneck_dilation=int(cfg.model.bottleneck_dilation),
            activation=str(cfg.model.activation),
            padding_type=str(cfg.model.padding_type),
            latlon_padding=None,
            use_batch_norm=bool(cfg.model.use_batch_norm),
            out_channels=int(cfg.data.out_channels),
        )

    elif model_type == "UnetCNN3D_Fixed":
        model = UnetCNN3D_Fixed(
            channels=int(cfg.data.in_channels),
            hidden_dim_base=int(cfg.model.hidden_dim_base),
            down_sample_method=str(cfg.model.down_sample_method),
            bottleneck_dilation=int(cfg.model.bottleneck_dilation),
            activation=str(cfg.model.activation),
            padding_type=str(cfg.model.padding_type),
            latlon_padding=None,
            use_batch_norm=bool(cfg.model.use_batch_norm),
            out_channels=int(cfg.data.out_channels),
        )

    elif model_type == "Transformer2DGlobalAttentionForecaster":
        patch_size_2d = (
            int(OmegaConf.select(cfg, "model.patch_height", default=OmegaConf.select(cfg, "model.patch_size", default=4))),
            int(OmegaConf.select(cfg, "model.patch_width", default=OmegaConf.select(cfg, "model.patch_size", default=4))),
        )

        model = Transformer2DGlobalAttentionForecaster(
            channels=int(cfg.data.in_channels),
            out_channels=int(cfg.data.out_channels),
            grid_height=int(cfg.data.grid_height),
            grid_width=int(cfg.data.grid_width),
            hidden_dim=int(cfg.model.hidden_dim),
            patch_size=patch_size_2d,
            num_layers=int(cfg.model.num_layers),
            num_heads=int(cfg.model.num_heads),
            mlp_ratio=float(OmegaConf.select(cfg, "model.mlp_ratio", default=4.0)),
            activation=str(cfg.model.activation),
            dropout=float(OmegaConf.select(cfg, "model.dropout", default=0.0)),
            attention_dropout=float(OmegaConf.select(cfg, "model.attention_dropout", default=0.0)),
        )

    elif model_type == "Transformer2DSwinForecaster":
        patch_size_2d = (
            int(OmegaConf.select(cfg, "model.patch_height", default=OmegaConf.select(cfg, "model.patch_size", default=4))),
            int(OmegaConf.select(cfg, "model.patch_width", default=OmegaConf.select(cfg, "model.patch_size", default=4))),
        )
        window_size_2d = (
            int(OmegaConf.select(cfg, "model.window_height", default=4)),
            int(OmegaConf.select(cfg, "model.window_width", default=4)),
        )
        model = Transformer2DSwinForecaster(
            channels=int(cfg.data.in_channels),
            out_channels=int(cfg.data.out_channels),
            grid_height=int(cfg.data.grid_height),
            grid_width=int(cfg.data.grid_width),
            hidden_dim=int(cfg.model.hidden_dim),
            patch_size=patch_size_2d,
            window_size=window_size_2d,
            num_layers=int(cfg.model.num_layers),
            num_heads=int(cfg.model.num_heads),
            mlp_ratio=float(OmegaConf.select(cfg, "model.mlp_ratio", default=4.0)),
            activation=str(cfg.model.activation),
            dropout=float(OmegaConf.select(cfg, "model.dropout", default=0.0)),
            attention_dropout=float(OmegaConf.select(cfg, "model.attention_dropout", default=0.0)),
        )

    elif model_type == "Transformer3DGlobalAttentionForecaster":
        patch_size_3d = (
            int(OmegaConf.select(cfg, "model.patch_depth", default=OmegaConf.select(cfg, "model.patch_size", default=5))),
            int(OmegaConf.select(cfg, "model.patch_height", default=OmegaConf.select(cfg, "model.patch_size", default=4))),
            int(OmegaConf.select(cfg, "model.patch_width", default=OmegaConf.select(cfg, "model.patch_size", default=4))),
        )
        model = Transformer3DGlobalAttentionForecaster(
            channels=int(cfg.data.in_channels),
            out_channels=int(cfg.data.out_channels),
            grid_depth=int(cfg.data.grid_depth),
            grid_height=int(cfg.data.grid_height),
            grid_width=int(cfg.data.grid_width),
            hidden_dim=int(cfg.model.hidden_dim),
            patch_size=patch_size_3d,
            num_layers=int(cfg.model.num_layers),
            num_heads=int(cfg.model.num_heads),
            mlp_ratio=float(OmegaConf.select(cfg, "model.mlp_ratio", default=4.0)),
            activation=str(cfg.model.activation),
            dropout=float(OmegaConf.select(cfg, "model.dropout", default=0.0)),
            attention_dropout=float(OmegaConf.select(cfg, "model.attention_dropout", default=0.0)),
        )

    elif model_type == "Transformer3DSwinForecaster":
        patch_size_3d = (
            int(OmegaConf.select(cfg, "model.patch_depth", default=OmegaConf.select(cfg, "model.patch_size", default=5))),
            int(OmegaConf.select(cfg, "model.patch_height", default=OmegaConf.select(cfg, "model.patch_size", default=4))),
            int(OmegaConf.select(cfg, "model.patch_width", default=OmegaConf.select(cfg, "model.patch_size", default=4))),
        )
        window_size_3d = (
            int(OmegaConf.select(cfg, "model.window_depth", default=2)),
            int(OmegaConf.select(cfg, "model.window_height", default=4)),
            int(OmegaConf.select(cfg, "model.window_width", default=4)),
        )
        model = Transformer3DSwinForecaster(
            channels=int(cfg.data.in_channels),
            out_channels=int(cfg.data.out_channels),
            grid_depth=int(cfg.data.grid_depth),
            grid_height=int(cfg.data.grid_height),
            grid_width=int(cfg.data.grid_width),
            hidden_dim=int(cfg.model.hidden_dim),
            patch_size=patch_size_3d,
            window_size=window_size_3d,
            num_layers=int(cfg.model.num_layers),
            num_heads=int(cfg.model.num_heads),
            mlp_ratio=float(OmegaConf.select(cfg, "model.mlp_ratio", default=4.0)),
            activation=str(cfg.model.activation),
            dropout=float(OmegaConf.select(cfg, "model.dropout", default=0.0)),
            attention_dropout=float(OmegaConf.select(cfg, "model.attention_dropout", default=0.0)),
        )

    elif model_type == "MeshGNN2D":
        model = MeshGNN2D(
            in_channels=int(cfg.data.in_channels),
            out_channels=int(cfg.data.out_channels),
            grid_node_feature_dim=int(cfg.data.grid_node_feature_dim),
            mesh_node_feature_dim=int(cfg.data.mesh_node_feature_dim),
            g2m_edge_dim=int(cfg.data.g2m_edge_feature_dim),
            mesh_edge_dim=int(cfg.data.mesh_edge_feature_dim),
            m2g_edge_dim=int(cfg.data.m2g_edge_feature_dim),
            hidden_dim=int(cfg.model.hidden_dim),
            mpnn_layer_type=str(OmegaConf.select(cfg, "model.mpnn_layer_type", default="GatedGCNConv")),
            grid2mesh_num_layers=int(OmegaConf.select(cfg, "model.grid2mesh_num_layers", default=1)),
            mesh2mesh_num_layers=int(cfg.model.mesh2mesh_num_layers),
            mesh2grid_num_layers=int(OmegaConf.select(cfg, "model.mesh2grid_num_layers", default=1)),
            node_encoder_type=str(OmegaConf.select(cfg, "model.node_encoder_type", default="mlp")),
            edge_encoder_type=str(OmegaConf.select(cfg, "model.edge_encoder_type", default="mlp")),
            decoder_type=str(OmegaConf.select(cfg, "model.decoder_type", default="mlp")),
            node_encoder_layers=int(OmegaConf.select(cfg, "model.node_encoder_layers", default=2)),
            edge_encoder_layers=int(OmegaConf.select(cfg, "model.edge_encoder_layers", default=2)),
            decoder_layers=int(OmegaConf.select(cfg, "model.decoder_layers", default=2)),
            activation=str(cfg.model.activation),
            dropout=float(OmegaConf.select(cfg, "model.dropout", default=0.0)),
        )

    elif model_type == "SimpleGNN2D":
        gnn_layer_type = str(cfg.model.gnn_layer_type)
        uses_attention = gnn_layer_type == "GATResBlock"
        uses_edge_features = gnn_layer_type in {"GATResBlock", "GatedGCNResBlock"}

        model = SimpleGNN2D(
            in_channels=int(cfg.data.in_channels),
            out_channels=int(cfg.data.out_channels),
            num_nodes=int(cfg.data.num_nodes),
            batch_size=int(cfg.train.batch_size),
            edge_dim=int(cfg.data.num_edge_features),
            hidden_dim=int(cfg.model.hidden_dim),
            num_layers=int(cfg.model.num_layers),
            num_heads=int(OmegaConf.select(cfg, "model.num_heads", default=4 if uses_attention else 1)),
            activation=str(cfg.model.activation),
            dropout=float(cfg.model.dropout),
            gnn_layer_type=gnn_layer_type,
            node_encoder_type=str(cfg.model.node_encoder_type),
            edge_encoder_type=str(OmegaConf.select(cfg, "model.edge_encoder_type", default="linear")),
            decoder_type=str(cfg.model.decoder_type),
            edge_encoder_hidden_dim=int(OmegaConf.select(cfg, "model.edge_encoder_hidden_dim", default=cfg.model.hidden_dim)),
            node_encoder_layers=int(cfg.model.node_encoder_layers),
            edge_encoder_layers=int(OmegaConf.select(cfg, "model.edge_encoder_layers", default=1)),
            decoder_layers=int(cfg.model.decoder_layers),
            use_edge_features=uses_edge_features,
        )

    else:
        raise ValueError(
            f"Model type {model_type} not supported. "
            "Please choose from ['SimpleCNN2D', 'SimpleCNN3D', 'UnetCNN2D_Fixed', 'UnetCNN3D_Fixed', "
            "'Transformer2DGlobalAttentionForecaster', 'Transformer2DSwinForecaster', "
            "'Transformer3DGlobalAttentionForecaster', 'Transformer3DSwinForecaster', "
            "'MeshGNN2D', 'SimpleGNN2D']"
        )

    print("Model architecture: " + str(model))
    return model
