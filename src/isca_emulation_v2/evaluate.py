from pathlib import Path

import os
import numpy as np
import torch
import typer
import wandb
import yaml
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from isca_emulation_v2.data.cnn2d import inverse_scale_cnn2d
from isca_emulation_v2.data.cnn3d import inverse_scale_cnn3d
from isca_emulation_v2.data.mesh_gnn import inverse_scale_mesh_gnn
from isca_emulation_v2.data.gnn2d import inverse_scale_gnn2d
from isca_emulation_v2.data.utils import load_isca_result_data
from isca_emulation_v2.models.model import load_model
from isca_emulation_v2.plotting.plotting_functions import plot_isca_emulator_result_multi
from isca_emulation_v2.process_data import DATASET_NAMES
from isca_emulation_v2.train_tools.loss_functions import get_loss_function
from isca_emulation_v2.utils import (
    add_data_cfg_from_manifest,
    cleanup_gpu_memory,
    get_data_attributes_cnn,
    get_data_attributes_mesh_gnn,
    get_data_attributes_gnn,
    run_validation_loop,
    select_best_gpu,
)

app = typer.Typer()


class EvaluateModel:
    def __init__(self, cfg):
        self.wandb_project_name = str(cfg.wandb_project_name)
        self.visualize = bool(cfg.visualize)
        self.error_percentile = float(OmegaConf.select(cfg, "error_percentile", default=1.0))
        if not (0 < self.error_percentile <= 1):
            raise ValueError(
                f"`error_percentile` must be in (0, 1], got {self.error_percentile}."
            )

        wandb_runs = OmegaConf.select(cfg, "wandb_runs", default=None)
        if wandb_runs is not None:
            self.wandb_runs = [str(run_path) for run_path in wandb_runs]
        else:
            single_run = OmegaConf.select(cfg, "wandb_run", default=None)
            if single_run is None:
                raise ValueError("Provide either `wandb_run` or `wandb_runs` in the inference config.")
            self.wandb_runs = [str(single_run)]

        if len(self.wandb_runs) == 0:
            raise ValueError("`wandb_runs` is empty. Provide at least one wandb run path.")

        test_data_names = OmegaConf.select(cfg, "test_data_names", default=None)
        single_test_data_name = OmegaConf.select(cfg, "test_data_name", default=None)
        if test_data_names is not None:
            self.test_data_names = [str(name) for name in test_data_names]
            if len(self.test_data_names) == 1 and len(self.wandb_runs) > 1:
                self.test_data_names = self.test_data_names * len(self.wandb_runs)
            if len(self.test_data_names) != len(self.wandb_runs):
                raise ValueError(
                    "`test_data_names` must have the same length as `wandb_runs`, "
                    f"got {len(self.test_data_names)} and {len(self.wandb_runs)}."
                )
        else:
            if single_test_data_name is None:
                raise ValueError("Provide either `test_data_name` or `test_data_names` in the inference config.")
            self.test_data_names = [str(single_test_data_name)] * len(self.wandb_runs)

    def _cfg_from_wandb_run(self, run) -> OmegaConf:
        wandb_config_dict = dict(run.config)
        nested_config = {}
        for flat_key, value in wandb_config_dict.items():
            if str(flat_key).startswith("_"):
                continue
            parts = str(flat_key).split(".")
            current = nested_config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

        cfg = OmegaConf.create(nested_config)
        model_type = OmegaConf.select(cfg, "model.model_type", default=None)
        if model_type is None:
            layer_type = OmegaConf.select(cfg, "model.layer_type", default=None)
            if layer_type is not None:
                with open_dict(cfg):
                    cfg.model.model_type = layer_type

        return add_data_cfg_from_manifest(cfg)

    def _load_wandb_context(self, wandb_run: str, test_data_name: str) -> dict:
        api = wandb.Api()
        run = api.run(wandb_run)
        cfg = self._cfg_from_wandb_run(run)

        artifacts = list(run.logged_artifacts())
        if len(artifacts) == 0:
            raise ValueError(f"No artifacts found in wandb run {wandb_run}")
        model_artifacts = [artifact for artifact in artifacts if artifact.type == "model"]
        if len(model_artifacts) == 0:
            raise ValueError(f"No model artifacts found in wandb run {wandb_run}")
        if len(model_artifacts) > 1:
            print(f"Multiple model artifacts found: {[a.name for a in model_artifacts]}")
            print("Using the first model artifact.")

        model_artifact = model_artifacts[0].name
        artifact = api.artifact(f"{run.project}/{model_artifact}")
        artifacts_dir = os.path.join("artifacts", f"{run.name}_{test_data_name}")
        model_checkpoint_dir = artifact.download(root=artifacts_dir)
        model_checkpoint = os.path.join(model_checkpoint_dir, model_artifact.split(":")[0])

        return {
            "cfg": cfg,
            "run_name": str(run.name),
            "artifacts_dir": artifacts_dir,
            "model_artifact": model_artifact,
            "model_checkpoint": model_checkpoint,
        }

    def _load_dataset(self, cfg, test_data_name: str):
        test_path_root = os.path.join("data", "processed", test_data_name)
        train_path_root = os.path.join("data", "processed", cfg.data.output.dataset_name)

        scaling = torch.load(f"{train_path_root}/stats.pt")
        manifest_path = os.path.join(test_path_root, "manifest.yaml")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}

        dataset_type = str(cfg.data.processing.dataset_type)
        dataset_type_obj = DATASET_NAMES[dataset_type]

        if dataset_type in {"SimpleDatasetCNN2D", "SimpleDatasetCNN3D"}:
            split_shards = manifest["split_shards"]["test"]
            shard_paths = [os.path.join(test_path_root, item["path"]) for item in split_shards]
            shard_lengths = [int(item["shape"][0]) for item in split_shards]
            grid = manifest.get("grid", {})
            cache_in_memory = bool(OmegaConf.select(cfg, "train.cache_in_memory", default=False))
            dataset = dataset_type_obj(
                shard_paths=shard_paths,
                shard_lengths=shard_lengths,
                mean=scaling["mean"],
                std=scaling["std"],
                batch_size=cfg.train.batch_size,
                lonlat_values=grid,
                add_coords=cfg.data.processing.params.add_coords,
                shuffle_shards=False,
                shuffle_in_shard=False,
                cache_in_memory=cache_in_memory,
            )

            data_point = next(iter(dataset))
            cfg = get_data_attributes_cnn(cfg, data_point)

            test_loader = DataLoader(
                dataset,
                batch_size=None,
                shuffle=False,
                num_workers=int(OmegaConf.select(cfg, "train.num_workers", default=0)),
                pin_memory=bool(OmegaConf.select(cfg, "train.pin_memory", default=False)),
            )
            return dataset, test_loader, cfg

        if dataset_type == "SimpleDatasetGNN2D":
            split_shards = manifest["split_shards"]["test"]
            shard_paths = [os.path.join(test_path_root, item["path"]) for item in split_shards]
            shard_lengths = [int(item["shape"][0]) for item in split_shards]
            cache_in_memory = False
            add_coords_node = OmegaConf.select(cfg, "data.processing.params.add_coords_node", default="trig4")
            dataset = dataset_type_obj(
                shard_paths=shard_paths,
                shard_lengths=shard_lengths,
                mean=scaling["mean"],
                std=scaling["std"],
                batch_size=cfg.train.batch_size,
                add_coords_node=add_coords_node,
                shuffle_shards=False,
                shuffle_in_shard=False,
                cache_in_memory=cache_in_memory,
            )

            data_point = next(iter(dataset))
            cfg = get_data_attributes_gnn(cfg, data_point, dataset.edge_attr)

            test_loader = DataLoader(
                dataset,
                batch_size=None,
                shuffle=False,
                num_workers=int(OmegaConf.select(cfg, "train.num_workers", default=0)),
                pin_memory=bool(OmegaConf.select(cfg, "train.pin_memory", default=False)),
            )
            return dataset, test_loader, cfg

        if dataset_type == "SimpleDatasetMeshGNN":
            split_shards = manifest["split_shards"]["test"]
            shard_paths = [os.path.join(test_path_root, item["path"]) for item in split_shards]
            shard_lengths = [int(item["shape"][0]) for item in split_shards]
            static_rel_path = manifest.get("artifacts", {}).get("static_graph", "mesh_gnn_static.pt")
            cache_in_memory = False
            dataset = dataset_type_obj(
                shard_paths=shard_paths,
                shard_lengths=shard_lengths,
                static_path=os.path.join(test_path_root, static_rel_path),
                mean=scaling["mean"],
                std=scaling["std"],
                batch_size=cfg.train.batch_size,
                shuffle_shards=False,
                shuffle_in_shard=False,
                cache_in_memory=cache_in_memory,
            )

            data_point = next(iter(dataset))
            cfg = get_data_attributes_mesh_gnn(cfg, data_point, dataset)

            test_loader = DataLoader(
                dataset,
                batch_size=None,
                shuffle=False,
                num_workers=int(OmegaConf.select(cfg, "train.num_workers", default=0)),
                pin_memory=bool(OmegaConf.select(cfg, "train.pin_memory", default=False)),
            )
            return dataset, test_loader, cfg

        raise ValueError(f"Dataset type {dataset_type} not implemented")

    def _load_model(self, cfg, dataset):
        model = load_model(cfg)
        if cfg.train.device == "cuda" and torch.cuda.is_available():
            num_parameters = sum(p.numel() for p in model.parameters())
            cfg.train.device = select_best_gpu()
            print(f"Total trainable parameters: {num_parameters:,}")
            print(f"Selected GPU: {cfg.train.device}")

        model = model.to(cfg.train.device)
        if cfg.data.processing.dataset_type == "SimpleDatasetGNN2D" and hasattr(model, "set_graph"):
            edge_index = dataset.edge_index.to(cfg.train.device)
            edge_attr = dataset.edge_attr.to(cfg.train.device)
            model.set_graph(edge_index=edge_index, edge_attr=edge_attr)
        if cfg.data.processing.dataset_type == "SimpleDatasetMeshGNN" and hasattr(model, "set_static_graph"):
            model.set_static_graph(
                grid_node_features=dataset.grid_static["node_features"],
                mesh_node_features=dataset.mesh_static["node_features"],
                g2m_edge_index=dataset.grid2mesh["edge_index"],
                g2m_edge_features=dataset.grid2mesh["edge_features"],
                mesh_edge_index=dataset.mesh_graph["edge_index"],
                mesh_edge_features=dataset.mesh_graph["edge_features"],
                m2g_edge_index=dataset.mesh2grid["edge_index"],
                m2g_edge_features=dataset.mesh2grid["edge_features"],
            )
        print(f"Loaded {cfg.model.model_type} model")
        return model, cfg

    def _nodes_to_grid(self, arr: np.ndarray, nlat: int, nlon: int) -> np.ndarray:
        if arr.ndim != 3:
            raise ValueError(f"GNN visualization expects rank-3 tensors [N, num_nodes, F], got shape={tuple(arr.shape)}")
        num_nodes_expected = nlat * nlon
        if int(arr.shape[1]) != num_nodes_expected:
            raise ValueError(
                f"GNN node/grid mismatch: tensor has num_nodes={int(arr.shape[1])}, "
                f"grid implies num_nodes={num_nodes_expected} ({nlat}x{nlon})"
            )
        return arr.reshape(arr.shape[0], nlat, nlon, arr.shape[2]).transpose(0, 3, 1, 2)

    def _prepare_visual_payload(
        self,
        cfg,
        dataset_type: str,
        inverse_scale_fn,
        x_true_list: list[torch.Tensor],
        y_true_np_unscaled: np.ndarray,
        y_pred_np_unscaled: np.ndarray,
    ) -> dict:
        x_true_tensor = torch.cat(x_true_list, dim=0)
        x_true_np = x_true_tensor.cpu().detach().numpy()
        if dataset_type == "SimpleDatasetGNN2D":
            x_true_np = x_true_np[..., : y_true_np_unscaled.shape[-1]]
        else:
            x_true_np = x_true_np[:, : y_true_np_unscaled.shape[1]]
        x_true_np_unscaled = inverse_scale_fn(x_true_np, cfg)

        ds_meta = load_isca_result_data(
            exp_folder_name=cfg.data.raw.exp_folder_name,
            file_name=cfg.data.raw.file_name,
        )
        pvals = ds_meta[cfg.data.processing.params.level_dim].values.astype(float)
        lat = ds_meta[cfg.data.processing.params.lat_dim].values
        lon = ds_meta[cfg.data.processing.params.lon_dim].values
        vars_cfg = list(cfg.data.processing.params.vars)

        if dataset_type in {"SimpleDatasetGNN2D", "SimpleDatasetMeshGNN"}:
            nlat = int(len(lat))
            nlon = int(len(lon))
            x_true_np_unscaled = self._nodes_to_grid(x_true_np_unscaled, nlat, nlon)
            y_true_np_unscaled = self._nodes_to_grid(y_true_np_unscaled, nlat, nlon)
            y_pred_np_unscaled = self._nodes_to_grid(y_pred_np_unscaled, nlat, nlon)

        channel_index = [(var_name, float(level)) for var_name in vars_cfg for level in pvals]
        if dataset_type == "SimpleDatasetCNN3D":
            if x_true_np_unscaled.ndim != 5:
                raise ValueError(
                    "CNN3D visualization expects rank-5 tensors [N, C, L, H, W], "
                    f"got shape={tuple(x_true_np_unscaled.shape)}"
                )
            expected_channels = len(vars_cfg)
            expected_levels = len(pvals)
            if int(x_true_np_unscaled.shape[1]) != expected_channels:
                raise ValueError(
                    "Channel mismatch for CNN3D visualization: "
                    f"tensor has C={int(x_true_np_unscaled.shape[1])}, metadata implies C={expected_channels}"
                )
            if int(x_true_np_unscaled.shape[2]) != expected_levels:
                raise ValueError(
                    "Level mismatch for CNN3D visualization: "
                    f"tensor has L={int(x_true_np_unscaled.shape[2])}, metadata implies L={expected_levels}"
                )
            # Canonicalize CNN3D to channel-major 2D maps [N, C*L, H, W] so it can
            # be compared directly with CNN2D/GNN outputs that already use flattened channels.
            n_samples, n_channels, n_levels, h, w = x_true_np_unscaled.shape
            flat_channels = n_channels * n_levels
            x_true_np_unscaled = x_true_np_unscaled.reshape(n_samples, flat_channels, h, w)
            y_true_np_unscaled = y_true_np_unscaled.reshape(n_samples, flat_channels, h, w)
            y_pred_np_unscaled = y_pred_np_unscaled.reshape(n_samples, flat_channels, h, w)
        else:
            if len(channel_index) != int(x_true_np_unscaled.shape[1]):
                raise ValueError(
                    f"Channel mismatch between tensor and metadata: "
                    f"tensor has C={int(x_true_np_unscaled.shape[1])}, metadata implies C={len(channel_index)}"
                )

        units_by_var = {var_name: ds_meta[var_name].attrs.get("units", "") for var_name in vars_cfg}
        return {
            "x_true_unscaled": x_true_np_unscaled,
            "y_true_unscaled": y_true_np_unscaled,
            "y_pred_unscaled": y_pred_np_unscaled,
            "lat": lat,
            "lon": lon,
            "channel_index": channel_index,
            "units_by_var": units_by_var,
        }

    def evaluate_single_run(self, wandb_run: str, test_data_name: str) -> dict:
        print(f"\n=== Evaluating run: {wandb_run} on {test_data_name} ===")
        context = self._load_wandb_context(wandb_run, test_data_name)
        cfg = context["cfg"]
        run_name = context["run_name"]
        model_checkpoint = context["model_checkpoint"]
        artifacts_dir = context["artifacts_dir"]

        train_dataset_name = str(cfg.data.output.dataset_name)
        is_ood_evaluation = train_dataset_name != test_data_name

        print(f"Model artifact: {context['model_artifact']}")
        print(f"Model checkpoint downloaded to: {model_checkpoint}")
        print(f"Evaluating {cfg.model.model_type} model")
        if is_ood_evaluation:
            print(f"Out-of-Distribution Evaluation: {train_dataset_name} -> {test_data_name}")
        else:
            print(f"Same-Distribution Evaluation: {test_data_name}")

        dataset, test_loader, cfg = self._load_dataset(cfg, test_data_name)
        model, cfg = self._load_model(cfg, dataset)
        model.load_state_dict(
            torch.load(model_checkpoint, map_location=cfg.train.device, weights_only=False)
        )
        model.eval()
        loss_fn = get_loss_function(cfg)

        y_pred_list: list[torch.Tensor] = []
        y_true_list: list[torch.Tensor] = []
        x_true_list: list[torch.Tensor] = []

        def _collect_batch(x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor) -> None:
            y_pred_list.append(y_pred.detach().cpu())
            y_true_list.append(y.detach().cpu())
            x_true_list.append(x.detach().cpu())

        def _persistence_model(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 3:
                return x[..., : cfg.data.out_channels]
            return x[:, : cfg.data.out_channels]

        persistence_loss, persistence_scaled_metrics = run_validation_loop(
            loader=test_loader,
            model=_persistence_model,
            loss_fn=loss_fn,
            device=cfg.train.device,
            return_metrics=True,
        )

        avg_loss, scaled_metrics = run_validation_loop(
            loader=test_loader,
            model=model,
            loss_fn=loss_fn,
            device=cfg.train.device,
            return_metrics=True,
            batch_hook=_collect_batch,
        )

        print("Test metrics (persistence):")
        print(f"  Test loss: {persistence_loss:.6f}")
        print(f"  MAE: {persistence_scaled_metrics['mae_mean']:.6f}")
        print(f"  MSE: {persistence_scaled_metrics['mse_mean']:.6f}")
        print(f"  RMSE: {persistence_scaled_metrics['rmse_mean']:.6f}")
        print("Test metrics (model):")
        print(f"  Test loss: {avg_loss:.6f}")
        print(f"  MAE: {scaled_metrics['mae_mean']:.6f}")
        print(f"  MSE: {scaled_metrics['mse_mean']:.6f}")
        print(f"  RMSE: {scaled_metrics['rmse_mean']:.6f}")

        y_pred_np = torch.cat(y_pred_list, dim=0).numpy()
        y_true_np = torch.cat(y_true_list, dim=0).numpy()

        dataset_type = str(cfg.data.processing.dataset_type)
        if dataset_type == "SimpleDatasetCNN3D":
            inverse_scale_fn = inverse_scale_cnn3d
        elif dataset_type == "SimpleDatasetGNN2D":
            inverse_scale_fn = inverse_scale_gnn2d
        elif dataset_type == "SimpleDatasetMeshGNN":
            inverse_scale_fn = inverse_scale_mesh_gnn
        else:
            inverse_scale_fn = inverse_scale_cnn2d

        y_pred_np_unscaled = inverse_scale_fn(y_pred_np, cfg)
        y_true_np_unscaled = inverse_scale_fn(y_true_np, cfg)

        evaluation_results = {
            "y_pred_unscaled": y_pred_np_unscaled,
            "y_true_unscaled": y_true_np_unscaled,
            "evaluation_info": {
                "train_dataset": train_dataset_name,
                "test_dataset": test_data_name,
                "is_ood_evaluation": is_ood_evaluation,
                "model_type": cfg.model.model_type,
                "wandb_run": wandb_run,
            },
            "metrics": {
                "loss": avg_loss,
                "scaled": scaled_metrics,
                "persistence": {
                    "loss": persistence_loss,
                    "scaled": persistence_scaled_metrics,
                },
            },
        }
        evaluation_results_path = os.path.join(artifacts_dir, "evaluation_results.pt")
        torch.save(evaluation_results, evaluation_results_path, pickle_protocol=4)
        print(f"Saved evaluation results to {evaluation_results_path}")

        if "cuda" in str(cfg.train.device):
            cleanup_gpu_memory()

        visual_payload = None
        if self.visualize:
            visual_payload = self._prepare_visual_payload(
                cfg=cfg,
                dataset_type=dataset_type,
                inverse_scale_fn=inverse_scale_fn,
                x_true_list=x_true_list,
                y_true_np_unscaled=y_true_np_unscaled,
                y_pred_np_unscaled=y_pred_np_unscaled,
            )

        return {
            "wandb_run": wandb_run,
            "run_name": run_name,
            "model_type": str(cfg.model.model_type),
            "dataset_type": dataset_type,
            "test_dataset_name": test_data_name,
            "evaluation_results_path": evaluation_results_path,
            "loss": float(avg_loss),
            "scaled_metrics": scaled_metrics,
            "visual": visual_payload,
        }

    def evaluate(self) -> None:
        run_results: list[dict] = []
        for wandb_run, test_data_name in zip(self.wandb_runs, self.test_data_names, strict=True):
            run_result = self.evaluate_single_run(wandb_run, test_data_name)
            run_results.append(run_result)

        print("\n=== Summary (model test metrics) ===")
        for result in run_results:
            scaled = result["scaled_metrics"]
            print(f"[{result['run_name']}] {result['model_type']}")
            print(f"  Test dataset={result['test_dataset_name']}")
            print(f"  Loss={result['loss']:.6f} MAE={scaled['mae_mean']:.6f} MSE={scaled['mse_mean']:.6f} RMSE={scaled['rmse_mean']:.6f}")

        if not self.visualize:
            return

        reference = run_results[0]["visual"]
        model_rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for result in run_results:
            visual = result["visual"]
            if tuple(visual["y_pred_unscaled"].shape[1:]) != tuple(reference["y_true_unscaled"].shape[1:]):
                raise ValueError(
                    f"Cannot compare runs with mismatched prediction shape (excluding sample dim). "
                    f"Run {result['wandb_run']} has {tuple(visual['y_pred_unscaled'].shape)}, "
                    f"reference is {tuple(reference['y_true_unscaled'].shape)}."
                )
            if tuple(visual["x_true_unscaled"].shape[1:]) != tuple(reference["x_true_unscaled"].shape[1:]):
                raise ValueError(
                    f"Cannot compare runs with mismatched input shape (excluding sample dim). "
                    f"Run {result['wandb_run']} has {tuple(visual['x_true_unscaled'].shape)}, "
                    f"reference is {tuple(reference['x_true_unscaled'].shape)}."
                )
            if len(visual["lat"]) != len(reference["lat"]) or len(visual["lon"]) != len(reference["lon"]):
                raise ValueError(f"Cannot compare runs with different grids: {result['wandb_run']}")
            if visual["channel_index"] != reference["channel_index"]:
                raise ValueError(f"Cannot compare runs with different channel mapping: {result['wandb_run']}")

            label = f"{result['model_type']} | {result['run_name']}"
            model_rows.append(
                (
                    label,
                    visual["x_true_unscaled"],
                    visual["y_true_unscaled"],
                    visual["y_pred_unscaled"],
                )
            )

        plot_isca_emulator_result_multi(
            model_rows,
            lat=reference["lat"],
            lon=reference["lon"],
            channel_index=reference["channel_index"],
            units_by_var=reference["units_by_var"],
            error_percentile=self.error_percentile,
        )


@app.command()
def main(
    inference_cfg_name: str = typer.Argument(
        ...,
        help="YAML filename located in config_inference (for example: default.yaml).",
    )
):

    cfg = OmegaConf.load(inference_cfg_name)

    evaluate = EvaluateModel(cfg)
    evaluate.evaluate()


if __name__ == "__main__":
    app()
