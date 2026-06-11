import torch
from torch.utils.data import DataLoader
from isca_emulation_v2.models.model import load_model
from isca_emulation_v2.train_tools.optimizers import get_optimizer
from isca_emulation_v2.train_tools.loss_functions import get_loss_function
from isca_emulation_v2.train_tools.schedulers import get_scheduler
from isca_emulation_v2.utils import (
    add_data_cfg_from_manifest,
    get_data_attributes_cnn,
    get_data_attributes_mesh_gnn,
    get_data_attributes_gnn,
    compute_validation_metrics,
    finalize_validation_metrics,
    run_validation_loop,
    select_best_gpu,
    cleanup_gpu_memory,
)

from isca_emulation_v2.process_data import DATASET_NAMES
from isca_emulation_v2.wandb import generate_sweep_configuration, WandbLogger

from hydra import initialize, compose
from omegaconf import OmegaConf
import os
import os.path as osp
import numpy as np
import random
import wandb
import typer
from typing import Dict, Any
import yaml
import time
import traceback
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR, ReduceLROnPlateau
from torch.amp import GradScaler, autocast
import gc
from tqdm.auto import tqdm

app = typer.Typer()

def _is_cuda_oom_error(err: BaseException) -> bool:
    oom_err_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_err_type is not None and isinstance(err, oom_err_type):
        return True

    err_msg = str(err).lower()
    return ("cuda" in err_msg and "out of memory" in err_msg) or "cuda oom" in err_msg

def _clear_exception_traceback_frames(err: BaseException) -> None:
    seen: set[int] = set()
    current: BaseException | None = err
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        try:
            if current.__traceback__ is not None:
                traceback.clear_frames(current.__traceback__)
        except Exception:
            pass
        current = current.__cause__ if current.__cause__ is not None else current.__context__

class TrainModel:
    def __init__(self, cfg: Dict[str, Any], add_agent: bool = False) -> None:
        self.add_agent = add_agent
        self.cfg = cfg
        
        # Validate config for non-agent runs
        if not add_agent and cfg is None:
            raise ValueError("cfg cannot be None when add_agent=False")

    def _check_data_path(self) -> None:
        processed_dir = "data/processed"
        self.processed_data_path = osp.join(processed_dir, self.cfg.data.output.dataset_name)

    def _set_seed(self) -> None:
        """Set the random seed for reproducibility."""
        seed = self.cfg.train.random_seed
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(False) # Set false for training using cuda

    @staticmethod
    def _worker_init_fn(worker_id) -> None:
        seed = torch.initial_seed() % (2**32)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def _sanitize_batch_tensors(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return x, y

    @staticmethod
    def _scheduler_steps_per_batch(scheduler) -> bool:
        return isinstance(scheduler, (OneCycleLR, CyclicLR))

    def _load_dataset(self, split: str):
        assert split in {"train", "val"}, "split must be 'train' or 'val'"
        root = osp.join("data/processed", self.cfg.data.output.dataset_name)        

        scaling = torch.load(f"{root}/stats.pt")
        manifest_path = osp.join(root, "manifest.yaml")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}

        dataset_type = self.cfg.data.processing.dataset_type
        dataset_type_obj = DATASET_NAMES[dataset_type]

        if dataset_type in {"SimpleDatasetCNN2D", "SimpleDatasetCNN3D"}:
            split_shards = manifest["split_shards"][split]
            shard_paths = [osp.join(root, item["path"]) for item in split_shards]
            shard_lengths = [int(item["shape"][0]) for item in split_shards]
            grid = manifest.get("grid", {})
            dataset = dataset_type_obj(
                shard_paths=shard_paths,
                shard_lengths=shard_lengths,
                mean=scaling["mean"],
                std=scaling["std"],
                batch_size=self.cfg.train.batch_size,
                lonlat_values=grid,
                add_coords=self.cfg.data.processing.params.add_coords,
                shuffle_shards=(split == "train"),
                shuffle_in_shard=(split == "train"),
                cache_in_memory=bool(self.cfg.train.cache_in_memory),
            )
        elif dataset_type == "SimpleDatasetGNN2D":
            split_shards = manifest["split_shards"][split]
            shard_paths = [osp.join(root, item["path"]) for item in split_shards]
            shard_lengths = [int(item["shape"][0]) for item in split_shards]
            dataset = dataset_type_obj(
                shard_paths=shard_paths,
                shard_lengths=shard_lengths,
                mean=scaling["mean"],
                std=scaling["std"],
                batch_size=self.cfg.train.batch_size,
                add_coords_node=self.cfg.data.processing.params.add_coords_node,
                shuffle_shards=(split == "train"),
                shuffle_in_shard=(split == "train"),
                cache_in_memory=bool(self.cfg.train.cache_in_memory),
            )
        elif dataset_type == "SimpleDatasetMeshGNN":
            split_shards = manifest["split_shards"][split]
            shard_paths = [osp.join(root, item["path"]) for item in split_shards]
            shard_lengths = [int(item["shape"][0]) for item in split_shards]
            static_rel_path = manifest.get("artifacts", {}).get("static_graph", "mesh_gnn_static.pt")
            dataset = dataset_type_obj(
                shard_paths=shard_paths,
                shard_lengths=shard_lengths,
                static_path=osp.join(root, static_rel_path),
                mean=scaling["mean"],
                std=scaling["std"],
                batch_size=self.cfg.train.batch_size,
                shuffle_shards=(split == "train"),
                shuffle_in_shard=(split == "train"),
                cache_in_memory=bool(self.cfg.train.cache_in_memory),
            )
        else:
            # Error dataset_type not implemented
            raise ValueError(f"Dataset type {dataset_type} not implemented")

        return dataset

    def _build_loader_kwargs(self, shuffle: bool) -> dict[str, Any]:
        num_workers = int(self.cfg.train.num_workers)
        pin_memory = bool(self.cfg.train.pin_memory)
        persistent_workers = bool(self.cfg.train.persistent_workers)
        prefetch_factor = int(self.cfg.train.prefetch_factor)

        kwargs: dict[str, Any] = {
            "batch_size": None,
            "shuffle": shuffle,
            "worker_init_fn": self._worker_init_fn,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = persistent_workers
            kwargs["prefetch_factor"] = prefetch_factor
        return kwargs

    def _compute_persistence_baseline(self, val_loader) -> tuple[float, dict[str, float]]:
        dataset_type = self.cfg.data.processing.dataset_type
        val_loss_sum = 0.0
        val_num_elements = 0
        metric_state = None
        with torch.inference_mode():
            for batch in val_loader:
                x_val, y_val = batch
                x_val = x_val.to(self.cfg.train.device, non_blocking=True)
                y_val = y_val.to(self.cfg.train.device, non_blocking=True)
                x_val, y_val = self._sanitize_batch_tensors(x_val, y_val)
                if dataset_type in {"SimpleDatasetGNN2D", "SimpleDatasetMeshGNN"}:
                    persistence_pred = x_val[..., : y_val.shape[-1]]
                elif dataset_type in {"SimpleDatasetCNN2D", "SimpleDatasetCNN3D"}:
                    persistence_pred = x_val[:, : y_val.shape[1]]
                else:
                    raise ValueError(f"Unsupported dataset type for persistence baseline: {dataset_type}")
                val_batch_loss = self.loss_fn(persistence_pred, y_val)
                batch_elements = y_val.numel()
                val_loss_sum += float(val_batch_loss.detach().item()) * batch_elements
                val_num_elements += batch_elements
                metric_state = compute_validation_metrics(
                    y_pred=persistence_pred,
                    y_true=y_val,
                    metric_state=metric_state,
                )
        return val_loss_sum / val_num_elements, finalize_validation_metrics(metric_state)

    def _initialize_early_stopping_state(self) -> dict[str, Any]:
        enabled = bool(OmegaConf.select(self.cfg, "train.early_stopping", default=False))
        patience = max(
            1,
            int(OmegaConf.select(self.cfg, "train.early_stopping_patience", default=10)),
        )
        state = {
            "enabled": enabled,
            "patience": patience,
            "best_val_mae": float("inf"),
            "best_epoch": -1,
            "epochs_without_improvement": 0,
            "best_ckpt_path": None,
            "stopped_epoch": None,
        }
        if enabled:
            os.makedirs("models", exist_ok=True)
            state["best_ckpt_path"] = osp.join(
                "models",
                f"best_model_tmp_{int(time.time() * 1e6)}.pt",
            )
            print(f"Early stopping enabled (patience={patience} epochs).")

        self._best_ckpt_path = state["best_ckpt_path"]
        return state

    def _attach_model_static_state(self, model, dataset_type: str, train_data) -> None:
        if dataset_type == "SimpleDatasetGNN2D":
            edge_index = train_data.edge_index.to(self.cfg.train.device)
            edge_attr = train_data.edge_attr.to(self.cfg.train.device)
            model.set_graph(edge_index=edge_index, edge_attr=edge_attr)
        elif dataset_type == "SimpleDatasetMeshGNN":
            model.set_static_graph(
                grid_node_features=train_data.grid_static["node_features"],
                mesh_node_features=train_data.mesh_static["node_features"],
                g2m_edge_index=train_data.grid2mesh["edge_index"],
                g2m_edge_features=train_data.grid2mesh["edge_features"],
                mesh_edge_index=train_data.mesh_graph["edge_index"],
                mesh_edge_features=train_data.mesh_graph["edge_features"],
                m2g_edge_index=train_data.mesh2grid["edge_index"],
                m2g_edge_features=train_data.mesh2grid["edge_features"],
            )
        elif dataset_type in {"SimpleDatasetCNN2D", "SimpleDatasetCNN3D"}:
            return
        else:
            raise ValueError(f"Dataset type {dataset_type} not implemented")

    def _run_training_fit_test(self, model, optimizer, train_loader) -> None:
        if "cuda" not in str(self.cfg.train.device):
            return

        print(f"Running one-batch training fit test on {self.cfg.train.device}...")
        model.train()
        probe_scaler = GradScaler("cuda")

        x = y = y_pred = loss = None
        try:
            torch.cuda.reset_peak_memory_stats()
            x, y = next(iter(train_loader))
            x = x.to(self.cfg.train.device, non_blocking=True)
            y = y.to(self.cfg.train.device, non_blocking=True)
            x, y = self._sanitize_batch_tensors(x, y)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                y_pred = model(x)
                loss = self.loss_fn(y_pred, y)

            probe_scaler.scale(loss).backward()
            probe_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            probe_scaler.step(optimizer)
            probe_scaler.update()
            torch.cuda.synchronize()

            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"Training fit test passed. Peak allocated GPU memory: {peak_mb:.0f}MB")
        except RuntimeError as err:
            if _is_cuda_oom_error(err):
                _clear_exception_traceback_frames(err)
                raise RuntimeError(
                    f"Training fit test failed on {self.cfg.train.device} with CUDA out of memory."
                ) from err
            raise
        finally:
            optimizer.zero_grad(set_to_none=True)
            del x, y, y_pred, loss, probe_scaler
            cleanup_gpu_memory(self.cfg.train.device)

    def train(self) -> None:
        """Train the GCN model on the Sioux Falls dataset."""

        # Initialize wandb logger object
        wandb_logger = WandbLogger(self.cfg, add_agent=self.add_agent)
        wandb_logger._initialize_wandb()
        self.cfg = wandb_logger._wandb_config_to_cfg()
        self.cfg = add_data_cfg_from_manifest(self.cfg)
        self._check_data_path()
        
        self._set_seed()  # Set seed for reproducibility

        # Load training configuration
        print("Training configuration:")
        print(OmegaConf.to_yaml(self.cfg.train))
        metrics_every_n_epochs = max(1, int(self.cfg.train.metrics_every_n_epochs))
        early_stopping = self._initialize_early_stopping_state()

        # Load datasets with on-the-fly scaling transform
        train_data = self._load_dataset("train")
        val_data   = self._load_dataset("val")

        # Get data attributes to cfg
        data_point = next(iter(train_data))
        dataset_type = self.cfg.data.processing.dataset_type
        if dataset_type == "SimpleDatasetGNN2D":
            self.cfg = get_data_attributes_gnn(self.cfg, data_point, train_data.edge_attr)
        elif dataset_type == "SimpleDatasetMeshGNN":
            self.cfg = get_data_attributes_mesh_gnn(self.cfg, data_point, train_data)
        elif dataset_type in {"SimpleDatasetCNN2D", "SimpleDatasetCNN3D"}:
            self.cfg = get_data_attributes_cnn(self.cfg, data_point)
        else:
            raise ValueError(f"Dataset type {dataset_type} not implemented")
 
        train_loader = DataLoader(train_data, **self._build_loader_kwargs(shuffle=False))
        val_loader = DataLoader(val_data, **self._build_loader_kwargs(shuffle=False))

        # Number of batches
        self.num_batches = len(train_loader)

        # Build a probe model first to estimate size and validate empirical fit on the chosen GPU.
        probe_model = load_model(self.cfg)
        num_parameters = sum(p.numel() for p in probe_model.parameters())
        print(f"Total trainable parameters: {num_parameters:,}")

        # Get loss function
        self.loss_fn = get_loss_function(self.cfg)

        # Place model on the device with sufficient memory
        if self.cfg.train.device == "cuda" and torch.cuda.is_available():
            self.cfg.train.device = select_best_gpu()

        probe_model = probe_model.to(self.cfg.train.device)
        probe_optimizer = None
        try:
            self._attach_model_static_state(probe_model, dataset_type, train_data)
            probe_optimizer = get_optimizer(self.cfg, probe_model)
            self._run_training_fit_test(probe_model, probe_optimizer, train_loader)
        finally:
            if probe_optimizer is not None:
                try:
                    probe_optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass
            probe_optimizer = None
            probe_model = None
            gc.collect()
            cleanup_gpu_memory(self.cfg.train.device)

        # Re-seed and rebuild the actual training model so the fit test does not affect training state.
        self._set_seed()
        self.model = load_model(self.cfg)

        # Add model to wandb
        # Avoid expensive watch hooks during sweeps in long-lived agent processes.
        watch_enabled = not bool(self.cfg.wandb.sweep.enabled)
        if watch_enabled:
            wandb_logger.watch_model(self.model, log_graph=False)

        # Get optimizer
        self.optimizer = get_optimizer(self.cfg, self.model)

        # Move model to the selected device
        self.model = self.model.to(self.cfg.train.device)
        self._attach_model_static_state(self.model, dataset_type, train_data)

        # Get learning rate scheduler
        self.scheduler = get_scheduler(self.cfg, self.optimizer, self.num_batches)

        # Load GradScaler for mixed precision training
        if "cuda" in self.cfg.train.device:
            self.grad_scaler = GradScaler("cuda")
            print("Using mixed precision training.")
        else:
            self.grad_scaler = None
            print("Using full precision training.")

        # One-time persistence baseline on validation set (before training starts).
        persistence_loss, persistence_metrics = self._compute_persistence_baseline(val_loader)
        print("Persistence baseline (validation):")
        print(f"  MAE mean: {persistence_metrics['mae_mean']:.6f}")
        print(f"  RMSE mean: {persistence_metrics['rmse_mean']:.6f}")
        print(f"  MSE mean: {persistence_metrics['mse_mean']:.6f}")
        print(f"  Validation loss: {persistence_loss:.6f}")

        # Training loop
        last_validation_metrics = None
        displayed_val_metrics = {"mae": "nan", "rmse": "nan", "mse": "nan"}
        progress_bar = tqdm(
            range(self.cfg.train.epochs),
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,
            leave=True,
        )
        for epoch in progress_bar:
            self._current_epoch = epoch
            should_compute_metrics = early_stopping["enabled"] or (
                (epoch + 1) % metrics_every_n_epochs == 0
            )

            if hasattr(self.loss_fn, "update_schedule"):
                self.loss_fn.update_schedule(epoch)

            self.model.train()
            train_loss_sum = 0.0
            train_num_elements = 0
            grad_norm = 0.0
            train_start_time = time.perf_counter()

            batch_progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.cfg.train.epochs}",
                unit="batch",
                total=self.num_batches,
                dynamic_ncols=True,
                leave=False,
                position=1,
            )
            for batch_idx, batch in enumerate(batch_progress, start=1):
                x, y = batch
                x = x.to(self.cfg.train.device, non_blocking=True)
                y = y.to(self.cfg.train.device, non_blocking=True)
                x, y = self._sanitize_batch_tensors(x, y)
                self.optimizer.zero_grad(set_to_none=True)

                if self.grad_scaler is not None:
                    with autocast("cuda"):
                        y_pred = self.model(x)
                        loss = self.loss_fn(y_pred, y)
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    y_pred = self.model(x)
                    loss = self.loss_fn(y_pred, y)
                    loss.backward()
                    self.optimizer.step()

                if self._scheduler_steps_per_batch(self.scheduler):
                    self.scheduler.step()

                batch_elements = y.numel()
                train_loss_sum += float(loss.detach().item()) * batch_elements
                train_num_elements += batch_elements
                running_train_loss = train_loss_sum / train_num_elements
                batch_progress.set_postfix_str(f"loss={running_train_loss:.4f} batch={batch_idx}/{self.num_batches}")
            batch_progress.close()

            if train_num_elements == 0:
                raise RuntimeError("Training loader produced zero elements.")
            train_loss = train_loss_sum / train_num_elements
            train_time_seconds = time.perf_counter() - train_start_time

            self.model.eval()
            val_start_time = time.perf_counter()
            val_loss, validation_metrics = run_validation_loop(
                loader=val_loader,
                model=self.model,
                loss_fn=self.loss_fn,
                device=self.cfg.train.device,
                return_metrics=should_compute_metrics,
            )
            val_time_seconds = time.perf_counter() - val_start_time

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif not self._scheduler_steps_per_batch(self.scheduler):
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            stop_early = False
            if validation_metrics is not None:
                last_validation_metrics = validation_metrics
                displayed_val_metrics = {
                    "mae": f"{validation_metrics['mae_mean']:.4f}",
                    "rmse": f"{validation_metrics['rmse_mean']:.4f}",
                    "mse": f"{validation_metrics['mse_mean']:.4f}",
                }
                if early_stopping["enabled"]:
                    current_val_mae = float(validation_metrics["mae_mean"])
                    if current_val_mae < early_stopping["best_val_mae"]:
                        early_stopping["best_val_mae"] = current_val_mae
                        early_stopping["best_epoch"] = epoch
                        early_stopping["epochs_without_improvement"] = 0
                        torch.save(self.model.state_dict(), early_stopping["best_ckpt_path"])
                    else:
                        early_stopping["epochs_without_improvement"] += 1
                        if early_stopping["epochs_without_improvement"] >= early_stopping["patience"]:
                            early_stopping["stopped_epoch"] = epoch
                            stop_early = True

            postfix = (
                f"tr={train_loss:.4f} "
                f"mae={displayed_val_metrics['mae']} "
                f"rmse={displayed_val_metrics['rmse']} "
                f"mse={displayed_val_metrics['mse']} "
                f"va={val_loss:.4f} "
                f"lr={lr:.1e}"
            )
            progress_bar.set_postfix_str(postfix)

            metrics = {
                "Epoch": epoch,
                "train/loss": train_loss,
                "model/Grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "model/Learning_rate": lr,
                "timing/train_epoch_seconds": train_time_seconds,
                "timing/val_epoch_seconds": val_time_seconds,
            }
            if validation_metrics is not None:
                metrics.update({
                    "validation/MAE_mean": validation_metrics["mae_mean"],
                    "validation/RMSE_mean": validation_metrics["rmse_mean"],
                    "validation/MSE_mean": validation_metrics["mse_mean"],
                })
            metrics.update({
                "validation/loss": val_loss,
                "validation/persistence_MAE_mean": persistence_metrics["mae_mean"],
                "validation/persistence_RMSE_mean": persistence_metrics["rmse_mean"],
                "validation/persistence_MSE_mean": persistence_metrics["mse_mean"],
                "validation/persistence_loss": persistence_loss,
            })
            wandb_logger.log_metrics(metrics, step=epoch)

            if stop_early:
                break

        if last_validation_metrics is None:
            _, last_validation_metrics = run_validation_loop(
                loader=val_loader,
                model=self.model,
                loss_fn=self.loss_fn,
                device=self.cfg.train.device,
                return_metrics=True,
            )

        if early_stopping["enabled"] and early_stopping["stopped_epoch"] is not None:
            print(
                f"Early stopping triggered at epoch {early_stopping['stopped_epoch'] + 1}. "
                f"Best validation MAE={early_stopping['best_val_mae']:.6f} "
                f"at epoch {early_stopping['best_epoch'] + 1}."
            )

        print("Final validation metrics:")
        final_val_loss = last_validation_metrics["mae_mean"]
        best_ckpt_path = early_stopping["best_ckpt_path"]
        if early_stopping["enabled"] and best_ckpt_path is not None and osp.exists(best_ckpt_path):
            self.model.load_state_dict(
                torch.load(best_ckpt_path, map_location=self.cfg.train.device, weights_only=False)
            )
            final_val_loss = early_stopping["best_val_mae"]
            print(
                f"Uploading best checkpoint from epoch {early_stopping['best_epoch'] + 1} "
                f"(validation MAE={early_stopping['best_val_mae']:.6f})."
            )

        # # after training save model
        wandb_logger.upload_and_remove_model(self.model, final_val_loss)
        if best_ckpt_path is not None and osp.exists(best_ckpt_path):
            os.remove(best_ckpt_path)
        self._best_ckpt_path = None
   
        # Finish logger
        wandb_logger.finish()
        
        # Clean up GPU memory
        if "cuda" in str(self.cfg.train.device):
            cleanup_gpu_memory(self.cfg.train.device)


def main(cfg: Dict[str, Any], add_agent: bool = False) -> None:
    # Train the model
    trainer = TrainModel(cfg, add_agent)
    try:
        trainer.train()
    except RuntimeError as err:
        if _is_cuda_oom_error(err):
            # OOM tracebacks can keep frame locals (and GPU tensors) alive across runs.
            _clear_exception_traceback_frames(err)
        raise
    finally:
        # Ensure sweeps release model/optimizer references even on failure/interruption.
        for attr in ("model", "optimizer", "scheduler", "loss_fn", "grad_scaler"):
            if hasattr(trainer, attr):
                try:
                    setattr(trainer, attr, None)
                except Exception:
                    pass

        best_ckpt_path = getattr(trainer, "_best_ckpt_path", None)
        if best_ckpt_path is not None and osp.exists(best_ckpt_path):
            try:
                os.remove(best_ckpt_path)
            except OSError:
                pass

        gc.collect()

        # Make sure no run remains open if train exits unexpectedly.
        if wandb.run is not None:
            wandb.finish()

        cleanup_device = None
        if hasattr(trainer, "cfg") and trainer.cfg is not None:
            try:
                cleanup_device = str(trainer.cfg.train.device)
            except Exception:
                cleanup_device = None
        cleanup_gpu_memory(cleanup_device)

@app.command()
def run_training(
    train_cfg_name: str = typer.Argument(
        ...,
        help="Training YAML path (for example: hydra_config.yaml).",
    )
) -> None:
    with initialize(version_base="1.1", config_path="../../config_train"):
        cfg = compose(config_name=train_cfg_name)
    wandb.init(project=cfg.wandb.project_name)
    main(cfg)

@app.command()
def run_sweep() -> None:
    with initialize(version_base="1.1", config_path="../../config_train"):
        cfg = compose(config_name="hydra_config.yaml")

    sweep_cfg = generate_sweep_configuration(cfg)

    print("Sweep configuration:")
    print(OmegaConf.to_yaml(sweep_cfg))

    sweep_id = wandb.sweep(sweep=sweep_cfg, project=cfg.wandb.project_name)

    def sweep_train():
        with initialize(version_base="1.1", config_path="../../config_train"):
            cfg = compose(config_name="hydra_config.yaml")
        main(cfg)

    wandb.agent(sweep_id, function=sweep_train)

@app.command()
def add_sweep_agent(sweep_id: str) -> None:
    def sweep_train():
        main(None, add_agent=True)

    wandb.agent(sweep_id, function=sweep_train)

if __name__ == "__main__":
    app()
