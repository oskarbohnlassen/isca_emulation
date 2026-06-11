import wandb
from omegaconf import OmegaConf
import os.path as osp
from typing import Dict, Any, Callable
import datetime
import os
import re
import torch
from dotenv import load_dotenv
import numpy as np
import tempfile

class WandbLogger:
    def __init__(self, cfg: Dict[str, Any], add_agent: bool = False, project_name: str | None = None) -> None:
        self.run = wandb.init(project=project_name) if project_name else wandb.init()
        self.add_agent = add_agent
        self.cfg = cfg

    def _append_model_type_to_run_name(self) -> None:
        """Append model type to auto-generated run names for clearer legends."""
        append_enabled = self.run.config.get("wandb.append_model_type_to_name", True)
        if isinstance(append_enabled, str):
            append_enabled = append_enabled.strip().lower() in {"1", "true", "yes", "on"}
        if not bool(append_enabled):
            return

        model_type = self.run.config.get("model.model_type")
        if model_type is None:
            return

        model_type_str = str(model_type).strip()
        if not model_type_str:
            return

        current_name = str(self.run.name or self.run.id)
        suffix = f" ({model_type_str})"

        # Keep names readable and avoid duplicate model tags.
        if current_name.endswith(suffix) or f"({model_type_str})" in current_name:
            return

        self.run.name = f"{current_name}{suffix}"
        print(f"Updated wandb run name to: {self.run.name}")

    @staticmethod
    def _sanitize_tag(value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip().lower())
        cleaned = cleaned.strip("_.-")
        if len(cleaned) > 64:
            cleaned = cleaned[:64].rstrip("_.-")
        return cleaned

    @staticmethod
    def _extract_time_window(dataset_name: str) -> str | None:
        match = re.search(r"(\d+)\s*_to_\s*(\d+)", dataset_name)
        if not match:
            return None
        return f"{match.group(1)}_{match.group(2)}"

    @staticmethod
    def _infer_regime_label(*values: str) -> str:
        text = " ".join(v.lower() for v in values if v).strip()
        if "with_ssw" in text:
            return "with_ssw"
        if "no_ssw" in text:
            return "no_ssw"
        return "unknown_regime"

    def _apply_run_metadata(self) -> None:
        cfg_flat = dict(self.run.config)

        dataset_name = str(
            cfg_flat.get("data.output.dataset_name")
            or cfg_flat.get("path_to_processed_data")
            or ""
        ).strip()
        model_type = str(cfg_flat.get("model.model_type") or "").strip()
        exp_name = str(cfg_flat.get("data.raw.exp_folder_name") or "").strip()
        time_window = self._extract_time_window(dataset_name)
        regime = self._infer_regime_label(dataset_name, exp_name)

        explicit_group = cfg_flat.get("wandb.group")
        group = str(explicit_group).strip() if explicit_group else ""
        if not group:
            parts = [p for p in [model_type, dataset_name, f"window_{time_window}" if time_window else ""] if p]
            group = "|".join(parts)
        if group:
            try:
                self.run.group = group
            except Exception:
                pass

        tags = list(self.run.tags or ())
        existing = {str(tag).lower() for tag in tags}

        def add_tag(tag_value: str) -> None:
            if not tag_value:
                return
            cleaned = self._sanitize_tag(tag_value)
            if not cleaned:
                return
            if cleaned.lower() in existing:
                return
            tags.append(cleaned)
            existing.add(cleaned.lower())

        explicit_tags = cfg_flat.get("wandb.tags")
        if isinstance(explicit_tags, str):
            add_tag(explicit_tags)
        elif isinstance(explicit_tags, list):
            for tag in explicit_tags:
                add_tag(str(tag))

        if regime != "unknown_regime":
            add_tag(regime)
        if time_window:
            add_tag(f"window_{time_window}")
        if model_type:
            add_tag(f"model_{model_type}")
        if dataset_name:
            add_tag(f"dataset_{dataset_name}")

        try:
            self.run.tags = tuple(tags)
        except Exception:
            pass

        try:
            if dataset_name:
                self.run.summary["meta/dataset_name"] = dataset_name
            if time_window:
                self.run.summary["meta/time_window"] = time_window
            if regime != "unknown_regime":
                self.run.summary["meta/regime"] = regime
        except Exception:
            pass

        print(
            "W&B metadata "
            f"group={getattr(self.run, 'group', None)} "
            f"tags={list(getattr(self.run, 'tags', []))}"
        )

    def _prepare_clean_config(self):
        """Prepare config with all parameters as flat keys for maximum consistency."""
        if self.cfg is None:
            raise ValueError("cfg is None - add_agent is True")
        
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        
        def flatten_config(d, prefix=""):
            flat_config = {}
            for key, value in d.items():
                current_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    # Recursively flatten nested dicts
                    flat_config.update(flatten_config(value, current_key))
                elif isinstance(value, list):
                    if current_key.startswith("data."):
                        # Keep data lists (e.g. vars, split) in run config.
                        flat_config[current_key] = value
                    elif len(value) == 1:
                        flat_config[current_key] = value[0]
                    else:
                        # Skip non-data multi-value lists (treated as sweep options).
                        continue
                else:
                    # Add scalar values as flat keys
                    flat_config[current_key] = value
            
            return flat_config
        
        return flatten_config(cfg_dict)
    
    def _save_config_to_wandb(self, artifact_id):
        """Save the config to wandb as an artifact."""
        artifact_name = f"{artifact_id}_config"
        
        # Use temporary directory instead of repository
        with tempfile.TemporaryDirectory() as temp_dir:
            file_name = os.path.join(temp_dir, f"{artifact_name}.yaml")

            # Convert to dict format that can be directly used with config.update()
            if hasattr(self.cfg, 'to_container'):
                # It's an OmegaConf object
                cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
            else:
                # It's already a dict (wandb config)
                cfg_dict = dict(self.cfg)

            # Save as yaml using OmegaConf
            OmegaConf.save(OmegaConf.create(cfg_dict), file_name)

            artifact = wandb.Artifact(artifact_name, type="config")
            artifact.add_file(file_name)
            self.run.log_artifact(artifact)
            
            # File is automatically cleaned up when temp_dir context exits

    def _load_config_from_wandb(self, artifact_id):
        """Load the config from wandb as an artifact."""
        artifact_name = f"{artifact_id}_config"
        artifact = self.run.use_artifact(f"{artifact_name}:latest")
        artifact_dir = artifact.download()
        cfg_path = os.path.join(artifact_dir, f"{artifact_name}.yaml")
        cfg = OmegaConf.load(cfg_path)
        return cfg


    def _check_if_artifact_exists(self, artifact_id):
        """Check if the artifact exists in wandb."""
        artifact_name = f"{artifact_id}_config"

        api = wandb.Api()
        try:
            _ = api.artifact(f"{self.run.project}/{artifact_name}:latest")
            return True
        except wandb.errors.CommError:
            return False

    def _initialize_wandb(self, add_agent: bool = False) -> None:
        """Start a wandb run (handles sweeps automatically)."""

        # Load the .env file
        load_dotenv(".env")
        # Example: Access WANDB_API_KEY from the environment
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            raise ValueError("WANDB_API_KEY is not set. Please check your .env file.")
        print("WANDB_API_KEY loaded successfully.")

        # If WANDB_CACHE_DIR is not set, print default is used else print the set directory
        wandb_cache_dir = os.getenv("WANDB_CACHE_DIR")
        if not wandb_cache_dir:
            print("WANDB_CACHE_DIR is not set. Using default cache directory.")
        else:
            print(f"WANDB_CACHE_DIR is set to {wandb_cache_dir}.")

        # If WANDB_ARTIFACT_CACHE_SIZE is not set, print default is used else print the cache size
        wandb_artifact_cache_size = os.getenv("WANDB_ARTIFACT_CACHE_SIZE")
        if not wandb_artifact_cache_size:
            print("WANDB_ARTIFACT_CACHE_SIZE is not set. Using default cache size.")
        else:
            print(f"WANDB_ARTIFACT_CACHE_SIZE is set to {wandb_artifact_cache_size}.")

        # Handle config consistency
        if self.run.sweep_id is not None and (self.add_agent or self._check_if_artifact_exists(self.run.sweep_id)):
            artifact_id = self.run.sweep_id
            artifact_cfg = self._load_config_from_wandb(artifact_id)
            print(f"Loaded config from artifact {artifact_id}. Merging with wandb.config with new sweep parameters.")
            # Convert loaded config to flattened format for proper merging
            self.cfg = artifact_cfg  # Temporarily set for _prepare_clean_config
            flattened_artifact_cfg = self._prepare_clean_config()
            # Backfill only missing keys so sweep-assigned keys are never overwritten.
            # This avoids "locked by sweep" warnings for parameters controlled by the sweep.
            existing_keys = set(self.run.config.keys())
            missing_cfg = {
                key: value
                for key, value in flattened_artifact_cfg.items()
                if key not in existing_keys
            }
            self.run.config.update(missing_cfg)
            self.cfg = self.run.config

        else:
            print(f"No config found in wandb.")
            # Use the config from the repository and format it for wandb
            repository_config = self._prepare_clean_config()
            # Merge the config from the repo with the wandb.config that contains the sweep parameters
            self.run.config.update(repository_config)
            # Use sweep_id for sweeps, run_id for individual runs
            artifact_id = self.run.sweep_id if self.run.sweep_id is not None else self.run.id
        
        self.cfg = self.run.config
        self._apply_run_metadata()
        self._append_model_type_to_run_name()
        self._save_config_to_wandb(artifact_id)
                

    def _wandb_config_to_cfg(self):
        """
        Use wandb.config as the source of truth, but reconstruct nested structure for ML pipeline.
        """
        # Convert wandb.config back to your cfg structure
        wandb_config_dict = dict(self.cfg)

        # Reconstruct nested structure from flat keys
        nested_config = {}
        
        for flat_key, value in wandb_config_dict.items():
            parts = flat_key.split('.')
            current = nested_config
            
            # Navigate/create nested structure
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the final value
            current[parts[-1]] = value
        
        # Use wandb.config as the source of truth but in nested format
        self.cfg = OmegaConf.create(nested_config)

        return self.cfg

    def watch_model(self, model, log_graph=True):
        wandb.watch(model, log="all", log_graph=log_graph)

        # count params once and log
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.log({"model/num_parameters": n_params}, step=0)
    
    def log_metrics(self, metrics:dict, step:int):
        wandb.log(metrics, step=step)

    def upload_and_remove_model(self, model, val_loss):
        thr = self.cfg.wandb.save_model_threshold
        if val_loss >= thr:
            print(f"val_loss {val_loss:.4f} >= {thr}  →  not saving model")
            return

        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{self.cfg.model.model_type}_{ts}.pt"
        path = os.path.join("models", name)
        os.makedirs("models", exist_ok=True)

        torch.save(model.state_dict(), path)

        art = wandb.Artifact(name, type="model",
               metadata={"val_loss":val_loss})
        art.add_file(path)
        self.run.log_artifact(art)
        os.remove(path)
        print("Model uploaded & local copy deleted")

    def save_predictions(self, y_pred, y_true):
        """Save predictions and targets to wandb as an artifact and table for visualization."""
        
        # 1. Save as numpy arrays (artifact for analysis)
        with tempfile.TemporaryDirectory() as temp_dir:
            pred_path = osp.join(temp_dir, "predictions.npy")
            true_path = osp.join(temp_dir, "targets.npy")
            
            np.save(pred_path, y_pred)
            np.save(true_path, y_true)
            
            # Create artifact
            artifact = wandb.Artifact(
                name="final_predictions",
                type="predictions",
                description="Final model predictions and targets"
            )
            
            artifact.add_file(pred_path, name="predictions.npy")
            artifact.add_file(true_path, name="targets.npy")
            
            self.run.log_artifact(artifact)
        
        # 2. Create table for visualization
        # Sample validation samples but keep ALL targets for each selected sample
        num_samples, num_targets = y_pred.shape
        
        # Limit number of samples for table (but keep all targets per sample)
        max_samples_for_table = min(200, num_samples)  # ~200 samples * 261 targets = ~52k rows
        
        if num_samples > max_samples_for_table:
            selected_samples = np.random.choice(num_samples, max_samples_for_table, replace=False)
            selected_samples = np.sort(selected_samples)  # Keep some order
        else:
            selected_samples = np.arange(num_samples)
        
        # Create table data - all targets for each selected sample
        table_data = []
        for sample_idx in selected_samples:
            for target_idx in range(num_targets):
                y_true_val = y_true[sample_idx, target_idx]
                y_pred_val = y_pred[sample_idx, target_idx]
                residual = y_pred_val - y_true_val
                abs_residual = abs(residual)
                
                table_data.append([
                    int(sample_idx),      # sample_idx
                    int(target_idx),      # target_idx  
                    float(y_true_val),    # y_true
                    float(y_pred_val),    # y_pred
                    float(residual),      # residual
                    float(abs_residual)   # abs_residual
                ])
        
        # Create wandb table
        table = wandb.Table(
            data=table_data,
            columns=["sample_idx", "target_idx", "y_true", "y_pred", "residual", "abs_residual"]
        )
        
        wandb.log({"analysis/predictions_table": table})
        
        print(f"Saved predictions artifact: {num_samples} samples x {num_targets} targets")
        print(f"Saved predictions table: {len(table_data)} rows ({len(selected_samples)} samples x {num_targets} targets)")
    
    def finish(self):
        if self.run:
            self.run.finish()

    @classmethod
    def from_wandb_run(cls, wandb_run_path: str, project_name: str | None = None):
        import wandb
        from omegaconf import OmegaConf
        api = wandb.Api()
        run = api.run(wandb_run_path)
        # Get the config directly from the wandb run
        wandb_config = run.config
        wandb_config_dict = dict(wandb_config)
        # Reconstruct nested structure from flat keys (same as _wandb_config_to_cfg)
        nested_config = {}
        for flat_key, value in wandb_config_dict.items():
            if flat_key.startswith('_'):
                continue
            parts = flat_key.split('.')
            current = nested_config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        cfg = OmegaConf.create(nested_config)
        return cls(cfg=cfg, project_name=project_name)


def generate_sweep_configuration(cfg: Dict[str, Any]) -> Dict[str, Any]:

    sweep_name = cfg.wandb.sweep.sweep_name
    metric_name = cfg.wandb.sweep.metric_name
    goal = cfg.wandb.sweep.metric_goal
    method = cfg.wandb.sweep.method
    run_cap = cfg.wandb.sweep.run_cap
    
    sweep_parameters = {}

    # Recursively find list-type entries in the config
    def find_list_entries(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                # if key is dataset, skip
                if key == "data":
                    continue
                current_path = f"{prefix}.{key}" if prefix else key
                find_list_entries(value, current_path)
        elif isinstance(node, list):
            # Only add multi-value lists as sweep parameters
            if len(node) > 1:
                sweep_parameters[prefix] = {"values": node}

    # Start processing the configuration
    find_list_entries(OmegaConf.to_container(cfg, resolve=True))

    # Create the sweep configuration
    sweep_configuration = {
        "method": method,
        "name": sweep_name,
        "metric": {"goal": goal, "name": metric_name},
        "parameters": sweep_parameters,
        "run_cap": run_cap,
    }

        # Check if the sweep_cfg contains any lists
    if not sweep_configuration["parameters"]:
        raise ValueError(
            "No list-type parameters found in the configuration. Please add at least one list-type parameter to run a sweep."
        )

    return sweep_configuration
