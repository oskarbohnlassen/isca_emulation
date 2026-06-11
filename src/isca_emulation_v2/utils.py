import os
import gc
import math
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import yaml
from omegaconf import OmegaConf, open_dict
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os.path as osp
import pickle
import subprocess
import time
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader


def add_data_cfg_from_manifest(cfg):
    processed_data_name = cfg.path_to_processed_data

    dataset_dir = Path("data/processed") / processed_data_name

    manifest_path = dataset_dir / "manifest.yaml"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}

    data_cfg = manifest['data']

    with open_dict(cfg):
        cfg.data = OmegaConf.create(data_cfg)

    return cfg

def get_data_attributes_cnn(cfg, data_point):
    """Get the data attributes for the model."""

    # Dataset items are typically (x, y); unwrap x when needed.
    x = data_point[0]
    y = data_point[1]

    # Supports both 2D CNN data [B,C,H,W] and 3D CNN data [B,C,L,H,W].
    channel_axis = -4 if x.dim() >= 5 else -3
    in_channels = int(x.shape[channel_axis])
    out_channels = int(y.shape[channel_axis])
    if x.dim() >= 5:
        grid_depth = int(x.shape[-3])
    else:
        grid_depth = None
    grid_height = int(x.shape[-2])
    grid_width = int(x.shape[-1])

    with open_dict(cfg):            # ← unlock
        # Add to cfg.data
        cfg.data.in_channels = in_channels
        cfg.data.out_channels = out_channels
        if grid_depth is not None:
            cfg.data.grid_depth = grid_depth
        cfg.data.grid_height = grid_height
        cfg.data.grid_width = grid_width
    
    return cfg


def get_data_attributes_gnn(cfg, data_point, edge_attr: torch.Tensor):
    """Get GNN data attributes from a pre-batched (x, y) sample and static edge features."""
    x = data_point[0]
    y = data_point[1]

    in_channels = int(x.shape[-1])
    out_channels = int(y.shape[-1])
    num_nodes = int(x.shape[-2])
    num_edge_features = int(edge_attr.shape[-1]) if edge_attr is not None else 0

    with open_dict(cfg):
        cfg.data.in_channels = in_channels
        cfg.data.out_channels = out_channels
        cfg.data.num_nodes = num_nodes
        cfg.data.num_edge_features = num_edge_features

    return cfg


def get_data_attributes_mesh_gnn(cfg, data_point, dataset):
    x = data_point[0]
    y = data_point[1]

    grid_shape = dataset.grid_static["shape"]
    grid_node_features = dataset.grid_static["node_features"]
    mesh_node_features = dataset.mesh_static["node_features"]
    g2m_edge_features = dataset.grid2mesh["edge_features"]
    mesh_edge_features = dataset.mesh_graph["edge_features"]
    m2g_edge_features = dataset.mesh2grid["edge_features"]

    with open_dict(cfg):
        cfg.data.in_channels = int(x.shape[-1])
        cfg.data.out_channels = int(y.shape[-1])
        cfg.data.num_nodes = int(x.shape[-2])
        cfg.data.grid_height = int(grid_shape[0])
        cfg.data.grid_width = int(grid_shape[1])
        cfg.data.grid_node_feature_dim = int(grid_node_features.shape[-1])
        cfg.data.mesh_num_nodes = int(mesh_node_features.shape[0])
        cfg.data.mesh_node_feature_dim = int(mesh_node_features.shape[-1])
        cfg.data.g2m_edge_feature_dim = int(g2m_edge_features.shape[-1])
        cfg.data.mesh_edge_feature_dim = int(mesh_edge_features.shape[-1])
        cfg.data.m2g_edge_feature_dim = int(m2g_edge_features.shape[-1])

    return cfg


def get_data_attributes(cfg, data_point):
    """Get the data attributes for the model."""
    # Get the number of nodes and edges
    num_nodes = data_point.x.shape[0]
    num_edges = data_point.edge_attr.shape[0]

    # Get the number of features
    num_node_features = data_point.x.shape[1]
    num_edge_features = data_point.edge_attr.shape[1]

    # Get the target dimension (edge-level flow)
    if hasattr(data_point, 'y_flow') and data_point.y_flow is not None:
        target_dim = data_point.y_flow.shape[0]
    else:
        raise AttributeError("Data object lacks 'y_flow' attribute required for target_dim calculation")

    # Calculate feature dimensions separately for clean parameter passing
    coordinate_dim = 0
    if hasattr(data_point, 'coordinates') and data_point.coordinates is not None:
        coordinate_dim = data_point.coordinates.shape[1]  # Usually 2 for [x, y]
    
    hop_distance_dim = 0
    if hasattr(data_point, 'hop_distances') and data_point.hop_distances is not None:
        hop_distance_dim = data_point.hop_distances.shape[1]
    
    euclidean_distance_dim = 0
    if hasattr(data_point, 'euclidean_distances') and data_point.euclidean_distances is not None:
        euclidean_distance_dim = data_point.euclidean_distances.shape[1]

    with open_dict(cfg):            # ← unlock
        # Add to cfg.data
        cfg.data.num_nodes = num_nodes
        cfg.data.num_edges = num_edges
        cfg.data.num_node_features = num_node_features
        cfg.data.num_edge_features = num_edge_features
        cfg.data.target_dim = target_dim
        # Add separate feature dimensions for encoders
        cfg.data.coordinate_dim = coordinate_dim
        cfg.data.hop_distance_dim = hop_distance_dim
        cfg.data.euclidean_distance_dim = euclidean_distance_dim
        cfg.data.num_mlp_features = num_node_features*num_nodes + num_edge_features*num_edges + coordinate_dim*num_nodes + hop_distance_dim*num_nodes + euclidean_distance_dim*num_nodes
    
    print(f"Feature dimensions: base_node={num_node_features}, coordinate={coordinate_dim}, hop={hop_distance_dim}, euclidean={euclidean_distance_dim}, edge={num_edge_features}")
    return cfg


def load_data_split(dataset_name: str, split: str):
    """
    Centralized function to load a data split.
    
    Args:
        dataset_name: Name of the dataset
        split: Split to load ('train', 'val', 'test')
        
    Returns:
        Loaded dataset
    """
    processed_dir = "data/processed"
    processed_data_path = osp.join(processed_dir, dataset_name)
    data_path = osp.join(processed_data_path, f"{split}.pt")
    
    if not osp.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} does not exist.")
    
    return torch.load(data_path, weights_only=False)


def load_scalers(dataset_name: str) -> Dict:
    """
    Centralized function to load scalers.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary containing scalers
    """
    processed_dir = "data/processed"
    processed_data_path = osp.join(processed_dir, dataset_name)
    scalers_path = osp.join(processed_data_path, "scalers.pkl")
    
    if not osp.exists(scalers_path):
        raise FileNotFoundError(f"Scalers file {scalers_path} does not exist.")
    
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
    
    return scalers


def add_mlp_x(data):
    """
    Centralized function to add mlp_x to data for MLP models.
    Includes all available features: node features, edge features, and spatial features.
    
    Args:
        data: Graph data object
        
    Returns:
        Data object with mlp_x added
    """
    # Start with basic node and edge features
    N, NF = data.x.size()
    E, EF = data.edge_attr.size()

    node_flat = data.x.view(N * NF)
    edge_flat = data.edge_attr.reshape(E * EF)
    
    # Collect all feature tensors
    feature_tensors = [node_flat, edge_flat]
    
    # Concatenate all available features
    mlp_x = torch.cat(feature_tensors, dim=0)
    data.mlp_x = mlp_x.unsqueeze(0)
    
    return data


def apply_noise_threshold(predictions: np.ndarray, noise_threshold: float) -> Tuple[np.ndarray, Dict]:
    """
    Apply noise thresholding to predictions.
    
    Args:
        predictions: Array of predictions
        noise_threshold: Threshold below which values are zeroed
        
    Returns:
        Tuple of (cleaned predictions, noise statistics)
    """
    total_values = predictions.size
    zeroed_values = np.sum(np.abs(predictions) < noise_threshold)
    zeroing_percentage = (zeroed_values / total_values) * 100
    
    predictions_clean = np.where(np.abs(predictions) < noise_threshold, 0, predictions)
    
    noise_stats = {
        'zeroing_percentage': float(zeroing_percentage),
        'total_values': int(total_values),
        'zeroed_values': int(zeroed_values)
    }
    
    return predictions_clean, noise_stats


def compute_metrics(targets: np.ndarray, 
                   predictions: np.ndarray, 
                   multioutput: str = 'raw_values') -> Dict:
    """
    Compute standardized metrics for model evaluation.
    
    This is the CENTRAL function that all models must use for metric calculation.
    
    Args:
        targets: True values in original units (unscaled) [n_samples, n_outputs]
        predictions: Predicted values in original units (unscaled) [n_samples, n_outputs]  
        multioutput: How to handle multiple outputs ('raw_values', 'uniform_average')
        
    Returns:
        Dict containing all standardized metrics
    """
    
    # Validate inputs
    if targets.shape != predictions.shape:
        raise ValueError(f"Targets shape {targets.shape} != predictions shape {predictions.shape}")
    
    if len(targets.shape) != 2:
        raise ValueError(f"Expected 2D arrays, got targets shape {targets.shape}")
        
    # Check for invalid values
    if not np.all(np.isfinite(targets)):
        print("Warning: Non-finite values found in targets")
        targets = np.where(np.isfinite(targets), targets, 0)
        
    if not np.all(np.isfinite(predictions)):
        print("Warning: Non-finite values found in predictions")
        predictions = np.where(np.isfinite(predictions), predictions, 0)
    
    # Calculate per-output metrics
    mae_scores = mean_absolute_error(targets, predictions, multioutput=multioutput)
    mse_scores = mean_squared_error(targets, predictions, multioutput=multioutput)
    rmse_scores = np.sqrt(mse_scores)
    r2_scores = r2_score(targets, predictions, multioutput=multioutput)
    
    # Calculate summary statistics
    metrics = {
        # Per-output scores
        'mae_scores': mae_scores,
        'mse_scores': mse_scores,
        'rmse_scores': rmse_scores, 
        'r2_scores': r2_scores,
        
        # Summary statistics
        'mae_mean': float(mae_scores.mean()),
        'mae_std': float(mae_scores.std()),
        'mse_mean': float(mse_scores.mean()),
        'mse_std': float(mse_scores.std()),
        'rmse_mean': float(rmse_scores.mean()),
        'rmse_std': float(rmse_scores.std()),
        'r2_mean': float(r2_scores.mean()),
        'r2_std': float(r2_scores.std()),
    }
    
    return metrics


def compute_validation_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    metric_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Accumulate streaming validation stats on torch tensors."""
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_pred {tuple(y_pred.shape)} != y_true {tuple(y_true.shape)}")

    if metric_state is None:
        metric_state = {
            "abs_error_sum": torch.zeros((), device=y_pred.device, dtype=torch.float64),
            "sq_error_sum": torch.zeros((), device=y_pred.device, dtype=torch.float64),
            "num_elements": torch.zeros((), device=y_pred.device, dtype=torch.int64),
        }

    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    err = (y_pred - y_true).to(torch.float64)
    metric_state["abs_error_sum"] += err.abs().sum()
    metric_state["sq_error_sum"] += err.square().sum()
    metric_state["num_elements"] += err.numel()
    return metric_state


def finalize_validation_metrics(
    metric_state: Dict[str, torch.Tensor],
    verbose: bool = False,
) -> Dict[str, float]:
    """Finalize MAE/MSE/RMSE from accumulated streaming stats."""
    num_elements = int(metric_state["num_elements"].item())
    if num_elements <= 0:
        raise ValueError("Cannot finalize validation metrics with zero elements.")

    mae_mean = float((metric_state["abs_error_sum"] / num_elements).item())
    mse_mean = float((metric_state["sq_error_sum"] / num_elements).item())
    rmse_mean = float(math.sqrt(mse_mean))

    if verbose:
        print(f"MAE mean: {mae_mean:.4f}")
        print(f"MSE mean: {mse_mean:.4f}")
        print(f"RMSE mean: {rmse_mean:.4f}")

    return {
        "mae_mean": mae_mean,
        "mse_mean": mse_mean,
        "rmse_mean": rmse_mean,
    }


def run_validation_loop(
    loader,
    model,
    loss_fn,
    device: str,
    return_metrics: bool = False,
    batch_hook: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None] | None = None,
) -> tuple[float, Dict[str, float] | None]:
    """Run a generic validation pass and optionally return MAE/MSE/RMSE metrics."""
    val_loss_sum = 0.0
    val_num_elements = 0
    metric_state = None

    with torch.inference_mode():
        for batch in loader:
            x_val, y_val = batch
            x_val = x_val.to(device, non_blocking=True)
            y_val = y_val.to(device, non_blocking=True)
            x_val = torch.nan_to_num(x_val, nan=0.0, posinf=0.0, neginf=0.0)
            y_val = torch.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
            y_pred = model(x_val)
            val_batch_loss = loss_fn(y_pred, y_val)
            batch_elements = y_val.numel()
            val_loss_sum += float(val_batch_loss.detach().item()) * batch_elements
            val_num_elements += batch_elements

            if return_metrics:
                metric_state = compute_validation_metrics(
                    y_pred=y_pred,
                    y_true=y_val,
                    metric_state=metric_state,
                )

            if batch_hook is not None:
                batch_hook(x_val, y_val, y_pred)

    val_loss = val_loss_sum / val_num_elements
    if return_metrics:
        return val_loss, finalize_validation_metrics(metric_state)
    return val_loss, None


def get_validation_metrics_per_output(
    y_pred_np: np.ndarray,
    y_true_np: np.ndarray,
    cfg):
    
    """
    Compute per-output validation metrics with unscaled data.
    """
    
    # Check if input is 4D (N, C, Lat, Lon) or generally > 2D
    if y_pred_np.ndim > 2:
        # Flatten for standard metrics compatibility (expects 2D)
        # Reshape to (N, -1) to preserve batch dimension but flatten features
        y_pred_flat = y_pred_np.reshape(y_pred_np.shape[0], -1)
        y_true_flat = y_true_np.reshape(y_true_np.shape[0], -1)
        
        metrics = compute_metrics(y_true_flat, y_pred_flat)
        
        # Calculate spatial metrics if 4D (N, C, H, W)
        # The user requested "loss per lon/lat combination"
        if y_pred_np.ndim == 4:
            # Calculate MAE per spatial location averaging over N and C
            # Input shape: (N, C, H, W) -> Output shape: (H, W)
            spatial_mae = np.mean(np.abs(y_true_np - y_pred_np), axis=(0, 1))
            metrics['spatial_mae'] = spatial_mae
            metrics['spatial_mae_max'] = np.max(spatial_mae)
            metrics['spatial_mae_mean_check'] = np.mean(spatial_mae)
    else:
        metrics = compute_metrics(y_true_np, y_pred_np)

    # print some stats about the metrics for mae and std, rmse and std, r2 and std
    print(f"MAE mean: {metrics['mae_mean']:.2f} ± {metrics['mae_std']:.2f}")
    if 'spatial_mae_max' in metrics:
        print(f"Spatial MAE max: {metrics['spatial_mae_max']:.2f}")
    print(f"RMSE mean: {metrics['rmse_mean']:.2f} ± {metrics['rmse_std']:.2f}")
    print(f"R2 mean: {metrics['r2_mean']:.2f} ± {metrics['r2_std']:.2f}")
        
    return metrics


def calculate_batch_memory_mb(train_data, batch_size):
    """
    Calculate actual memory usage of a batch by examining tensor sizes.
    
    Args:
        train_data: Training dataset to sample from
        batch_size: Training batch size
        
    Returns:
        Batch memory in MB (float)
    """
    # Calculate actual batch memory by counting tensor elements and dtypes
    def calculate_tensor_memory_mb(tensor):
        if tensor is None:
            return 0.0
        element_size = tensor.element_size()  # bytes per element (e.g., 4 for float32)
        num_elements = tensor.numel()  # total number of elements
        return (element_size * num_elements) / (1024 * 1024)  # Convert to MB

    def accumulate_tensor(name: str, tensor: torch.Tensor | None, total: float) -> float:
        if tensor is None:
            return total
        mem = calculate_tensor_memory_mb(tensor)
        print(f"    {name}: {tuple(tensor.shape)} = {mem:.1f}MB")
        return total + mem

    # For pre-batched iterable datasets (CNN sharded path), use emitted batch directly.
    sample_item = next(iter(train_data))
    is_pair_of_tensors = (
        isinstance(sample_item, (tuple, list))
        and len(sample_item) == 2
        and torch.is_tensor(sample_item[0])
        and torch.is_tensor(sample_item[1])
    )
    if is_pair_of_tensors:
        x_batch = sample_item[0]
        y_batch = sample_item[1]
        is_cnn_prebatched = x_batch.dim() >= 4 and y_batch.dim() >= 4
        is_gnn_prebatched = x_batch.dim() == 3 and y_batch.dim() == 3
        if not (is_cnn_prebatched or is_gnn_prebatched):
            is_pair_of_tensors = False

    if is_pair_of_tensors:
        batch_memory_mb = 0.0
        print("  Batch tensor breakdown:")
        batch_memory_mb = accumulate_tensor("x tensor", x_batch, batch_memory_mb)
        batch_memory_mb = accumulate_tensor("y tensor", y_batch, batch_memory_mb)
        print(f"    Total batch memory: {batch_memory_mb:.1f}MB")
        return batch_memory_mb

    # For map-style datasets, build one batch with the appropriate DataLoader.
    if hasattr(sample_item, "edge_index"):
        temp_loader = PyGDataLoader(train_data, batch_size=batch_size, shuffle=False)
    else:
        temp_loader = TorchDataLoader(train_data, batch_size=batch_size, shuffle=False)
    sample_batch = next(iter(temp_loader))
    
    batch_memory_mb = 0.0
    
    print("  Batch tensor breakdown:")
    
    # Calculate memory for all tensors in the batch
    if hasattr(sample_batch, 'x') and sample_batch.x is not None:
        batch_memory_mb = accumulate_tensor("x tensor", sample_batch.x, batch_memory_mb)
    
    if hasattr(sample_batch, 'edge_index') and sample_batch.edge_index is not None:
        batch_memory_mb = accumulate_tensor("edge_index", sample_batch.edge_index, batch_memory_mb)
    
    if hasattr(sample_batch, 'edge_attr') and sample_batch.edge_attr is not None:
        batch_memory_mb = accumulate_tensor("edge_attr", sample_batch.edge_attr, batch_memory_mb)
    
    if hasattr(sample_batch, 'y_flow') and sample_batch.y_flow is not None:
        batch_memory_mb = accumulate_tensor("y_flow", sample_batch.y_flow, batch_memory_mb)
    
    if hasattr(sample_batch, 'y_intermediate') and sample_batch.y_intermediate is not None:
        batch_memory_mb = accumulate_tensor("y_intermediate", sample_batch.y_intermediate, batch_memory_mb)
    
    if hasattr(sample_batch, 'coordinates') and sample_batch.coordinates is not None:
        batch_memory_mb = accumulate_tensor("coordinates", sample_batch.coordinates, batch_memory_mb)
    
    if hasattr(sample_batch, 'hop_distances') and sample_batch.hop_distances is not None:
        batch_memory_mb = accumulate_tensor("hop_distances", sample_batch.hop_distances, batch_memory_mb)
    
    if hasattr(sample_batch, 'euclidean_distances') and sample_batch.euclidean_distances is not None:
        batch_memory_mb = accumulate_tensor("euclidean_distances", sample_batch.euclidean_distances, batch_memory_mb)
    
    if hasattr(sample_batch, 'batch') and sample_batch.batch is not None:
        batch_memory_mb = accumulate_tensor("batch indices", sample_batch.batch, batch_memory_mb)
    
    if hasattr(sample_batch, 'ptr') and sample_batch.ptr is not None:
        batch_memory_mb = accumulate_tensor("ptr", sample_batch.ptr, batch_memory_mb)
    
    print(f"    Total batch memory: {batch_memory_mb:.1f}MB")
    
    # Clean up
    del sample_batch
    
    return batch_memory_mb


def estimate_model_memory_mb(num_parameters, train_data, batch_size):
    """
    Estimate GPU memory needed for training in MB.
    
    Args:
        num_parameters: Actual number of model parameters
        train_data: Training dataset to sample from
        batch_size: Training batch size
        
    Returns:
        Estimated memory usage in MB (int)
    """
    # Calculate batch memory using separate function
    batch_memory_mb = calculate_batch_memory_mb(train_data, batch_size)
    
    overhead_factor = 0.5
    
    # Calculate parameter-related memory
    param_mb = num_parameters * 4 / (1024**2)
    grad_mb = param_mb * 2 
    optim_mb = param_mb * 2
    act_mb = param_mb * 3

    # Fixed cost for CUDA context + cuBLAS/cuDNN workspaces
    fixed_mb = 500

    # Variable allocator/fragmentation overhead: e.g. 25% of everything above
    var_overhead_mb = overhead_factor * (
        param_mb + grad_mb + optim_mb + batch_memory_mb + act_mb
    )

    total = (
        param_mb + grad_mb + optim_mb +
        batch_memory_mb + act_mb +
        fixed_mb + var_overhead_mb
    )
    
    print(f"Memory estimate: {round(total):.0f}MB")
    print(f"  - Parameters: {param_mb:.0f}MB")
    print(f"  - Gradients: {grad_mb:.0f}MB") 
    print(f"  - Optimizer: {optim_mb:.0f}MB")
    print(f"  - Batch data: {batch_memory_mb:.0f}MB")
    print(f"  - Activations: {act_mb:.0f}MB")
    print(f"  - Fixed CUDA overhead: {fixed_mb:.0f}MB")
    print(f"  - Variable overhead ({overhead_factor*100:.0f}%): {var_overhead_mb:.0f}MB")
    
    return round(total)


def _check_gpu_memory(required_memory_mb, verbose=True):
    """Helper function to check GPU memory availability and utilization."""
    
    num_gpus = torch.cuda.device_count()
    
    # Get GPU memory usage and utilization from nvidia-smi
    result = subprocess.run([
        'nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,name', 
        '--format=csv,nounits,noheader'
    ], capture_output=True, text=True, check=True)
    
    # Parse nvidia-smi output
    nvidia_smi_gpus = {}
    for line in result.stdout.strip().split('\n'):
        if line:
            parts = line.split(', ')
            gpu_id = int(parts[0])
            used_mb = int(parts[1])
            total_mb = int(parts[2])
            utilization = int(parts[3])  # GPU utilization percentage
            gpu_name = parts[4]
            nvidia_smi_gpus[gpu_id] = {
                'used_mb': used_mb,
                'total_mb': total_mb,
                'name': gpu_name,
                'utilization': utilization
            }

    # Create mapping between PyTorch cuda indices and nvidia-smi indices by matching GPU names
    suitable_gpus = []
    
    for cuda_idx in range(num_gpus):
        # Get PyTorch GPU name
        torch_gpu_name = torch.cuda.get_device_name(cuda_idx)
        
        # Find matching nvidia-smi GPU by name
        nvidia_gpu = None
        for smi_idx, smi_info in nvidia_smi_gpus.items():
            if torch_gpu_name in smi_info['name'] or smi_info['name'] in torch_gpu_name:
                nvidia_gpu = smi_info
                break
        
        if nvidia_gpu:
            free_mb = nvidia_gpu['total_mb'] - nvidia_gpu['used_mb']
            free_gb = free_mb / 1024
            total_gb = nvidia_gpu['total_mb'] / 1024
            utilization = nvidia_gpu['utilization']
            
            # Check if GPU has enough memory (with 200MB safety buffer)
            memory_sufficient = free_mb > (required_memory_mb + 200)
            
            if verbose:
                status = "✓" if memory_sufficient else "✗ (insufficient)"
                util_info = f", {utilization}% util" 
                print(f"cuda:{cuda_idx} ({torch_gpu_name}): {free_gb:.1f}GB free / {total_gb:.1f}GB total{util_info} {status}")
            
            if memory_sufficient:
                suitable_gpus.append((cuda_idx, free_mb, nvidia_gpu['total_mb'], torch_gpu_name, utilization))
        else:
            if verbose:
                print(f"cuda:{cuda_idx} ({torch_gpu_name}): Could not match with nvidia-smi")
    
    return suitable_gpus


def select_best_gpu():
    """
    Select GPU with sufficient memory and lowest utilization for training.
    
    Args:
        required_memory_mb: Required memory in MB
        
    Returns:
        Device string (e.g., 'cuda:0', 'cuda:1')
    """
    
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs detected"
    
    print(f"Detected {num_gpus} GPU(s), preferring idle GPUs and then the card with the most VRAM...")
    suitable_gpus = _check_gpu_memory(0, verbose=True)
    if not suitable_gpus:
        raise RuntimeError("No CUDA device has enough free memory for even a fallback allocation.")
    
    # Prefer idle GPUs first. Among idle GPUs, prefer the card with the most total VRAM,
    # then the most currently free memory. If no GPU is idle, fall back to lowest utilization.
    # suitable_gpus format: (cuda_idx, free_mb, total_mb, gpu_name, utilization)
    suitable_gpus.sort(key=lambda x: (x[4] != 0, x[4], -x[2], -x[1]))
    
    best_cuda_idx, _, _, gpu_name, utilization = suitable_gpus[0]
    
    selected_device = f'cuda:{best_cuda_idx}'
    torch.cuda.set_device(best_cuda_idx)
    
    if utilization == 0:
        print(f"Selected device: {selected_device} ({gpu_name}) - idle GPU ✓")
    elif utilization <= 10:
        print(f"Selected device: {selected_device} ({gpu_name}) - low utilization ({utilization}%) ✓")
    else:
        print(f"Selected device: {selected_device} ({gpu_name}) - sharing GPU with {utilization}% utilization")
    
    return selected_device


def cleanup_gpu_memory(device: str | int | None = None):
    """Best-effort GPU memory cleanup for a target CUDA device or all visible devices."""
    gc.collect()

    if not torch.cuda.is_available():
        return

    target_devices: list[int]
    if isinstance(device, int):
        target_devices = [device]
    elif isinstance(device, str) and device.startswith("cuda:"):
        try:
            target_devices = [int(device.split(":", maxsplit=1)[1])]
        except (TypeError, ValueError):
            target_devices = []
    elif device == "cuda":
        target_devices = [torch.cuda.current_device()]
    else:
        target_devices = list(range(torch.cuda.device_count()))

    num_devices = torch.cuda.device_count()
    target_devices = [idx for idx in target_devices if 0 <= idx < num_devices]
    if not target_devices:
        return

    print("Cleaning up GPU memory...")
    for device_idx in target_devices:
        with torch.cuda.device(device_idx):
            torch.cuda.empty_cache()
            # Frees cached IPC memory held by CUDA allocator in long-lived processes.
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

    for device_idx in target_devices:
        try:
            torch.cuda.synchronize(device_idx)
        except RuntimeError:
            pass

    print("GPU memory cleanup completed")
