import torch
from omegaconf import OmegaConf
import os.path as osp
import torch.nn.functional as F

class MultiStageLoss(torch.nn.Module):
    """
    Multi-stage loss that computes weighted loss across all intermediate stages.
    Uses exponential weighting to emphasize later stages (equilibrium) more heavily.
    """
    def __init__(self, base_loss_fn, num_stages: int, equilibrium_weight: float = 4.0):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.num_stages = num_stages
        self.equilibrium_weight = equilibrium_weight
        
        # Create exponential weights: early stages get weight ~1, final stage gets equilibrium_weight
        # Formula: weight[i] = exp(log(equilibrium_weight) * i / (num_stages - 1))
        self.stage_weights = []
        for stage in range(num_stages):
            if num_stages == 1:
                weight = 1.0
            else:
                weight = torch.exp(torch.log(torch.tensor(equilibrium_weight)) * stage / (num_stages - 1))
            self.stage_weights.append(weight.item())
        
        print(f"MultiStageLoss weights: {[f'{w:.2f}' for w in self.stage_weights]}")
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size * num_edges, num_stages] 
            targets: [batch_size * num_edges, num_stages] (data.y_intermediate)
        """
        total_loss = 0.0
        
        # Compute weighted loss for each stage
        for stage in range(self.num_stages):
            stage_pred = predictions[:, stage:stage+1]  # [batch_size * num_edges, 1]
            stage_target = targets[:, stage:stage+1]    # [batch_size * num_edges, 1]
            
            stage_loss = self.base_loss_fn(stage_pred, stage_target)
            weighted_loss = self.stage_weights[stage] * stage_loss
            total_loss += weighted_loss
            
        return total_loss


class RatioConservationLoss(torch.nn.Module):
    r"""Implements Eqs. (11)–(14) from Liu & Meidani (2024).

    L = L_u + λ_f · L_f + β · L_c
    """

    def __init__(self,
                 lambda_flow: float = 0.0001,          # w_f   in the paper
                 lambda_conservation: float = 0.0001,              # w_c   "
                 lambda_ratio: float = 1,
                 warmup_epochs: int = 5,
                 tau: float = 10.0) -> None:
        super().__init__()

        # permanent base weights
        self.lambda_flow_base = lambda_flow
        self.lambda_conservation_base = lambda_conservation

        # current (possibly scheduled) weights
        self.lambda_flow = lambda_flow
        self.lambda_conservation = lambda_conservation
        self.lambda_ratio = lambda_ratio

        self.warmup_epochs = warmup_epochs
        self.tau = tau                          # time-constant for exp ramp

    # ------------------------------------------------------------------
    def _conservation(self,
                    f_hat: torch.Tensor,
                    edge_index: torch.Tensor,
                    od_raw: torch.Tensor) -> torch.Tensor:
        """L_c = Σ_i | inflow_i − outflow_i − (arrivals_i − departures_i) |"""
        src, dst = edge_index
        N  = self.N
        B  = od_raw.size(0) // N if od_raw.dim() == 2 else od_raw.size(0)

        edge_batch = src // N
        src_rel, dst_rel = src % N, dst % N

        net = torch.zeros(B * N, device=f_hat.device)
        net.index_add_(0, edge_batch * N + src_rel,  f_hat)   # outgoing flow
        net.index_add_(0, edge_batch * N + dst_rel, -f_hat)   # ingoing flow
        net_flow = net.view(B, N)
     
        od = od_raw if od_raw.dim() == 3 else od_raw.view(B, N, N)
        od_outgoing = od.sum(1)
        od_ingoing = od.sum(2)

        od_difference = od_outgoing - od_ingoing

        resid = (net_flow - od_difference).abs()

        return resid.sum()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_schedule(self, epoch: int) -> None:
        """Exponential ramp-up of λ_f and β after warm-up."""
        if epoch < self.warmup_epochs:
            scale = 0.0
        else:
            t = epoch - self.warmup_epochs + 1
            scale = 1.0 - torch.exp(torch.tensor(-t / self.tau)).item()

        self.lambda_flow = self.lambda_flow_base * scale
        self.lambda_conservation = self.lambda_conservation_base * scale

    # ------------------------------------------------------------------
    def forward(self, u_hat: torch.Tensor, data) -> torch.Tensor:
        # get number of nodes per graph
        self.N = 24
        
        # --- flatten everything to 1-D --------------------------------
        u_hat = u_hat.squeeze(-1)
        cap   = data.capacity_raw.squeeze(-1)
        y_r   = data.y_ratio.squeeze(-1)
        y_f   = data.y_flow.squeeze(-1)
        

        f_hat = u_hat * cap                          # reconstruct flow

        # --- edge losses (mean over |E|) -------------------------------
        L_u = F.l1_loss(u_hat, y_r, reduction="mean")
        L_f = F.l1_loss(f_hat, y_f, reduction="mean")

        # --- node-balance residual ------------------------------------
        L_c = self._conservation(f_hat, data.edge_index, data.od_matrix_raw)
        #print(f"L_u*lambda_ratio: {self.lambda_ratio * L_u}, L_f*lambda_flow: {self.lambda_flow * L_f}, L_c*lambda_conservation: {self.lambda_conservation * L_c}")
        # --- composite -------------------------------------------------
        loss = self.lambda_ratio * L_u + self.lambda_flow * L_f + self.lambda_conservation * L_c
        return loss


def get_loss_function(cfg) -> torch.nn.Module:
        """Get the loss function for the model."""

        name = cfg.train.loss_fn
        print(f"Loading loss function: {name}")
        path_to_loss_fn_config = osp.join("config_train/train/loss_fn", f"{name}.yaml")
        # load yaml file with arguments
        with open(path_to_loss_fn_config, "r") as f:
            args = OmegaConf.load(f)

        if name == "L1Loss":
            loss_fn = torch.nn.L1Loss()

        elif name == "MSELoss":
            loss_fn = torch.nn.MSELoss()

        elif name == "SmoothL1Loss":
            beta = args.beta
            loss_fn = torch.nn.SmoothL1Loss(beta=beta)

        elif name == "MSEPlusMax":
            lam = args.lam
            def loss_fn(pred, target):
                mse = torch.nn.MSELoss()(pred, target)
                linf = torch.max(torch.abs(pred - target))
                return mse + lam * linf
            
            loss_fn = loss_fn

        elif name == "RatioConservationLoss":
            lambda_flow = args.lambda_flow
            lambda_conservation = args.lambda_conservation
            lambda_ratio = args.lambda_ratio
            warmup_epochs = args.warmup_epochs
            tau = args.tau
            loss_fn = RatioConservationLoss(lambda_flow, lambda_conservation, lambda_ratio, warmup_epochs, tau)

        else:
            raise ValueError(f"Loss function {name} not supported. Please choose from ['L1Loss', 'MSELoss', 'SmoothL1Loss', 'MSEPlusMax', 'RatioConservationLoss']")

        # Check if this is a multi-stage model
        if hasattr(cfg.model, 'model_type') and cfg.model.model_type == "MultiStageGNN":
            # Get equilibrium weight from config, with fallback to default
            equilibrium_weight = getattr(cfg.model, 'equilibrium_weight', 4.0)
            loss_fn = MultiStageLoss(loss_fn, cfg.data.num_intermediate_values, equilibrium_weight)

        return loss_fn