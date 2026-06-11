from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR, CyclicLR
from omegaconf import OmegaConf
import os.path as osp

def get_scheduler(cfg, optimizer, num_batches):
        
        name = cfg.train.scheduler
        path_to_scheduler_config = osp.join("config_train/train/scheduler", f"{name}.yaml")
        # load yaml file with arguments
        with open(path_to_scheduler_config, "r") as f:
            args = OmegaConf.load(f)

        if name == "ReduceLROnPlateau":
            sched = ReduceLROnPlateau(
                optimizer,
                mode=args.mode,
                factor=args.factor,
                patience=args.patience,
            )

        elif name == "ExponentialLR":
            sched = ExponentialLR(
                optimizer,
                gamma=args.gamma,
            )

        elif name == "CosineAnnealingLR":
            sched = CosineAnnealingLR(
                optimizer,
                T_max=args.T_max,
                eta_min=args.eta_min,
            )

        elif name == "CosineAnnealingWarmRestarts":
            sched = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=args.T_0,
                T_mult=args.T_mult,
                eta_min=args.eta_min,
            )

        elif name == "OneCycleLR":
            sched = OneCycleLR(
                optimizer,
                max_lr=cfg.train.lr,
                epochs=cfg.train.epochs,
                steps_per_epoch = num_batches,
                pct_start=args.pct_start,
                anneal_strategy=args.anneal_strategy,
            )

        elif name == "CyclicLR":
            up_ep = args.step_size_up_epochs
            down_ep = args.step_size_down_epochs

            step_size_up = int(up_ep * num_batches)
            step_size_down = int(down_ep * num_batches)

            sched = CyclicLR(
                optimizer,
                base_lr=args.lower_bound_lr * cfg.train.lr,
                max_lr=cfg.train.lr,
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                mode=args.mode,
            )

        else:
            raise ValueError(f"Scheduler {name} not supported.")

        return sched