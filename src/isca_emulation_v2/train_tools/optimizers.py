from omegaconf import OmegaConf
import torch
import torch_optimizer as optim
import os.path as osp

def get_optimizer(cfg, model):
        """Get the optimizer for the model."""
        name = cfg.train.optimizer

        path_to_optimizer_config = osp.join("config_train/train/optimizer", f"{name}.yaml")
        # load yaml file with arguments
        with open(path_to_optimizer_config, "r") as f:
            args = OmegaConf.load(f)
        
        if name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

        elif name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        
        elif name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=args.momentum, weight_decay=cfg.train.weight_decay)

        elif name == "RAdam":
            optimizer = optim.RAdam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        
        elif name == "AdaBelief":
            optimizer = optim.AdaBelief(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        
        elif name == "AdamP":
            optimizer = optim.AdamP(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        elif name == "RMSprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.train.lr, momentum=args.momentum, weight_decay=cfg.train.weight_decay)
        
        elif name == "AdaGrad":
            optimizer = torch.optim.Adagrad(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        elif name == "AdaDelta":
            optimizer = torch.optim.Adadelta(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        elif name == "AdaBound":
            optimizer = optim.AdaBound(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        elif name == "Yogi":
            optimizer = optim.Yogi(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        else:
            raise ValueError(f"Optimizer {name} not supported. Please choose from ['Adam', 'AdamW', 'SGD', 'RAdam', 'AdaBelief', 'AdamP', 'RMSprop', 'AdaGrad', 'AdaDelta', 'AdaBound', 'Yogi']")

        return optimizer