import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler

class LinearWarmupScheduler(_LRScheduler):

    def __init__(self, optimizer, total_epoch, last_epoch=-1):
        self.total_epoch = total_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * min(1, self.last_epoch / self.total_epoch)
                for base_lr in self.base_lrs]

def build_optimizer(cfg, model):
    params = [p for p in model.parameters() if p.requires_grad]
    name = cfg.get("optimizer", "adamw")
    if name == "sgd":
        optimizer = optim.SGD(params, lr=cfg.lr, momentum=cfg.get("momentum", 0.9))
    elif name == "adam":
        optimizer = optim.Adam(params, lr=cfg.lr,
                               betas=(cfg.get("momentum", 0.9), cfg.get("momentum_sqr", 0.999)))
    elif name == "adamw":
        optimizer = optim.AdamW(params, lr=cfg.lr,
                                betas=(cfg.get("momentum", 0.9), cfg.get("momentum_sqr", 0.999)))
    else:
        raise ValueError(f"Unknown optimizer: {name}")

    warmup = cfg.get("warmup", None)
    scheduler_iter = LinearWarmupScheduler(optimizer, warmup) if warmup else None
    milestones = cfg.get("milestones", []) or []
    scheduler_epoch = MultiStepLR(optimizer, milestones=milestones,
                                  gamma=cfg.get("gamma", 0.1)) if milestones else None
    return optimizer, scheduler_iter, scheduler_epoch
