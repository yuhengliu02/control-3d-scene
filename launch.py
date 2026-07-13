import os

import torch

from datasets.data import build_infer_loader, build_train_loader
from generate import Generator
from models import build_model
from train import Trainer
from utils.config import parse_args
from utils.distributed import launch
from utils.optim import build_optimizer

def load_weights(module, path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = module.load_state_dict(state, strict=False)
    print(f"Loaded weights from {path} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")

def start(local_rank, cfg):
    cfg.local_rank = local_rank
    torch.cuda.set_device(local_rank)

    model = build_model(cfg)
    if cfg.get("resume", False) and cfg.get("resume_path", ""):
        load_weights(model, cfg.resume_path)

    if cfg.stage == "sg_to_bev":
        if cfg.mode == "loc_posttrain":
            model.freeze_for_loc_posttrain()
        elif cfg.mode == "train":
            model.freeze_localization_head()

    model = model.cuda()
    if cfg.get("distributed", False):
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
            find_unused_parameters=cfg.get("find_unused_parameters", False))

    if cfg.mode in ("train", "loc_posttrain"):
        loader, sampler = build_train_loader(cfg)
        optimizer, sch_iter, sch_epoch = build_optimizer(cfg, model)
        Trainer(cfg, model, optimizer, sch_iter, sch_epoch, loader, sampler).train()
    elif cfg.mode == "inference":
        loader = build_infer_loader(cfg)
        Generator(cfg, model, loader).run()
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

def main():
    cfg = parse_args()
    if not cfg.get("log_path", ""):
        cfg.log_path = os.path.join(cfg.get("log_home", None) or "checkpoints", cfg.exp_name)
    if not cfg.get("node_rank", None):
        cfg.node_rank = 0
    if cfg.mode == "inference":
        cfg.gpus = list(cfg.get("gpus", [0]))[:1]
    launch(start, cfg)

if __name__ == "__main__":
    main()
