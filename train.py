import os

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.distributed import is_main_process

class Trainer:
    def __init__(self, cfg, model, optimizer, scheduler_iter, scheduler_epoch,
                 loader, sampler):
        self.cfg = cfg
        self.model = model
        self.module = model.module if hasattr(model, "module") else model
        self.optimizer = optimizer
        self.scheduler_iter = scheduler_iter
        self.scheduler_epoch = scheduler_epoch
        self.loader = loader
        self.sampler = sampler
        self.device = torch.device("cuda", cfg.get("local_rank", 0))
        self.start_epoch = 0

        self.log_path = cfg.log_path
        self.writer = None
        if is_main_process():
            os.makedirs(self.log_path, exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.log_path, "tb"))

        if cfg.get("resume_optimizer", False) and cfg.get("resume_path", ""):
            ckpt = torch.load(cfg.resume_path, map_location="cpu")
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            self.start_epoch = ckpt.get("epoch", 0)
            if is_main_process():
                print(f"Resumed optimizer/epoch from {cfg.resume_path} "
                      f"(start epoch {self.start_epoch})")

    def _step(self, batch):
        if self.cfg.stage == "sg_to_bev":
            sg = batch["scene_graph"].to(self.device)
            if self.cfg.mode == "loc_posttrain":
                return self.model(sg, mode="loc")
            bev = batch["bev"].to(self.device)
            return self.model(sg, bev, mode="joint")
        bev = batch["bev"].to(self.device)
        scene = batch["scene_3d"].to(self.device)
        return self.model(bev, scene)

    def train(self):
        for epoch in range(self.start_epoch, self.cfg.epochs):
            self.model.train()
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            running, count = {}, 0
            max_iters = self.cfg.get("max_iters", 0)
            for batch in self.loader:
                if max_iters and count >= max_iters:
                    break
                self.optimizer.zero_grad()
                losses = self._step(batch)
                losses["total"].backward()
                if self.cfg.get("clip_norm", None):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_norm)
                if self.cfg.get("clip_value", None):
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.cfg.clip_value)
                self.optimizer.step()
                if self.scheduler_iter:
                    self.scheduler_iter.step()

                count += 1
                for k, v in losses.items():
                    running[k] = running.get(k, 0.0) + float(v.detach())
                if is_main_process() and count % 20 == 0:
                    msg = ", ".join(f"{k}: {running[k] / count:.4f}" for k in running)
                    print(f"[{self.cfg.stage}/{self.cfg.mode}] epoch {epoch + 1}/{self.cfg.epochs} "
                          f"iter {count}/{len(self.loader)} | {msg}", end="\r")

            if self.scheduler_epoch:
                self.scheduler_epoch.step()
            if is_main_process():
                avg = {k: running[k] / max(count, 1) for k in running}
                print(f"\n[{self.cfg.stage}/{self.cfg.mode}] epoch {epoch + 1} done | "
                      + ", ".join(f"{k}: {v:.4f}" for k, v in avg.items()))
                for k, v in avg.items():
                    self.writer.add_scalar(f"train/{k}", v, epoch + 1)
                if (epoch + 1) % self.cfg.check_every == 0:
                    self._save_checkpoint(epoch + 1)
        if is_main_process():
            self._save_checkpoint(self.cfg.epochs, name="latest.tar")

    def _save_checkpoint(self, epoch, name=None):
        ckpt = {"epoch": epoch, "model": self.module.state_dict(),
                "optimizer": self.optimizer.state_dict()}
        path = os.path.join(self.log_path, name or f"epoch{epoch}.tar")
        torch.save(ckpt, path)
        torch.save(ckpt, os.path.join(self.log_path, "latest.tar"))
        print(f"Saved checkpoint: {path}")
