import os

import numpy as np
import torch
from tqdm import tqdm

from utils.visualize import load_color_map, save_ply

class Generator:
    def __init__(self, cfg, model, loader):
        self.cfg = cfg
        self.module = model.module if hasattr(model, "module") else model
        self.loader = loader
        self.device = torch.device("cuda", cfg.get("local_rank", 0))
        self.num_steps = cfg.get("num_steps", None)
        self.out_dir = os.path.join(cfg.log_path, "Generated")
        self.ply_dir = os.path.join(cfg.log_path, "GeneratedPly")
        os.makedirs(self.out_dir, exist_ok=True)
        if cfg.get("save_ply", True):
            os.makedirs(self.ply_dir, exist_ok=True)
        self.colors = load_color_map()

    @torch.no_grad()
    def run(self):
        self.module.eval()
        produced = 0
        limit = self.cfg.get("generation_num", 0)
        for batch in tqdm(self.loader, desc=f"generate[{self.cfg.stage}]"):
            if self.cfg.stage == "sg_to_bev":
                sg = batch["scene_graph"].to(self.device)
                out = self.module.sample(sg, num_steps=self.num_steps)
            else:
                bev = batch["bev"].to(self.device)
                out = self.module.sample(bev, num_steps=self.num_steps)
            out = out.cpu().numpy().astype(np.uint8)

            for i in range(out.shape[0]):
                tag = f"{produced:05d}"
                np.save(os.path.join(self.out_dir, f"{tag}.npy"), out[i])
                if self.cfg.get("save_ply", True):
                    save_ply(out[i], os.path.join(self.ply_dir, f"{tag}.ply"), self.colors)
                produced += 1
                if limit and produced >= limit:
                    print(f"Generated {produced} samples -> {self.out_dir}")
                    return
        print(f"Generated {produced} samples -> {self.out_dir}")
