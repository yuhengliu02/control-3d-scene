import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .scene_graph_dataset import SceneGraphDataset, collate_scene_graphs

def _load_flags(cfg, training):
    if cfg.stage == "sg_to_bev":
        return dict(load_scene_graph=True, load_bev=training, load_scene=False)
    return dict(load_scene_graph=False, load_bev=True, load_scene=True)

def build_train_loader(cfg):
    sg_dir = os.path.join(cfg.sg_root, cfg.train_split)
    flags = _load_flags(cfg, training=True)
    ds = SceneGraphDataset(
        sg_dir=sg_dir, bev_dir=cfg.bev_root, scene_dir=cfg.scene_root,
        num_classes=cfg.num_classes, num_road_types=cfg.num_road_types,
        feature_mask_ratio=(cfg.feature_mask_ratio if cfg.stage == "sg_to_bev" else 0.0),
        **flags)
    sampler = DistributedSampler(ds, shuffle=True) if cfg.get("distributed", False) else None
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=cfg.num_workers,
                        collate_fn=collate_scene_graphs, pin_memory=cfg.get("pin_memory", False),
                        drop_last=True)
    return loader, sampler

def build_infer_loader(cfg):
    if cfg.stage == "bev_to_3d" and cfg.get("infer_source", "dataset") == "generation":
        ds = GeneratedBEVDataset(cfg.prev_scene_path)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, collate_fn=GeneratedBEVDataset.collate)

    sg_dir = os.path.join(cfg.sg_root, cfg.val_split)
    flags = _load_flags(cfg, training=False)
    ds = SceneGraphDataset(
        sg_dir=sg_dir, bev_dir=cfg.bev_root, scene_dir=cfg.scene_root,
        num_classes=cfg.num_classes, num_road_types=cfg.num_road_types,
        feature_mask_ratio=0.0, **flags)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                      num_workers=cfg.num_workers, collate_fn=collate_scene_graphs)

class GeneratedBEVDataset(Dataset):

    def __init__(self, directory):
        self.files = sorted(glob.glob(os.path.join(directory, "*.npy")))
        if not self.files:
            raise RuntimeError(f"No .npy BEV maps found in {directory}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        bev = np.load(self.files[idx]).astype(np.int64)
        return {"file_path": os.path.basename(self.files[idx]),
                "bev": torch.from_numpy(bev)}

    @staticmethod
    def collate(batch):
        return {"file_path": [b["file_path"] for b in batch],
                "bev": torch.stack([b["bev"] for b in batch])}
