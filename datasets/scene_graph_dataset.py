import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm

BEV_SIZE = 64
PATCH_GRID = 4
COORD_RANGE = 256.0

class SceneGraphDataset(Dataset):
    def __init__(self, sg_dir, bev_dir, scene_dir, num_classes=11, num_road_types=6,
                 load_scene_graph=True, load_bev=True, load_scene=True,
                 feature_mask_ratio=0.0):
        self.bev_dir = bev_dir
        self.scene_dir = scene_dir
        self.num_classes = num_classes
        self.num_road_types = num_road_types
        self.load_scene_graph = load_scene_graph
        self.load_bev = load_bev
        self.load_scene = load_scene
        self.feature_mask_ratio = feature_mask_ratio

        nodes = pd.read_csv(os.path.join(sg_dir, "nodes.csv"))
        edges = pd.read_csv(os.path.join(sg_dir, "edges.csv"))
        self.nodes_by_frame = {fp: g.to_dict("records")
                               for fp, g in nodes.groupby("file_path")}
        self.edges_by_frame = {fp: g.to_dict("records")
                               for fp, g in edges.groupby("file_path")}

        self.frames = []
        for fp in tqdm(sorted(self.nodes_by_frame.keys()), desc="Indexing frames"):
            if self.load_bev and not os.path.exists(self._npy(self.bev_dir, fp)):
                continue
            if self.load_scene and not os.path.exists(self._npy(self.scene_dir, fp)):
                continue
            self.frames.append(fp)
        if not self.frames:
            raise RuntimeError(
                f"No usable frames found under {bev_dir}/{scene_dir}. "
                "Did you run tools/prepare_carla.py?")
        print(f"SceneGraphDataset: {len(self.frames)} frames from {sg_dir}")

    @staticmethod
    def _npy(root, file_path):
        rel = file_path.lstrip("/")[:-5] + ".npy"
        return os.path.join(root, rel)

    def __len__(self):
        return len(self.frames)

    def _position_patch(self, cx, cy):
        xi = min(PATCH_GRID - 1, int(cx / COORD_RANGE * PATCH_GRID))
        yi = min(PATCH_GRID - 1, int(cy / COORD_RANGE * PATCH_GRID))
        feat = np.zeros(1 + PATCH_GRID * PATCH_GRID, dtype=np.float32)
        if xi == 0 and yi == 0:
            feat[0] = 1.0
        else:
            feat[1 + xi * PATCH_GRID + yi] = 1.0
        return feat

    def _heatmap(self, cx, cy):
        xi = min(BEV_SIZE - 1, int(cx / COORD_RANGE * BEV_SIZE))
        yi = min(BEV_SIZE - 1, int(cy / COORD_RANGE * BEV_SIZE))
        hm = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.float32)
        hm[xi, yi] = 1.0
        return hm

    @staticmethod
    def _parse_centroid(s):
        return [float(v) for v in str(s).strip("()").split(",")]

    def _build_graph(self, file_path):
        nodes = self.nodes_by_frame[file_path]
        edges = self.edges_by_frame.get(file_path, [])

        id_to_idx = {n["instance_id"]: i for i, n in enumerate(nodes)}
        feats, labels, heatmaps = [], [], []
        for n in nodes:
            label = int(n["label_id"])
            cx, cy = self._parse_centroid(n["centroid"])[:2]

            label_oh = np.eye(self.num_classes, dtype=np.float32)[label]
            pos = self._position_patch(cx, cy)
            if self.feature_mask_ratio > 0 and np.random.rand() < self.feature_mask_ratio:
                pos = np.zeros_like(pos)
            road = np.eye(self.num_road_types, dtype=np.float32)[int(float(n["road_type"]))]

            feats.append(np.concatenate([label_oh, pos, road]))
            labels.append(label)
            heatmaps.append(self._heatmap(cx, cy))

        edge_index = []
        for e in edges:
            s, d = id_to_idx.get(e["subject_instance_id"]), id_to_idx.get(e["object_instance_id"])
            if s is None or d is None:
                continue
            edge_index += [[s, d], [d, s]]
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(
            x=torch.tensor(np.stack(feats), dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(labels, dtype=torch.long),
            heatmaps=torch.tensor(np.stack(heatmaps), dtype=torch.float),
        )

    def __getitem__(self, idx):
        fp = self.frames[idx]
        sample = {"file_path": fp}
        if self.load_scene_graph:
            sample["scene_graph"] = self._build_graph(fp)
        if self.load_bev:
            sample["bev"] = torch.from_numpy(
                np.load(self._npy(self.bev_dir, fp)).astype(np.int64))
        if self.load_scene:
            sample["scene_3d"] = torch.from_numpy(
                np.load(self._npy(self.scene_dir, fp)).astype(np.int64))
        return sample

def collate_scene_graphs(batch):
    out = {"file_path": [b["file_path"] for b in batch]}
    if "scene_graph" in batch[0]:
        out["scene_graph"] = Batch.from_data_list([b["scene_graph"] for b in batch])
    if "bev" in batch[0]:
        out["bev"] = torch.stack([b["bev"] for b in batch])
    if "scene_3d" in batch[0]:
        out["scene_3d"] = torch.stack([b["scene_3d"] for b in batch])
    return out
