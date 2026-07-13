import os

import numpy as np
import yaml

_CFG = os.path.join(os.path.dirname(__file__), "..", "datasets", "carla.yaml")

def load_color_map(config_path=_CFG):
    cfg = yaml.safe_load(open(config_path, "r"))
    n = max(cfg["color_map"].keys()) + 1
    colors = np.zeros((n, 3), dtype=np.uint8)
    for k, v in cfg["color_map"].items():
        colors[k] = v
    return colors

def save_txt(volume, path):
    vol = volume[..., None] if volume.ndim == 2 else volume
    xs, ys, zs = np.nonzero(vol)
    labels = vol[xs, ys, zs]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savetxt(path, np.stack([labels, xs, ys, zs], axis=1), fmt="%d")

def save_ply(volume, path, colors=None):
    if colors is None:
        colors = load_color_map()
    vol = volume[..., None] if volume.ndim == 2 else volume
    xs, ys, zs = np.nonzero(vol)
    labels = vol[xs, ys, zs]
    rgb = colors[labels]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xs)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for x, y, z, (r, g, b) in zip(xs, ys, zs, rgb):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
