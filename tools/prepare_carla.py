import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import yaml
from numba import njit
from tqdm import tqdm

RAW_SHAPE = (256, 256, 16)

BEV_PRIORITY = np.array(
    [0,    6,    7,     4,    10,   8,    1,    2,     3,   5,   9],
    dtype=np.int64,
)

def load_learning_map(config_path):
    cfg = yaml.safe_load(open(config_path, "r"))
    return np.asarray(list(cfg["learning_map"].values()), dtype=np.uint8)

@njit(cache=True)
def _majority_label(block):
    counts = np.bincount(block)
    if counts.shape[0] <= 1:
        return np.uint8(0)
    nonzero = counts[1:]
    best = np.argmax(nonzero)
    if nonzero[best] == 0:
        return np.uint8(0)
    return np.uint8(best + 1)

@njit(cache=True)
def resample_majority(labels, out_shape):
    ax, ay, az = labels.shape
    ox, oy, oz = out_shape
    reshaped = labels.reshape(ox, ax // ox, oy, ay // oy, oz, az // oz)
    out = np.zeros((ox, oy, oz), dtype=labels.dtype)
    for i in range(ox):
        for j in range(oy):
            for k in range(oz):
                out[i, j, k] = _majority_label(
                    reshaped[i, :, j, :, k, :].copy().flatten()
                )
    return out

def project_bev(scene_3d):
    prio = BEV_PRIORITY[scene_3d]
    top = prio.argmax(axis=2)
    bev = np.take_along_axis(scene_3d, top[..., None], axis=2)[..., 0]
    return bev.astype(np.uint8)

def process_one(label_path, carla_root, out_root, remap_lut, scene_shape):
    rel = os.path.relpath(label_path, carla_root)
    rel_npy = os.path.splitext(rel)[0] + ".npy"
    scene_out = os.path.join(out_root, "scene", rel_npy)
    bev_out = os.path.join(out_root, "bev", rel_npy)
    if os.path.exists(scene_out) and os.path.exists(bev_out):
        return

    raw = np.fromfile(label_path, dtype=np.uint8)
    if raw.size != RAW_SHAPE[0] * RAW_SHAPE[1] * RAW_SHAPE[2]:
        print(f"[skip] unexpected size {raw.size} in {label_path}")
        return
    labels = remap_lut[raw].reshape(RAW_SHAPE)
    scene_3d = resample_majority(np.ascontiguousarray(labels), scene_shape)
    bev = project_bev(scene_3d)

    os.makedirs(os.path.dirname(scene_out), exist_ok=True)
    os.makedirs(os.path.dirname(bev_out), exist_ok=True)
    np.save(scene_out, scene_3d)
    np.save(bev_out, bev)

def collect_label_files(carla_root, splits):
    files = []
    for split in splits:
        split_dir = os.path.join(carla_root, split)
        if not os.path.isdir(split_dir):
            print(f"[warn] split not found: {split_dir}")
            continue
        for root, _, names in os.walk(split_dir):
            if os.path.basename(root) != "evaluation_fine":
                continue
            files.extend(
                os.path.join(root, n) for n in sorted(names) if n.endswith(".label")
            )
    return files

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--carla_root", default="data/Cartesian",
                        help="Extracted CarlaSC root containing Train/Val/Test.")
    parser.add_argument("--out_root", default="data/CarlaSG",
                        help="Where to write processed scene/ and bev/ npy files.")
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "..", "datasets", "carla.yaml"))
    parser.add_argument("--splits", nargs="+", default=["Train", "Val", "Test"])
    parser.add_argument("--scene_shape", nargs=3, type=int, default=[64, 64, 8])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most this many frames (0 = all). For testing.")
    args = parser.parse_args()

    remap_lut = load_learning_map(args.config)
    scene_shape = tuple(args.scene_shape)

    label_files = collect_label_files(args.carla_root, args.splits)
    if args.limit:
        label_files = label_files[: args.limit]
    print(f"Found {len(label_files)} frames to process -> {args.out_root}")

    worker = partial(process_one, carla_root=args.carla_root, out_root=args.out_root,
                     remap_lut=remap_lut, scene_shape=scene_shape)

    if args.workers > 1:
        with Pool(args.workers) as pool:
            list(tqdm(pool.imap_unordered(worker, label_files, chunksize=16),
                      total=len(label_files)))
    else:
        for f in tqdm(label_files):
            worker(f)

    print("Done.")

if __name__ == "__main__":
    main()
