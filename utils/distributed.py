import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def _find_free_port():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def launch(worker, cfg):
    gpus = cfg.get("gpus", [0])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    world_size = len(gpus)
    cfg.world_size = world_size
    cfg.distributed = world_size > 1

    if world_size > 1:
        port = _find_free_port()
        cfg.dist_url = f"tcp://127.0.0.1:{port}"
        mp.spawn(_worker_wrapper, nprocs=world_size, args=(worker, cfg), daemon=False)
    else:
        worker(0, cfg)

def _worker_wrapper(local_rank, worker, cfg):
    dist.init_process_group(backend="nccl", init_method=cfg.dist_url,
                            world_size=cfg.world_size, rank=local_rank)
    torch.cuda.set_device(local_rank)
    worker(local_rank, cfg)
    dist.destroy_process_group()

def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
