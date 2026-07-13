from .bev_to_3d import BEVto3D
from .sg_to_bev import SGtoBEV

__all__ = ["SGtoBEV", "BEVto3D", "build_model"]

def build_model(cfg):
    if cfg.stage == "sg_to_bev":
        return SGtoBEV(
            num_classes=cfg.num_classes,
            in_channels=cfg.node_in_channels,
            hidden_channels=cfg.gnn_hidden,
            heads=cfg.gnn_heads,
            bev_size=cfg.bev_size,
            num_timesteps=cfg.diffusion_steps,
            auxiliary_loss_weight=cfg.auxiliary_loss_weight,
            recon_loss_weight=cfg.recon_loss_weight,
            seg_loss_weight=cfg.seg_loss_weight,
            uncond_prob=cfg.uncond_prob,
            gumbel_tau=cfg.gumbel_tau,
            overlap_move=cfg.overlap_move,
        )
    if cfg.stage == "bev_to_3d":
        return BEVto3D(
            num_classes=cfg.num_classes,
            num_timesteps=cfg.diffusion_steps,
            auxiliary_loss_weight=cfg.auxiliary_loss_weight,
            scene_depth=cfg.scene_shape[2],
        )
    raise ValueError(f"Unknown stage: {cfg.stage}")
