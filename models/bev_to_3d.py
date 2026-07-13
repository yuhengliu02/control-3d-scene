import torch
import torch.nn as nn

from .denoise import Denoise
from .diffusion import ConditionalDiffusion

class BEVto3D(nn.Module):
    def __init__(self, num_classes=11, num_timesteps=100, auxiliary_loss_weight=1e-4,
                 scene_depth=8):
        super().__init__()
        self.scene_depth = scene_depth
        denoise = Denoise(num_classes, cond_channels=1, up_height_pooling=True)
        self.diffusion = ConditionalDiffusion(denoise, num_classes, num_timesteps,
                                              auxiliary_loss_weight)

    def _condition(self, bev):
        cond = bev.unsqueeze(-1).expand(-1, -1, -1, self.scene_depth)
        return cond.unsqueeze(1).float()

    def forward(self, bev, scene_3d):
        return {"total": self.diffusion(scene_3d, self._condition(bev))}

    @torch.no_grad()
    def sample(self, bev, num_steps=None):
        return self.diffusion.sample(self._condition(bev), num_steps)
