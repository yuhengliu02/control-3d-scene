import torch
import torch.nn as nn
import torch.nn.functional as F

from .allocation import build_bem, positions_from_logits
from .denoise import Denoise
from .diffusion import ConditionalDiffusion
from .gnn import SceneGraphGNN

class SGtoBEV(nn.Module):
    def __init__(self, num_classes=11, in_channels=34, hidden_channels=32, heads=4,
                 bev_size=64, num_timesteps=100, auxiliary_loss_weight=1e-4,
                 recon_loss_weight=0.2, seg_loss_weight=0.2,
                 uncond_prob=0.1, gumbel_tau=2.0, overlap_move=5):
        super().__init__()
        self.bev_size = bev_size
        self.hidden_channels = hidden_channels
        self.recon_loss_weight = recon_loss_weight
        self.seg_loss_weight = seg_loss_weight
        self.uncond_prob = uncond_prob
        self.gumbel_tau = gumbel_tau
        self.overlap_move = overlap_move

        self.gnn = SceneGraphGNN(in_channels, hidden_channels, heads, num_classes, bev_size)
        denoise = Denoise(num_classes, cond_channels=hidden_channels,
                          up_height_pooling=False)
        self.diffusion = ConditionalDiffusion(denoise, num_classes, num_timesteps,
                                              auxiliary_loss_weight)

    def _bem(self, gnn_out, scene_graph, positions):
        num_graphs = int(scene_graph.batch.max()) + 1
        bem = build_bem(gnn_out["cane"], positions, scene_graph.batch, num_graphs)
        if self.training and self.uncond_prob > 0:
            drop = (torch.rand(bem.size(0), device=bem.device) < self.uncond_prob)
            drop = drop.view(-1, 1, 1, 1, 1)
            bem = torch.where(drop, torch.ones_like(bem), bem)
        return bem

    def forward(self, scene_graph, bev=None, mode="joint"):
        if mode == "loc":
            return {"total": self.loc_loss(scene_graph)}
        gnn_out = self.gnn(scene_graph)
        bem = self._bem(gnn_out, scene_graph, scene_graph.heatmaps)
        diff_loss = self.diffusion(bev.unsqueeze(-1), bem)
        recon_loss = self.gnn.edge_recon_loss(gnn_out["node_emb"], scene_graph.edge_index)
        seg_loss = F.cross_entropy(gnn_out["cls_logits"], scene_graph.y)
        total = ((1 - self.recon_loss_weight - self.seg_loss_weight) * diff_loss
                 + self.recon_loss_weight * recon_loss
                 + self.seg_loss_weight * seg_loss)
        return {"total": total, "diffusion": diff_loss,
                "recon": recon_loss, "seg": seg_loss}

    def loc_loss(self, scene_graph):
        gnn_out = self.gnn(scene_graph)
        target = scene_graph.heatmaps.flatten(start_dim=1).argmax(dim=1)
        return F.cross_entropy(gnn_out["loc_logits"], target)

    @torch.no_grad()
    def sample(self, scene_graph, tau=None, num_steps=None):
        gnn_out = self.gnn(scene_graph)
        positions = positions_from_logits(
            gnn_out["loc_logits"], scene_graph.batch, self.bev_size,
            tau=self.gumbel_tau if tau is None else tau, move=self.overlap_move)
        num_graphs = int(scene_graph.batch.max()) + 1
        bem = build_bem(gnn_out["cane"], positions, scene_graph.batch, num_graphs)
        bev = self.diffusion.sample(bem, num_steps)
        return bev.squeeze(-1)

    def freeze_localization_head(self):
        for p in self.gnn.localization_head.parameters():
            p.requires_grad = False

    def freeze_for_loc_posttrain(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.gnn.localization_head.parameters():
            p.requires_grad = True
