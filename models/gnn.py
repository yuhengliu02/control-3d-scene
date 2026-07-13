import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, InnerProductDecoder, global_mean_pool
from torch_geometric.utils import negative_sampling

EPS = 1e-15

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        node_emb = self.conv2(x, edge_index)
        global_emb = global_mean_pool(node_emb, batch)
        return node_emb, global_emb

class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class SceneGraphGNN(nn.Module):

    def __init__(self, in_channels=34, hidden_channels=32, heads=4,
                 num_classes=11, bev_size=64):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.bev_size = bev_size
        self.encoder = GATEncoder(in_channels, hidden_channels, heads)
        self.reduce_dim = nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=1)
        self.localization_head = MLPHead(hidden_channels, bev_size * bev_size)
        self.node_classifier = MLPHead(hidden_channels, num_classes)
        self.edge_decoder = InnerProductDecoder()

    def encode(self, scene_graph):
        node_emb, global_emb = self.encoder(
            scene_graph.x, scene_graph.edge_index, scene_graph.batch)
        combined = torch.cat([node_emb, global_emb[scene_graph.batch]], dim=1)
        cane = self.reduce_dim(combined.unsqueeze(-1)).squeeze(-1)
        return cane, node_emb

    def forward(self, scene_graph):
        cane, node_emb = self.encode(scene_graph)
        return {
            "cane": cane,
            "node_emb": node_emb,
            "loc_logits": self.localization_head(cane),
            "cls_logits": self.node_classifier(cane),
        }

    def edge_recon_loss(self, node_emb, pos_edge_index):
        pos = -torch.log(self.edge_decoder(node_emb, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_edge_index = negative_sampling(pos_edge_index, node_emb.size(0))
        neg = -torch.log(1 - self.edge_decoder(node_emb, neg_edge_index, sigmoid=True) + EPS).mean()
        return pos + neg
