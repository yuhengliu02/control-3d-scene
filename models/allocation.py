import torch
from torch_scatter import scatter_sum

EPS = 1e-20

def build_bem(cane, positions, batch, num_graphs):
    n, c = cane.shape
    s = positions.shape[-1]
    mult = cane.unsqueeze(-1).unsqueeze(-1) * positions.unsqueeze(1)
    agg = scatter_sum(mult.reshape(n, -1), batch, dim=0, dim_size=num_graphs)
    return agg.view(num_graphs, c, s, s).unsqueeze(-1)

def gumbel_sample_indices(logits, tau=2.0):
    u = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(u + EPS) + EPS)
    return (logits / tau + gumbel).argmax(dim=-1)

def resolve_overlaps(indices, batch, bev_size, move=5):
    coords = torch.stack([indices // bev_size, indices % bev_size], dim=1).clone()
    for b in batch.unique():
        nodes = torch.nonzero(batch == b, as_tuple=False).squeeze(1)
        seen = {}
        for node in nodes.tolist():
            x, y = int(coords[node, 0]), int(coords[node, 1])
            if (x, y) not in seen:
                seen[(x, y)] = True
                continue
            for dx, dy in ((0, move), (move, 0), (0, -move), (-move, 0)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < bev_size and 0 <= ny < bev_size and (nx, ny) not in seen:
                    coords[node, 0], coords[node, 1] = nx, ny
                    seen[(nx, ny)] = True
                    break
            else:
                seen[(x, y)] = True
    return coords[:, 0] * bev_size + coords[:, 1]

def positions_from_logits(loc_logits, batch, bev_size, tau=2.0, move=5):
    idx = gumbel_sample_indices(loc_logits, tau)
    idx = resolve_overlaps(idx, batch, bev_size, move)
    positions = torch.zeros(loc_logits.size(0), bev_size * bev_size, device=loc_logits.device)
    positions.scatter_(1, idx.unsqueeze(1), 1.0)
    return positions.view(-1, bev_size, bev_size)
