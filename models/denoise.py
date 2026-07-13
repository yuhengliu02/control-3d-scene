import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

def conv3x3x3(i, o, s=1):
    return nn.Conv3d(i, o, 3, stride=s, padding=1, bias=False)

def conv1x3x3(i, o, s=1):
    return nn.Conv3d(i, o, (1, 3, 3), stride=s, padding=(0, 1, 1), bias=False)

def conv1x1x3(i, o, s=1):
    return nn.Conv3d(i, o, (1, 1, 3), stride=s, padding=(0, 0, 1), bias=False)

def conv1x3x1(i, o, s=1):
    return nn.Conv3d(i, o, (1, 3, 1), stride=s, padding=(0, 1, 0), bias=False)

def conv3x1x1(i, o, s=1):
    return nn.Conv3d(i, o, (3, 1, 1), stride=s, padding=(1, 0, 0), bias=False)

def conv3x1x3(i, o, s=1):
    return nn.Conv3d(i, o, (3, 1, 3), stride=s, padding=(1, 0, 1), bias=False)

def conv1x1(i, o, s=1):
    return nn.Conv3d(i, o, 1, stride=s)

def _gn(channels):
    return nn.GroupNorm(16 if channels < 32 else 32, channels)

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

class AsymmetricResidualBlock(nn.Module):

    def __init__(self, in_c, out_c, time_c):
        super().__init__()
        self.gn = _gn(in_c)
        self.bn0, self.bn0_2 = _gn(out_c), _gn(out_c)
        self.bn1, self.bn2 = _gn(out_c), _gn(out_c)
        self.time_layers = nn.Sequential(nn.SiLU(), nn.Linear(time_c, in_c * 2))
        self.conv1, self.act1 = conv1x3x3(in_c, out_c), nn.LeakyReLU()
        self.conv1_2, self.act1_2 = conv3x1x3(out_c, out_c), nn.LeakyReLU()
        self.conv2, self.act2 = conv3x1x3(in_c, out_c), nn.LeakyReLU()
        self.conv3, self.act3 = conv1x3x3(out_c, out_c), nn.LeakyReLU()

    def forward(self, x, t):
        t = self.time_layers(t)
        while len(t.shape) < len(x.shape):
            t = t[..., None]
        scale, shift = torch.chunk(t, 2, dim=1)
        x = self.gn(x) * (1 + scale) + shift

        short = self.bn0_2(self.act1_2(self.conv1_2(self.bn0(self.act1(self.conv1(x))))))
        res = self.bn2(self.act3(self.conv3(self.bn1(self.act2(self.conv2(x))))))
        return res + short

class DDCM(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1, self.bn0, self.act1 = conv3x1x1(in_c, out_c), _gn(out_c), nn.Sigmoid()
        self.conv1_2, self.bn0_2, self.act1_2 = conv1x3x1(in_c, out_c), _gn(out_c), nn.Sigmoid()
        self.conv1_3, self.bn0_3, self.act1_3 = conv1x1x3(in_c, out_c), _gn(out_c), nn.Sigmoid()

    def forward(self, x):
        a = self.act1(self.bn0(self.conv1(x)))
        b = self.act1_2(self.bn0_2(self.conv1_2(x)))
        c = self.act1_3(self.bn0_3(self.conv1_3(x)))
        return (a + b + c) * x

class Attention(nn.Module):
    def __init__(self, dim, heads=4, scale=10):
        super().__init__()
        self.scale, self.heads = scale, heads
        self.to_qkv = conv1x1(dim, dim * 3)
        self.to_out = conv1x1(dim, dim)

    def forward(self, x):
        b, c, h, w, z = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        sim = einsum("b h d i, b h d j -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y z) d -> b (h d) x y z", x=h, y=w, z=z)
        return self.to_out(out)

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, time_c, height_pooling):
        super().__init__()
        self.residual_block = AsymmetricResidualBlock(in_c, out_c, time_c)
        stride = 2 if height_pooling else (2, 2, 1)
        self.pool = nn.Conv3d(out_c, out_c, 3, stride=stride, padding=1, bias=False)

    def forward(self, x, t):
        res = self.residual_block(x, t)
        return self.pool(res), res

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, time_c, height_pooling):
        super().__init__()
        self.trans_bn, self.bn1, self.bn2, self.bn3 = _gn(in_c), _gn(out_c), _gn(out_c), _gn(out_c)
        self.trans_dilao, self.trans_act = conv3x3x3(in_c, in_c), nn.LeakyReLU()
        self.time_layers = nn.Sequential(nn.SiLU(), nn.Linear(time_c, in_c * 2))
        self.conv1, self.act1 = conv1x3x3(in_c, out_c), nn.LeakyReLU()
        self.conv2, self.act2 = conv3x1x3(out_c, out_c), nn.LeakyReLU()
        self.conv3, self.act3 = conv3x3x3(out_c, out_c), nn.LeakyReLU()
        if height_pooling:
            self.up_subm = nn.ConvTranspose3d(in_c, in_c, 3, stride=2, padding=1,
                                              output_padding=1, bias=False)
        else:
            self.up_subm = nn.ConvTranspose3d(in_c, in_c, (3, 3, 1), stride=(2, 2, 1),
                                              padding=(1, 1, 0), output_padding=(1, 1, 0),
                                              bias=False)

    def forward(self, x, residual, t):
        up = self.trans_act(self.trans_dilao(x))
        t = self.time_layers(t)
        while len(t.shape) < len(x.shape):
            t = t[..., None]
        scale, shift = torch.chunk(t, 2, dim=1)
        up = self.trans_bn(up) * (1 + scale) + shift
        up = self.up_subm(up) + residual
        up = self.bn1(self.act1(self.conv1(up)))
        up = self.bn2(self.act2(self.conv2(up)))
        up = self.bn3(self.act3(self.conv3(up)))
        return up

class Denoise(nn.Module):
    def __init__(self, num_classes=11, cond_channels=1, init_size=32,
                 up_height_pooling=True):
        super().__init__()
        self.init_size = init_size
        self.time_size = init_size * 4

        self.time_embed = nn.Sequential(
            nn.Linear(init_size, self.time_size), nn.SiLU(),
            nn.Linear(self.time_size, self.time_size),
        )
        self.embedding = nn.Embedding(num_classes, init_size)
        self.conv_in = nn.Conv3d(init_size + cond_channels, init_size, 1)

        tc = init_size * 4
        self.A = AsymmetricResidualBlock(init_size, init_size, tc)
        self.downBlock1 = DownBlock(init_size, 2 * init_size, tc, height_pooling=True)
        self.downBlock2 = DownBlock(2 * init_size, 4 * init_size, tc, height_pooling=True)
        self.downBlock3 = DownBlock(4 * init_size, 8 * init_size, tc, height_pooling=False)
        self.downBlock4 = DownBlock(8 * init_size, 16 * init_size, tc, height_pooling=False)
        self.midBlock1 = AsymmetricResidualBlock(16 * init_size, 16 * init_size, tc)
        self.attention = Attention(16 * init_size, 32)
        self.midBlock2 = AsymmetricResidualBlock(16 * init_size, 16 * init_size, tc)
        self.upBlock4 = UpBlock(16 * init_size, 8 * init_size, tc, height_pooling=False)
        self.upBlock3 = UpBlock(8 * init_size, 4 * init_size, tc, height_pooling=False)
        self.upBlock2 = UpBlock(4 * init_size, 2 * init_size, tc, height_pooling=up_height_pooling)
        self.upBlock1 = UpBlock(2 * init_size, 2 * init_size, tc, height_pooling=up_height_pooling)
        self.DDCM = DDCM(2 * init_size, 2 * init_size)
        self.logits = nn.Conv3d(4 * init_size, num_classes, 3, padding=1, bias=True)

    def forward(self, x, x_cond, t):
        x = self.embedding(x).permute(0, 4, 1, 2, 3)
        x = torch.cat([x, x_cond], dim=1)
        x = self.conv_in(x)

        t = self.time_embed(timestep_embedding(t, self.init_size))
        x = self.A(x, t)
        d1c, d1b = self.downBlock1(x, t)
        d2c, d2b = self.downBlock2(d1c, t)
        d3c, d3b = self.downBlock3(d2c, t)
        d4c, d4b = self.downBlock4(d3c, t)
        m = self.midBlock2(self.attention(self.midBlock1(d4c, t)), t)
        u4 = self.upBlock4(m, d4b, t)
        u3 = self.upBlock3(u4, d3b, t)
        u2 = self.upBlock2(u3, d2b, t)
        u1 = self.upBlock1(u2, d1b, t)
        u0 = self.DDCM(u1)
        return self.logits(torch.cat([u1, u0], dim=1))
