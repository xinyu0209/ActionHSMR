import torch
import torch.nn as nn
from timm.models.layers import DropPath


class Mamba(nn.Module):
    def __init__(self, dim, depth=2, d_state=16, d_conv=4, expand=2, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaLayer(dim, d_state, d_conv, expand, drop_path, norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(dim)

    def forward(self, x):
        # x: (N, T, D)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state, d_conv, expand, drop_path, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim * expand, bias=False)
        self.conv1d = nn.Conv1d(dim * expand, dim * expand, d_conv, padding=d_conv-1, groups=dim * expand, bias=False)
        self.x_proj = nn.Linear(dim * expand, d_state*2, bias=False)
        self.dt_proj = nn.Linear(d_state, dim * expand, bias=True)
        self.A_proj = nn.Linear(d_state, dim * expand, bias=True)
        self.out_proj = nn.Linear(dim * expand, dim, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (N, T, D)
        shortcut = x
        x = self.norm(x)
        x = self.in_proj(x)
        x = x.transpose(1, 2)    # (N, C, T)
        x = self.conv1d(x)[:, :, :x.shape[2]]
        x = x.transpose(1, 2)    # (N, T, C)
        x = self.act(x)
        x_dbl = self.x_proj(x)
        dt, A = torch.split(x_dbl, x_dbl.shape[-1]//2, dim=-1)
        dt = self.dt_proj(dt)
        A = self.A_proj(A)
        x = x * torch.sigmoid(dt) + A
        x = self.out_proj(x)
        x = shortcut + self.drop_path(x)
        return x
