import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from omegaconf import ListConfig
from .vit import PatchEmbed
from mamba_ssm.modules.mamba_simple import Mamba

class MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(dim)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class SpatialConvEnhancer(nn.Module):
    def __init__(self, embed_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size, padding=kernel_size//2, groups=embed_dim)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()

    def forward(self, x, Hp, Wp):
        B, N, D = x.shape
        x = x.transpose(1, 2).reshape(B, D, Hp, Wp)
        x = self.act(self.bn(self.conv(x)))
        x = x.flatten(2).transpose(1, 2)
        return x

class Mamba2(nn.Module):
    def __init__(
        self,
        img_size=[256,192],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_frames=3,
        **kwargs
    ):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        from .vit import PatchEmbed
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # 图像尺寸计算
        if isinstance(img_size, (list, tuple, ListConfig)):
            img_h, img_w = img_size
        else:
            img_h = img_w = img_size
        patch_h = img_h // patch_size
        patch_w = img_w // patch_size

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.num_patches = patch_h * patch_w

        # 位置信息等
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        self.layers = nn.ModuleList([MambaBlock(embed_dim) for _ in range(depth)])
        self.temporal_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(2)
        ])
        self.key_frame_selector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, T, C, H, W)  
        return: (B, D, Hp, Wp)  ← for skel_model
        """
        B, T, C, H, W = x.shape
        assert C == 3 and T == self.num_frames

        # 1. Patch Embedding
        x = x.view(B * T, C, H, W)
        x, _ = self.patch_embed(x)                 # (B*T, P, D)
        x = x.view(B, T, self.num_patches, self.embed_dim)  # (B, T, P, D)

        # 2. 空间建模
        for t in range(T):
            x_t = x[:, t] + self.pos_embed         # (B, P, D)
            for layer in self.layers:
                x_t = layer(x_t)
            x[:, t] = x_t

        # 3. 时序融合
        x = x + self.temporal_pos_embed.view(1, T, 1, -1)
        for blk in self.temporal_blocks:
            x = x + blk(x)

        # 4. 加权关键帧融合
        scores = self.key_frame_selector(x.mean(dim=2))     # (B, T, 1)
        scores = torch.softmax(scores, dim=1)
        key_tokens = (x * scores.unsqueeze(-1)).sum(dim=1)  # (B, P, D)

        # 5. reshape 输出为卷积格式 (B, D, Hp, Wp)
        key_tokens = self.norm(key_tokens)                  # (B, P, D)
        B, P, D = key_tokens.shape
        Hp, Wp = self.patch_h, self.patch_w                 # e.g. 16, 12
        out = key_tokens.permute(0, 2, 1).contiguous().view(B, D, Hp, Wp)  # (B, D, Hp, Wp)

        return out
