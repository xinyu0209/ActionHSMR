import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalPatchFusion(nn.Module):
    def __init__(self, D, T=3):
        super().__init__()
        self.attn_linear = nn.Linear(D, 1, bias=False)  # 可学权重，关注每帧
        # 或直接用nn.Parameter手动设置偏向中帧的权重

    def forward(self, x):
        # x: [B, T, D, Hp, Wp]
        B, T, D, Hp, Wp = x.shape
        x = x.permute(0, 3, 4, 1, 2)  # [B, Hp, Wp, T, D]
        x_flat = x.reshape(-1, T, D)   # [B*Hp*Wp, T, D]
        
        # Attention权重
        attn_score = self.attn_linear(x_flat)    # [B*Hp*Wp, T, 1]
        attn_score = attn_score.squeeze(-1)      # [B*Hp*Wp, T]
        attn_weight = F.softmax(attn_score, dim=1)  # [B*Hp*Wp, T]
        
        # 加权融合
        fused = torch.sum(x_flat * attn_weight.unsqueeze(-1), dim=1)  # [B*Hp*Wp, D]
        fused = fused.view(B, Hp, Wp, D).permute(0, 3, 1, 2)         # [B, D, Hp, Wp]
        
        return fused
