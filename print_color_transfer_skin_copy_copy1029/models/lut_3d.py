# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class LUT3D(nn.Module):
    """
    3D LUT 模块（RGB → RGB 查找表，端到端可微）
    强制 float32：F.grid_sample 要求 input/grid 同精度
    """
    def __init__(self, dim=33, channels=3):
        super().__init__()
        self.dim = dim
        self.channels = channels

        # 恒等初始化：lut[c][i][j][k] 在坐标 (i,j,k) 处输出 c 通道值
        identity = torch.zeros(channels, dim, dim, dim, dtype=torch.float32)
        coords = torch.linspace(0, 1, dim, dtype=torch.float32)
        # 手动广播构造网格（兼容旧版 torch，不用 indexing=）
        grid_z, grid_y, grid_x = torch.meshgrid(coords, coords, coords)  # (D,D,D)
        identity[0] = grid_x
        identity[1] = grid_y
        identity[2] = grid_z
        self.lut = nn.Parameter(identity)  # (C, D, D, D)

    def forward(self, x):
        """
        x: (B, 3, H, W) RGB，取值范围 [0, 1]，任意 dtype
        返回: (B, 3, H, W)，与 x 同 dtype（外部调用处统一用 float32）
        """
        # ---- 关键修复：全程强制 float32 ----
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        self.lut.data = self.lut.data.to(torch.float32)

        B, C, H, W = x.shape
        assert C == 3, f"LUT 输入必须为 3 通道 RGB，得到 {C}"

        # 采样网格: (B, H, W, 3)，值域 [-1, 1]
        grid = x.permute(0, 2, 3, 1) * 2.0 - 1.0   # (B, H, W, 3)

        # LUT 扩展到 batch: (B, C, D, D, D)
        lut_batch = self.lut.unsqueeze(0).expand(B, -1, -1, -1, -1)

        # 三线性插值（grid_sample 要求 5D input + 5D grid）
        grid_5d = grid.unsqueeze(1)                  # (B, 1, H, W, 3)
        out = F.grid_sample(
            lut_batch, grid_5d,
            mode='bilinear', padding_mode='border', align_corners=True
        )  # (B, C, 1, H, W)
        out = out.squeeze(2)                         # (B, C, H, W)

        return out.clamp(0, 1).to(orig_dtype)

    def extra_repr(self):
        return f"dim={self.dim}, channels={self.channels}, params={self.lut.numel()}"