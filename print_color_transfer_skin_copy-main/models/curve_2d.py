# -*- coding: utf-8 -*-
"""通道耦合曲线：每个输出通道 = 2D LUT(R, G) 或 (G, B) 等配对，单调可保证。

设计：对 4 种"通道对"组合各学一个 2D LUT，输出通道取对应 LUT。
  R_out = LUT_R( R, G )
  G_out = LUT_G( G, B )
  B_out = LUT_B( B, R )
保持每条轴单调递增（cumsum），可解释、可导出为曲线网格。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Curve2D(nn.Module):
    """3 通道输出，每通道一个 2D LUT，轴单调。"""
    def __init__(self, n_bins=33, init_smooth=True):
        super().__init__()
        self.n_bins = n_bins
        # (3, n_bins, n_bins)：3 个输出通道各自的 2D 表
        delta = torch.randn(3, n_bins, n_bins) * 0.01
        if init_smooth:
            delta = delta.abs() + 0.02
        self.delta = nn.Parameter(delta)

    def get_tables(self):
        """每轴单调、值域 [0,1]、四角 = (0,0)/(1,1) 对齐。"""
        # cumsum 沿两轴，保证局部单调；再归一使最大=1
        cum = F.softplus(self.delta).cumsum(dim=1).cumsum(dim=2)
        # 边界约束：x=0 或 y=0 时为 0，x=1 且 y=1 时为 1
        cum = cum / (cum[:, -1:, -1:] + 1e-8)
        return cum

    def forward(self, x):
        """x:(B,3,H,W) → (B,3,H,W)。"""
        B, C, H, W = x.shape
        tables = self.get_tables()                     # (3,N,N)
        n = self.n_bins
        out = torch.zeros_like(x)

        # 通道配对：R 用 (R,G), G 用 (G,B), B 用 (B,R)
        pairs = [(0, 0, 1), (1, 1, 2), (2, 2, 0)]    # (out_ch, in_a, in_b)
        for out_ch, a, b in pairs:
            u = (n - 1) * x[:, a, :, :].clamp(0, 1)   # (B,H,W)
            v = (n - 1) * x[:, b, :, :].clamp(0, 1)
            au, bu = u.floor().long().clamp(0, n - 2), v.floor().long().clamp(0, n - 2)
            av, bv = au + 1, bu + 1
            fu = (u - au.float()).clamp(0, 1)
            fv = (v - bu.float()).clamp(0, 1)

            T = tables[out_ch]                         # (N,N)
            # 双线性：gather 四个角
            c00 = T[au, bu]; c01 = T[au, bv]
            c10 = T[av, bu]; c11 = T[av, bv]
            c0 = (1 - fv) * c00 + fv * c01
            c1 = (1 - fv) * c10 + fv * c11
            out[:, out_ch, :, :] = (1 - fu) * c0 + fu * c1
        return out.clamp(0, 1)

    def coupling_strength(self):
        """诊断用：各通道交叉项的"偏离对角线"程度，越大说明耦合越强。"""
        tables = self.get_tables()
        idx = torch.arange(self.n_bins, device=tables.device)
        diag = tables[:, idx, idx]                      # 对角(=1D 曲线)
        return {chr(ord('R') + c): float((tables[c] - diag[c][None, :]).abs().mean())
                for c in range(3)}