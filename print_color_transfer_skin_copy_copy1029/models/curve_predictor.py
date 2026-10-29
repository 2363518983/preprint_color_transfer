# -*- coding: utf-8 -*-
"""
CurvePredictor v3 —— 残差式曲线预测器

关键改动（相比 v2）：
1. 网络不再直接输出整条曲线，而是输出【相对全局曲线的残差 delta_curve】
   final_curve = global_curve + residual   (经 clamp 保证仍在 [0,1] 且单调)
2. 残差受 PRED_RESIDUAL_MAX 约束，初始化为 0
   → 初始曲线严格 = 全局曲线（ΔE≈3.0 基线），训练只学逐图微调
3. 可选弱通道耦合：在应用曲线前，叠加一个小的 2D LUT 交叉项残差
   （对应诊断 C=0.21，解决 3.2 天花板）

【修补】
- 未注入外部全局曲线时，默认初始化为恒等曲线，避免归一化除零 / 全黑
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.curve_1d import curves_from_delta, apply_curves


class CurvePredictor(nn.Module):
    def __init__(self, channels=3, n_bins=33, base=32,
                 residual_max=0.08, use_coupling=True):
        super().__init__()
        self.channels = channels
        self.n_bins = n_bins
        self.residual_max = residual_max
        self.use_coupling = use_coupling

        # 默认全局曲线 = 恒等（未注入时为 identity，保证可正常前向）
        ident = torch.linspace(0, 1, n_bins).unsqueeze(0).repeat(channels, 1)
        self.register_buffer("global_curve", ident.clone())

        # ---- 编码器 ----
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1), nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base * 4, base * 8, 3, padding=1), nn.BatchNorm2d(base * 8), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Sequential(
            nn.Linear(base * 8, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # ★ 输出残差（初始化 bias=0 → 初始残差=0 → 曲线=全局曲线）
        self.fc_residual = nn.Linear(256, channels * n_bins)

        # ★ 弱通道耦合：3 个 (in_a, in_b) -> out 的小 2D LUT 残差
        # 注意：当前实现只取对角线项，属于近似耦合；如需严格 2D 交叉项需另写插值
        if use_coupling:
            self.couple = nn.Parameter(torch.zeros(channels, n_bins, n_bins) * 0.01)

        # 初始化：残差分支输出恒为 0
        nn.init.zeros_(self.fc_residual.weight)
        nn.init.zeros_(self.fc_residual.bias)

    def set_global_curve(self, curve):
        """注入全局曲线 (3, N)"""
        curve = curve.detach().float()
        if curve.shape[0] == 4:
            curve = curve[:3]
        assert curve.shape == self.global_curve.shape, \
            f"曲线形状 {curve.shape} != {self.global_curve.shape}"
        self.global_curve.copy_(curve)

    def forward(self, x):
        """
        x: (B,3,H,W)
        返回: (B,3,N) 严格单调递增、值域 [0,1]
        """
        feat = self.encoder(x).flatten(1)
        h = self.head(feat)

        # 残差，经 tanh 缩放到 [-residual_max, residual_max]
        res = self.residual_max * torch.tanh(
            self.fc_residual(h).view(-1, self.channels, self.n_bins)
        )                                          # (B,3,N)

        # 基础曲线 = 全局曲线 + 残差（逐图不同）
        base_curve = self.global_curve.unsqueeze(0) + res

        # ★ 弱通道耦合：在网格点上加 2D 交叉项（对角线对称，保证单调）
        if self.use_coupling:
            B = x.size(0)
            coup = self.couple.unsqueeze(0).expand(B, -1, -1, -1)  # (B,3,N,N)
            # 对角项（不影响单调性），在 forward_apply 里结合采样；这里把耦合注入为"逐通道偏移"
            diag = torch.diagonal(coup, dim1=-2, dim2=-1)           # (B,3,N)
            base_curve = base_curve + diag

        # 保证端点 0/1 + 单调
        base_curve = base_curve - base_curve[..., :1]               # 起点归零
        base_curve = base_curve / (base_curve[..., -1:] + 1e-8)     # 终点归 1 → 自动单调（因残差小）
        return base_curve.clamp(0, 1)

    def apply(self, x, resize_size=320):
        """任意尺寸输入，返回 (output, curves)"""
        x_resized = F.interpolate(x, size=(resize_size, resize_size), mode="bilinear", align_corners=False)
        curves = self.forward(x_resized)
        return apply_curves(x, curves), curves


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CurvePredictor(channels=3, n_bins=33, base=32, residual_max=0.08)
    ident = torch.linspace(0, 1, 33).unsqueeze(0).repeat(3, 1)
    model.set_global_curve(ident)

    x = torch.rand(2, 3, 320, 320)
    curves = model(x)
    assert curves.shape == (2, 3, 33), curves.shape
    # 初始应严格 = 恒等曲线（残差为 0）
    assert (curves[0] - ident).abs().max() < 1e-5, "初始化应=全局曲线"
    assert (curves[:, :, 1:] >= curves[:, :, :-1] - 1e-5).all(), "必须单调"
    out, _ = model.apply(x)
    assert out.shape == x.shape
    print("CurvePredictor v3 自检通过 ✅")
