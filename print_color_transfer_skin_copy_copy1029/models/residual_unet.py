# -*- coding: utf-8 -*-
"""
残差 U-Net v2（可学习残差门控）：
- out_conv 用 Kaiming 初始化（正常）
- ★ residual_gate 可学习参数，初始 0.01 → 初始残差 std≈0.0067
- 训练时 gate 自动决定残差幅度，无需外部 scale
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.curve_1d import Curve1D


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            _, _, h, w = x.shape
            skip = F.interpolate(skip, size=(h, w), mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResidualUNet(nn.Module):
    """输入 (B,6,H,W) → 输出 (B,3,H,W)。
    ★ 输出的残差被 residual_gate 缩放，初始≈0.0067，安全接入主链路。
    """

    def __init__(self, in_ch=6, base=48, out_ch=3):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4

        self.enc1 = DoubleConv(in_ch, c1)
        self.enc2 = Down(c1, c2)
        self.enc3 = Down(c2, c3)
        self.bottleneck = Down(c3, c3)

        self.up3 = Up(c3, c3, c2)
        self.up2 = Up(c2, c2, c1)
        self.up1 = Up(c1, c1, c1)

        self.out_conv = nn.Conv2d(c1, out_ch, 1)
        nn.init.kaiming_normal_(self.out_conv.weight, mode="fan_in", nonlinearity="linear")

        # ★★★ 可学习残差门控（初始 0.01）★★★
        self.residual_gate = nn.Parameter(torch.tensor(0.01))

    def forward(self, x, B=None):
        c1 = self.enc1(x)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        b = self.bottleneck(c3)

        u3 = self.up3(b, c3)
        u2 = self.up2(u3, c2)
        u1 = self.up1(u2, c1)
        raw = self.out_conv(u1)                      # (B,3,H,W)，std≈0.67
        return raw * self.residual_gate              # 初始 std≈0.0067


def _check_bchw(name, t, expected_b=None):
    if t.dim() != 4:
        raise RuntimeError(f"{name} 应为 BCHW，实际 {tuple(t.shape)}")
    if expected_b is not None and t.size(0) != expected_b:
        raise RuntimeError(f"{name} batch 维应为 {expected_b}，实际 {t.size(0)}")


class HybridColorModel(nn.Module):
    """Curve1D（全局）+ ResidualUNet（局部残差）。供旧训练脚本使用。"""

    def __init__(self, n_bins=33, unet_base=32, residual_scale=0.03,
                 pretrained_curve=None):
        super().__init__()
        self.curve = Curve1D(channels=3, n_bins=n_bins)
        self.unet = ResidualUNet(in_ch=6, base=unet_base, out_ch=3)
        self.residual_scale = residual_scale
        if pretrained_curve is not None:
            self._load_curve(pretrained_curve)

    def _load_curve(self, path):
        if isinstance(path, str):
            if path.endswith(".npy"):
                curves = torch.from_numpy(np.load(path)).float()
                with torch.no_grad():
                    self.curve.curves.copy_(curves)
                return
            sd = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "curves" in sd:
                sd = {"curves": sd["curves"]}
            self.curve.load_state_dict(sd, strict=False)
        else:
            self.curve.load_state_dict(path, strict=False)

    def freeze_curve(self):
        for p in self.curve.parameters():
            p.requires_grad = False
        self.curve.eval()

    def apply_curves(self, x):
        return self.curve(x).clamp(0, 1)

    def forward_base(self, x):
        return self.apply_curves(x)

    def forward(self, x, use_unet=True):
        base = self.apply_curves(x)
        _check_bchw("base", base, x.size(0))
        if not use_unet:
            return base
        residual = self.unet(torch.cat([x, base], dim=1), B=x.size(0))
        return (base + self.residual_scale * residual).clamp(0, 1)


if __name__ == "__main__":
    torch.manual_seed(0)
    unet = ResidualUNet(in_ch=6, base=48, out_ch=3)
    x = torch.rand(2, 6, 256, 256)
    out = unet(x)
    print(f"初始化: shape={tuple(out.shape)}, std={out.std().item():.6f}")
    print(f"residual_gate: {unet.residual_gate.item():.4f}")

    loss = out.abs().mean()
    loss.backward()
    grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in unet.parameters())
    print(f"梯度流通: {'✅' if grad_ok else '❌'}")
    # 期望 std ≈ 0.67 * 0.01 ≈ 0.0067