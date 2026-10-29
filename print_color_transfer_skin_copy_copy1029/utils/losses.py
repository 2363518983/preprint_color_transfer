# -*- coding: utf-8 -*-
"""软直方图（可导）+ 单调约束 + ΔE 评估。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftHistogramLoss(nn.Module):
    """三角核软分箱，梯度可回传。"""
    def __init__(self, bins=32, dark_w=1.0, mid_w=1.0, bright_w=1.0):
        super().__init__()
        self.bins = bins
        self.weights = torch.tensor([dark_w, mid_w, bright_w])

    def _soft_hist(self, x):
        N = x.numel()
        if N == 0:
            return torch.zeros(self.bins, device=x.device, dtype=x.dtype)
        bin_w = 1.0 / self.bins
        centers = (torch.arange(self.bins, device=x.device, dtype=x.dtype) + 0.5) * bin_w
        dist = (x.unsqueeze(1) - centers.unsqueeze(0)).abs()
        w = (1.0 - (dist / bin_w)).clamp(min=0.0)
        hist = w.sum(dim=0)
        return hist / (hist.sum() + 1e-8)

    def forward(self, pred, target):
        loss = 0.0
        n = self.bins
        n1 = n // 3
        seg = torch.cat([
            torch.full((n1,), self.weights[0], device=pred.device),
            torch.full((n1,), self.weights[1], device=pred.device),
            torch.full((n - 2 * n1,), self.weights[2], device=pred.device),
        ])
        for c in range(pred.size(1)):
            p = pred[:, c].reshape(-1)
            t = target[:, c].reshape(-1)
            hp, ht = self._soft_hist(p), self._soft_hist(t)
            loss = loss + ((hp - ht).abs() * seg).mean()
        return loss / pred.size(1)


class MonotoneLoss(nn.Module):
    """曲线单调约束（cumsum 下理想为 0）。"""
    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin

    def forward(self, curves):
        if curves.dim() == 3:
            curves = curves.mean(dim=0)
        diff = curves[:, 1:] - curves[:, :-1]
        return F.relu(self.margin - diff).mean()


def _lab_f(t):
    d = 6.0 / 29.0
    return torch.where(t > d**3, t.pow(1.0 / 3.0), t / (3 * d**2) + 4.0 / 29.0)


def rgb_to_lab(rgb):
    rgb = rgb.clamp(0, 1)
    mask = rgb <= 0.04045
    lin = torch.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    xn, yn, zn = 0.95047, 1.0, 1.08883
    fx = _lab_f(x / xn); fy = _lab_f(y / yn); fz = _lab_f(z / zn)
    return torch.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], dim=-1)


def ciede2000_rgb(rgb_pred, rgb_target):
    """RGB → 逐像素 ΔE（此处用 ΔE76 近似，可导；验收可用离线严格 ΔE2000 复核）。"""
    def _to_2d(img):
        return img if img.dim() == 4 else img.unsqueeze(0)
    p, t = _to_2d(rgb_pred), _to_2d(rgb_target)
    Lp, ap, bp = rgb_to_lab(p).chunk(3, dim=1)
    Lt, at, bt = rgb_to_lab(t).chunk(3, dim=1)
    dL, da, db = Lp - Lt, ap - at, bp - bt
    return torch.sqrt(dL**2 + da**2 + db**2 + 1e-8).mean()

def rgb_to_lab_pytorch(rgb):
    """可导 sRGB→Lab（D65）。rgb: (B,3,H,W)∈[0,1]。用于损失计算。"""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

    def lin(c):
        return torch.where(c > 0.04045,
                           ((c + 0.055) / 1.055) ** 2.4,
                           c / 12.92)

    rl, gl, bl = lin(r), lin(g), lin(b)

    x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl

    def f(t):
        delta = 6 / 29
        return torch.where(t > delta ** 3,
                           torch.pow(t, 1 / 3),
                           t / (3 * delta ** 2) + 4 / 29)

    fx, fy, fz = f(x / 0.95047), f(y / 1.00000), f(z / 1.08883)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return torch.cat([L, a, b_val], dim=1)


def ciede2000_rgb(pred, target):
    """可导 ΔE 代理：Lab 空间 L2 范数（方向与 CIEDE2000 一致）。"""
    pred_lab = rgb_to_lab_pytorch(pred)
    tgt_lab = rgb_to_lab_pytorch(target)
    delta = pred_lab - tgt_lab
    return torch.mean(torch.sqrt((delta ** 2).sum(dim=1) + 1e-8))
if __name__ == "__main__":
    x = torch.rand(2, 4, 8, 8, requires_grad=True)
    y = torch.rand(2, 4, 8, 8)
    sh = SoftHistogramLoss(bins=16)
    loss = sh(x, y)
    loss.backward()
    print("SoftHistogramLoss:", loss.item(), "| grad exists:", x.grad.abs().sum().item() > 0)
    print("MonotoneLoss(单调):", MonotoneLoss()(torch.randn(4, 33).cumsum(dim=1)).item())