# -*- coding: utf-8 -*-
"""
全局 1D 曲线：cumsum 重参数化保证严格单调。
提供公共函数 curves_from_delta / apply_curves，供 CurvePredictor 复用。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def curves_from_delta(delta):
    """
    delta: (..., C, N) -> curves: (..., C, N)
    每条曲线严格单调递增，起点 0，终点 1。
    """
    cum = F.softplus(delta).cumsum(dim=-1)
    cum = cum - cum[..., :1]
    return cum / (cum[..., -1:] + 1e-8)


def apply_curves(x, curves):
    """
    x: (B, C, H, W)
    curves: (B, C, N) 或 (C, N)（自动扩展 batch）
    返回与 x 同形状的曲线应用结果。
    """
    orig_shape = x.shape
    B, C, H, W = orig_shape
    N = curves.shape[-1]

    if curves.dim() == 2:
        curves = curves.unsqueeze(0).expand(B, -1, -1)

    idx_f = (N - 1) * x.clamp(0, 1)
    lo = idx_f.floor().long().clamp(0, N - 2)
    hi = lo + 1
    frac = (idx_f - lo.float()).clamp(0, 1)

    L = H * W
    flat_lo = lo.reshape(B, C, L)
    flat_hi = hi.reshape(B, C, L)
    flat_frac = frac.reshape(B, C, L)

    c_lo = curves.gather(2, flat_lo)
    c_hi = curves.gather(2, flat_hi)
    out = (1 - flat_frac) * c_lo + flat_frac * c_hi
    return out.reshape(orig_shape).clamp(0, 1)


class Curve1D(nn.Module):
    def __init__(self, channels=4, n_bins=33, init_smooth=True):
        super().__init__()
        self.channels = channels
        self.n_bins = n_bins
        delta = torch.randn(channels, n_bins) * 0.01
        if init_smooth:
            delta = delta.abs() + 0.02
        self.delta = nn.Parameter(delta)

    def get_curves(self):
        return curves_from_delta(self.delta)

    def forward(self, x):
        curves = self.get_curves().unsqueeze(0)
        return apply_curves(x, curves)

    def curve_prior_loss(self):
        curves = self.get_curves()
        ident = torch.linspace(0, 1, self.n_bins, device=curves.device)
        return ((curves - ident.unsqueeze(0)) ** 2).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    m = Curve1D(4, 33)
    x = torch.rand(8, 4, 64, 64)
    out = m(x)
    assert out.shape == x.shape
    curves = m.get_curves()
    assert (curves[:, 1:] >= curves[:, :-1] - 1e-6).all()
    print("Curve1D 自检通过")