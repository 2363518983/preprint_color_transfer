# -*- coding: utf-8 -*-
"""
色彩空间转换（数值稳定，无 NaN）
包含 RGB ↔ CMYK（简化版）、RGB → Lab（标准 CIELAB）
注意：训练/推理中 CMYK 转换使用 ICC（utils/icc_color.py），此文件仅用于 Lab 损失和辅助。
"""
import torch

_RGB2XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float32)

_WHITE_D65 = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)


def _mat(device):
    return _RGB2XYZ.to(device)


def _f_safe(t):
    delta = 6.0 / 29.0
    delta3 = delta ** 3
    EPS = 1e-8
    root = torch.sign(t) * (t.abs() + EPS).pow(1.0 / 3.0)
    lin = t / (3.0 * delta ** 2) + 4.0 / 29.0
    w = torch.sigmoid(100.0 * (t - delta3))
    return w * root + (1.0 - w) * lin


def rgb_to_lab(rgb):
    """
    RGB (B,3,H,W) ∈ [0,1] → Lab (B,3,H,W)
    归一化: L∈[0,1], a∈[-1,1], b∈[-1,1] (L/100, a/128, b/128)
    """
    rgb = rgb.clamp(0, 1)
    device = rgb.device
    M = _mat(device)
    white = _WHITE_D65.to(device)

    shape = rgb.shape
    xyz = rgb.permute(0, 2, 3, 1).reshape(-1, 3) @ M.T
    xyz = xyz / white.unsqueeze(0)
    xyz = xyz.clamp(min=-0.5, max=2.0)

    fx = _f_safe(xyz[:, 0]); fy = _f_safe(xyz[:, 1]); fz = _f_safe(xyz[:, 2])

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    lab = torch.stack([L, a, b], dim=1)
    lab[:, 0] = lab[:, 0] / 100.0
    lab[:, 1] = lab[:, 1] / 128.0
    lab[:, 2] = lab[:, 2] / 128.0
    return lab.reshape(shape[0], 3, shape[2], shape[3])


# 以下为简化 CMYK 转换（仅用于兼容，实际训练请用 ICC 版本）
def rgb_to_cmyk_simple(rgb):
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    k = (1.0 - torch.max(torch.cat([r, g, b], dim=1), dim=1, keepdim=True)[0]).clamp(0, 1)
    eps = 1e-6
    c = (1.0 - r - k) / (1.0 - k + eps)
    m = (1.0 - g - k) / (1.0 - k + eps)
    y = (1.0 - b - k) / (1.0 - k + eps)
    return torch.cat([c, m, y, k], dim=1).clamp(0, 1)


def cmyk_to_rgb_simple(cmyk):
    c, m, y, k = cmyk[:, 0:1], cmyk[:, 1:2], cmyk[:, 2:3], cmyk[:, 3:4]
    r = (1.0 - c) * (1.0 - k)
    g = (1.0 - m) * (1.0 - k)
    b = (1.0 - y) * (1.0 - k)
    return torch.cat([r, g, b], dim=1).clamp(0, 1)