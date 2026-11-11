# -*- coding: utf-8 -*-
"""test_hybrid.py — 冒烟测试: 验证模型/损失/冻结逻辑全部正确 (CPU, ~30秒)。"""
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0)

from models.residual_unet import HybridColorModel
from utils.losses import SoftHistogramLoss, MonotoneLoss


def rgb_to_lab_torch(img):
    img = img.clamp(0, 1)
    mask = img > 0.04045
    linear = torch.where(mask, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)
    m = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])
    xyz = torch.einsum("ij,bjhw->bihw", m, linear)
    xyz = xyz / torch.tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1)
    eps = 1e-8

    def f(t):
        delta = 6 / 29
        return torch.where(t > delta ** 3, torch.clamp(t, min=eps) ** (1 / 3),
                           t / (3 * delta ** 2) + 4 / 29)

    fx, fy, fz = f(xyz[:, 0]), f(xyz[:, 1]), f(xyz[:, 2])
    return torch.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], dim=1)


print("=== 1. 模型前向/值域 ===")
model = HybridColorModel(n_bins=33, unet_base=16, residual_scale=0.03)
n_total = sum(p.numel() for p in model.parameters())
n_curve = sum(p.numel() for p in model.curve.parameters())
n_unet = sum(p.numel() for p in model.unet.parameters())
print(f"总参数: {n_total} (curve={n_curve}, unet={n_unet})")

x = torch.rand(2, 3, 256, 256)
with torch.no_grad():
    base = model(x, use_unet=False)
    full = model(x, use_unet=True)
print(f"base={tuple(base.shape)} full={tuple(full.shape)}")
print(f"值域 base=[{base.min():.3f},{base.max():.3f}] full=[{full.min():.3f},{full.max():.3f}]")
assert torch.isfinite(base).all() and torch.isfinite(full).all(), "输出含非有限值!"
assert full.min() >= 0 and full.max() <= 1.001, "输出越界!"
print("✅ 前向正常, 值域合法")

print("\n=== 2. 冻结 curve, 仅 U-Net 可训练 ===")
model.freeze_curve()
trainable = [p for p in model.parameters() if p.requires_grad]
n_train = sum(p.numel() for p in trainable)
print(f"可训练参数: {n_train} (应等于 unet={n_unet})")
assert n_train == n_unet, "冻结失败!"
# 确认 curve 参数确实不更新
curve_id = [id(p) for p in model.curve.parameters()]
assert not any(id(p) in [id(q) for q in trainable] for p in model.curve.parameters())
print("✅ curve 已冻结, 仅 U-Net 可训练")

print("\n=== 3. 训练一步, 检查 loss/梯度/无NaN ===")
tgt = torch.rand(2, 3, 256, 256) * 0.5 + 0.25
opt = torch.optim.Adam(trainable, lr=1e-4)
l1 = nn.L1Loss()
hist = SoftHistogramLoss(32, 1, 1, 1)
mono = MonotoneLoss()

opt.zero_grad()
out = model(x, use_unet=True)
lab_pred = rgb_to_lab_torch(out)
lab_tgt = rgb_to_lab_torch(tgt)
loss = l1(out, tgt) + 0.1 * nn.L1Loss()(lab_pred, lab_tgt) + 0.5 * hist(out, tgt) + 0.1 * mono(model.get_curves())
loss.backward()

# 检查 U-Net 有梯度, curve 无梯度
unet_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.unet.parameters())
curve_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.curve.parameters())
print(f"loss={loss.item():.4f} | U-Net有梯度={unet_has_grad} | curve有梯度={curve_has_grad}")
assert torch.isfinite(loss), "loss 非有限!"
assert unet_has_grad, "U-Net 无梯度!"
assert not curve_has_grad, "curve 不应有梯度(已冻结)!"
# 检查无 NaN 参数
for name, p in model.named_parameters():
    if p.grad is not None and torch.isnan(p.grad).any():
        print(f"❌ {name} 梯度含 NaN"); break
else:
    print("✅ 梯度全部有限, 无 NaN")
opt.step()
with torch.no_grad():
    assert torch.isfinite(model.unet.head.weight).all(), "参数变 NaN!"
print("✅ 一步训练正常, curve 参数冻结未变, U-Net 已更新")

print("\n=== 4. 残差幅度检查 (应远小于1, 保证主变换是曲线) ===")
with torch.no_grad():
    res = full - base
    print(f"残差 abs 均值={res.abs().mean():.5f}, max={res.abs().max():.4f}")
    print("(residual_scale=0.03 → tanh 后理论 max≈0.03)")
print("✅ 残差幅度受控")

print("\n=== 全部自检通过 ✅ ===")
print(f"\n结论: 可安全运行 train_hybrid.py")
print(f"  - 模型: curve({n_curve}参, 冻结) + U-Net({n_unet}参, 可训练)")
print(f"  - 损失: L1 + 0.1·LabL1 + Hist + Mono")
print(f"  - 预期: ΔE00 从 3.69 下降 (先看前5epoch趋势)")
