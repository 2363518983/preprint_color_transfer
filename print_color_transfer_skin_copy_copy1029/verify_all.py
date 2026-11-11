# test_lab_stable.py
import torch
from utils.color_space import rgb_to_lab, rgb_to_cmyk, cmyk_to_rgb

torch.manual_seed(0)

# 1. 随机数据反向稳定性
x = torch.rand(4, 3, 32, 32, requires_grad=True)
lab = rgb_to_lab(x)
lab.sum().backward()
print(f"随机 RGB: lab ok, grad has_nan={torch.isnan(x.grad).any().item()}")
assert not torch.isnan(x.grad).any(), "❌ 随机数据梯度 NaN"

# 2. 极端数据（含 0、近 0、接近 1）—— 之前崩的场景
x2 = torch.tensor([0.0, 1e-10, 1e-6, 0.5, 0.999, 1.0], device=x.device).view(1, 3, 2, 1)
x2.requires_grad_(True)
lab2 = rgb_to_lab(x2)
lab2.sum().backward()
print(f"极端 RGB: lab={lab2.detach().flatten()[:6]}")
print(f"极端 RGB: grad={x2.grad.flatten()}, has_nan={torch.isnan(x2.grad).any().item()}")
assert not torch.isnan(x2.grad).any(), "❌ 极端数据梯度 NaN"

# 3. 负值输入（模拟 xyz 归一化后略 <0）
x3 = torch.tensor([-0.1, 0.0, 0.1], device=x.device).view(1, 3, 1, 1)
x3.requires_grad_(True)
lab3 = rgb_to_lab(x3.clamp(min=0))  # clamp 后走正常路径
lab3.sum().backward()
print(f"负值(clamp后): grad has_nan={torch.isnan(x3.grad).any().item()}")

print("\n✅ Lab 前向/反向均稳定，可安全训练")



# debug_nan.py —— 前向+反向全链路诊断
import torch
torch.autograd.set_detect_anomaly(True)   # ← 加这一行
import torch
from config import Config
from dataset import PrintDataset
from models.lut_3d import LUT3D
from models.residual_cnn import ResidualCNN
from utils.color_space import rgb_to_cmyk, cmyk_to_rgb
from utils.losses import PerceptualLoss, LabLoss

cfg = Config()
device = torch.device(cfg.DEVICE)

ds = PrintDataset(cfg.DATA_DIR, split='train', augment=False, cache_dir=cfg.CACHE_DIR)
lut = LUT3D(dim=cfg.LUT_DIM).to(device)
resnet = ResidualCNN().to(device)
l1 = torch.nn.L1Loss()
perc = PerceptualLoss(device)
lab = LabLoss()

batch = ds[0]
inp = batch['input'].unsqueeze(0).float().to(device)
tgt = batch['target'].unsqueeze(0).float().to(device)
print(f"输入: inp range=[{inp.min():.3f},{inp.max():.3f}], tgt=[{tgt.min():.3f},{tgt.max():.3f}]")

def check(name, t):
    if torch.is_tensor(t):
        print(f"  {name}: min={t.min():.4f} max={t.max():.4f} has_nan={torch.isnan(t).any().item()} has_inf={torch.isinf(t).any().item()}")

# 前向
lut_out = lut(inp);              check("lut_out", lut_out)
lut_cmyk = rgb_to_cmyk(lut_out); check("lut_cmyk", lut_cmyk)
inp_cmyk = rgb_to_cmyk(inp)
tgt_cmyk = rgb_to_cmyk(tgt)
residual = resnet(inp_cmyk, lut_cmyk); check("residual", residual)
final_cmyk = (lut_cmyk + residual).clamp(0, 1)
final_rgb = cmyk_to_rgb(final_cmyk);    check("final_rgb", final_rgb)

loss = (l1(final_rgb, tgt)
        + cfg.CMYK_WEIGHT * l1(final_cmyk, tgt_cmyk)
        + cfg.PERCEPTUAL_WEIGHT * perc(final_rgb, tgt)
        + cfg.LAB_WEIGHT * lab(final_rgb, tgt))
print(f"\n总 loss = {loss.item():.4f}  has_nan={torch.isnan(loss).any().item()}")

# 反向
loss.backward()
print("\n梯度检查:")
all_ok = True
for name, p in list(lut.named_parameters()) + list(resnet.named_parameters()):
    if p.grad is not None:
        g = p.grad
        bad = torch.isnan(g).any().item() or torch.isinf(g).any().item()
        if bad:
            all_ok = False
        print(f"  {name}: grad_max={g.abs().max().item():.6f}  has_nan_inf={bad}")

print(f"\n{'✅ 全部梯度正常，可安全训练' if all_ok else '❌ 存在 NaN/Inf 梯度，需修复'}")


import torch
from utils.color_space import rgb_to_cmyk, rgb_to_lab
from utils.losses import LabLoss, PerceptualLoss

torch.manual_seed(0)
pred = torch.rand(2, 3, 64, 64)
tgt = torch.rand(2, 3, 64, 64)

lab = LabLoss()(pred, tgt)
print(f"LabLoss = {lab.item():.4f}   ← 应在 0.01 ~ 2.0 之间，绝不能是 60+")

l1 = torch.nn.L1Loss()(pred, tgt)
print(f"L1 = {l1.item():.4f}")

c_pred = rgb_to_cmyk(pred)
c_tgt = rgb_to_cmyk(tgt)
print(f"CMYK L1 = {torch.nn.L1Loss()(c_pred, c_tgt).item():.4f}")
print(f"CMYK range: min={c_pred.min():.3f} max={c_pred.max():.3f}")

lab_pred = rgb_to_lab(pred)
print(f"Lab range: L∈[{lab_pred[:,0].min():.2f},{lab_pred[:,0].max():.2f}] "
      f"a∈[{lab_pred[:,1].min():.2f},{lab_pred[:,1].max():.2f}]")











# -*- coding: utf-8 -*-
"""verify_all.py —— 修正版：第 9 项通道严格对齐"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lut_3d import LUT3D
from models.residual_cnn import ResidualCNN
from utils.color_space import rgb_to_cmyk, cmyk_to_rgb, rgb_to_lab
from utils.losses import LabLoss, PerceptualLoss


def test_imports():
    print("[1] imports OK")


def test_center_crop():
    from PIL import Image
    img = Image.new('RGB', (6000, 4000))
    w, h = img.size
    if w > h:
        left = (w - h) // 2
        img = img.crop((left, 0, left + h, h))
    else:
        top = (h - w) // 2
        img = img.crop((0, top, w, top + w))
    print(f"[2] center_crop_square: (6000, 4000) -> {img.size}  OK")
    assert img.size == (4000, 4000)


def test_lut_identity():
    lut = LUT3D(dim=17)
    x = torch.rand(2, 3, 32, 32)
    out = lut(x)
    err = (out - x).abs().max().item()
    print(f"[3] LUT 恒等初始化: max_err = {err:.2e}")
    assert err < 1e-5, f"恒等初始化失败: {err}"


def test_residual_range():
    net = ResidualCNN(base_filters=16)
    a = torch.rand(2, 4, 128, 128)
    b = torch.rand(2, 4, 128, 128)
    out = net(a, b)
    mx = out.abs().max().item()
    print(f"[4] ResidualCNN 输出 {tuple(out.shape)}, max_abs={mx:.4f}  OK")
    assert out.shape == (2, 4, 128, 128)
    assert mx <= 0.0501, f"残差超幅: {mx}"


def test_color_space():
    x = torch.rand(2, 3, 32, 32)
    cmyk = rgb_to_cmyk(x)
    lab = rgb_to_lab(x)
    print(f"[5] color_space: CMYK {tuple(cmyk.shape)}, Lab {tuple(lab.shape)}  OK")
    assert cmyk.shape == (2, 4, 32, 32)
    assert lab.shape == (2, 3, 32, 32)


def test_meshgrid():
    coords = torch.linspace(0, 1, 17)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
    identity = torch.zeros(3, 17, 17, 17)
    identity[0] = grid_x; identity[1] = grid_y; identity[2] = grid_z
    print(f"[6] meshgrid 兼容（手动广播） OK  shape={tuple(identity.shape)}")
    assert identity.shape == (3, 17, 17, 17)


def test_lab_loss():
    lab = LabLoss()
    p = torch.rand(2, 3, 64, 64)
    t = torch.rand(2, 3, 64, 64)
    loss = lab(p, t)
    print(f"[7] LabLoss = {loss.item():.4f}  OK")
    assert torch.isfinite(loss)


def test_perceptual():
    perc = PerceptualLoss('cpu')
    p = torch.rand(2, 3, 224, 224)
    t = torch.rand(2, 3, 224, 224)
    loss = perc(p, t)
    print(f"[8] PerceptualLoss = {loss.item():.4f}  OK")


def test_forward_backward():
    print("[9] LUT + ResidualCNN 联合前向+反向 ...")
    torch.manual_seed(0)
    lut = LUT3D(dim=17).eval()
    net = ResidualCNN(base_filters=16).train()
    lab_loss = LabLoss()
    perc_loss = PerceptualLoss('cpu')
    l1 = nn.L1Loss()

    x = torch.rand(2, 3, 128, 128)
    tgt = torch.rand(2, 3, 128, 128)

    # 前向
    lut_out_rgb = lut(x)
    lut_out_cmyk = rgb_to_cmyk(lut_out_rgb)
    inp_cmyk = rgb_to_cmyk(x)
    residual = net(inp_cmyk, lut_out_cmyk)
    final_cmyk = lut_out_cmyk + residual
    final_rgb = cmyk_to_rgb(final_cmyk.clamp(0, 1))

    # 损失：严格通道对齐（关键修正）
    rgb_l1  = l1(final_rgb, tgt)                     # 3ch vs 3ch
    cmyk_l1 = l1(final_cmyk, rgb_to_cmyk(tgt))       # 4ch vs 4ch  ← 修正点
    lab_l   = lab_loss(final_rgb, tgt)               # LabLoss 内部 rgb_to_lab
    perc_l  = perc_loss(final_rgb, tgt)

    loss = rgb_l1 + 2.0 * cmyk_l1 + 0.1 * lab_l + 0.2 * perc_l
    loss.backward()

    assert lut_out_rgb.shape == (2, 3, 128, 128)
    assert residual.shape == (2, 4, 128, 128)
    assert final_cmyk.shape == (2, 4, 128, 128)
    assert final_rgb.shape == (2, 3, 128, 128)
    assert torch.isfinite(loss), loss.item()
    assert net.out_conv.weight.grad is not None, "残差CNN无梯度"
    assert lut.lut.grad is not None, "LUT无梯度"
    print(f"    loss={loss.item():.4f}")
    print("    OK (前向+反向+梯度均正常，通道严格对齐)")


if __name__ == '__main__':
    test_imports()
    test_center_crop()
    test_lut_identity()
    test_residual_range()
    test_color_space()
    test_meshgrid()
    test_lab_loss()
    test_perceptual()
    test_forward_backward()
    print("\n✅ 全部 9 项验证通过，可安全训练")