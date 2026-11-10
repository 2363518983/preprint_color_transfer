# -*- coding: utf-8 -*-
"""export_curve.py — 把训练好的 RGB 曲线导出为多格式交付物。

用法：python export_curve.py
产物（在 deliverables/ 目录）：
  rgb_curve.npy  : (3,33) 控制点，numpy 格式
  rgb_curve.acv  : Photoshop 曲线预设，调图师可直接加载
  rgb_curve.csv  : 文本值表，可用于文档/其他工具
"""
import os
import struct
import numpy as np
import torch

from config import Config
from models.curve_1d import Curve1D


def save_npy(curves_np, path):
    np.save(path, curves_np)
    print(f"[导出] {path}  shape={curves_np.shape}")


def save_acv(curves_np, path):
    """标准 Photoshop .acv（版本4，3通道，每通道33点，16bit 大端）。"""
    n_ch, n_pts = curves_np.shape
    with open(path, "wb") as f:
        f.write(struct.pack(">H", 4))      # 版本
        f.write(struct.pack(">H", n_ch))   # 通道数
        for ch in range(n_ch):
            f.write(struct.pack(">H", n_pts))  # 控制点数量
            xs = np.linspace(0, 65535, n_pts).astype(np.uint16)
            ys = (np.clip(curves_np[ch], 0, 1) * 65535).astype(np.uint16)
            for x, y in zip(xs, ys):
                f.write(struct.pack(">HH", int(x), int(y)))
    print(f"[导出] {path}")


def save_csv(curves_np, path):
    """人类可读的曲线表。"""
    with open(path, "w") as f:
        f.write("x,R,G,B\n")
        xs = np.linspace(0, 1, curves_np.shape[1])
        for i, x in enumerate(xs):
            f.write(f"{x:.6f},{curves_np[0,i]:.6f},{curves_np[1,i]:.6f},{curves_np[2,i]:.6f}\n")
    print(f"[导出] {path}")


def main():
    cfg = Config()
    cfg.setup_dirs()
    ckpt = os.path.join(cfg.CHECKPOINT_DIR, "curve_rgb_best.pth")
    if not os.path.exists(ckpt):
        print(f"⚠ 找不到 {ckpt}，请先运行 python train_rgb.py")
        return

    # 加载训练好的曲线
    curve = Curve1D(channels=3, n_bins=cfg.LUT_DIM)
    curve.load_state_dict(torch.load(ckpt, map_location="cpu"))
    curves_np = curve.get_curves().detach().cpu().numpy()  # (3, 33)

    print("学习到的 RGB 曲线控制点：")
    print(curves_np)

    out_dir = "deliverables"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "rgb_curve")

    save_npy(curves_np, base + ".npy")
    save_acv(curves_np, base + ".acv")
    save_csv(curves_np, base + ".csv")

    print(f"\n✅ 导出完成 → {out_dir}/")
    print("   rgb_curve.npy : np.load('deliverables/rgb_curve.npy') → (3,33)")
    print("   rgb_curve.acv  : Photoshop 曲线预设")
    print("   rgb_curve.csv  : 文本值表")


if __name__ == "__main__":
    main()