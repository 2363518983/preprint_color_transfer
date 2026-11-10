# -*- coding: utf-8 -*-
"""convert_to_cmyk.py — 对任意 RGB 图片应用训练好的曲线，然后 ICC 转 CMYK。

用法：python convert_to_cmyk.py
输入：curves 来自 checkpoints/curve_rgb_best.pth 或 deliverables/rgb_curve.npy
      图片来自 DATA_DIR 下的 *_input.jpg 或 input_samples/*.jpg
输出：rgb_curve_cmyk_output/ 下的 *_curved.jpg (CMYK JPEG)
"""
import os
import glob
import numpy as np
import torch
from PIL import Image, ImageCms

from config import Config
from models.curve_1d import Curve1D


def apply_curve_rgb(img_np, curves_np):
    """img_np: (H,W,3)∈[0,1]；curves_np: (3,33)。返回曲线后的 RGB。"""
    out = np.zeros_like(img_np)
    n = curves_np.shape[1]
    for c in range(3):
        x = img_np[:, :, c]
        idx_f = (n - 1) * x
        lo = np.clip(np.floor(idx_f).astype(int), 0, n - 2)
        hi = lo + 1
        frac = np.clip(idx_f - lo, 0, 1)
        out[:, :, c] = (1 - frac) * curves_np[c, lo] + frac * curves_np[c, hi]
    return np.clip(out, 0, 1)


def main():
    cfg = Config()
    cfg.setup_dirs()

    # ---- 加载曲线 ----
    ckpt = os.path.join(cfg.CHECKPOINT_DIR, "curve_rgb_best.pth")
    if os.path.exists(ckpt):
        curve = Curve1D(channels=3, n_bins=cfg.LUT_DIM)
        curve.load_state_dict(torch.load(ckpt, map_location="cpu"))
        curves_np = curve.get_curves().detach().cpu().numpy()
        print(f"[曲线] 加载自 {ckpt}")
    else:
        npy_path = os.path.join("deliverables", "rgb_curve.npy")
        if not os.path.exists(npy_path):
            print(f"⚠ 找不到 {ckpt} 或 {npy_path}，请先运行 train_rgb.py 和 export_curve.py")
            return
        curves_np = np.load(npy_path)
        print(f"[曲线] 加载自 {npy_path}")

    # ---- 找输入图片 ----
    imgs = sorted(glob.glob(os.path.join(cfg.DATA_DIR, "*_input.jpg")))
    if not imgs:
        os.makedirs("input_samples", exist_ok=True)
        print("⚠ 未在 DATA_DIR 找到 *_input.jpg。")
        print("   可把任意测试图片放到 input_samples/ 目录（支持 .jpg/.png）")
        imgs = sorted(glob.glob("input_samples/*.jpg") + glob.glob("input_samples/*.png"))
    if not imgs:
        return

    # ---- ICC：sRGB → PSOcoated_v3 CMYK ----
    cmyk_icc = os.path.join("utils", "PSOcoated_v3.icc")
    if not os.path.exists(cmyk_icc):
        print(f"⚠ 找不到 ICC: {cmyk_icc}")
        return
    prof_src = ImageCms.createProfile("sRGB")
    prof_dst = ImageCms.getOpenProfile(cmyk_icc)
    transform = ImageCms.buildTransformFromOpenProfiles(
        prof_src, prof_dst, "RGB", "CMYK",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)
    print(f"[ICC] {cmyk_icc} | relative + BPC（与调图师一致）")

    # ---- 批量转换 ----
    out_dir = "rgb_curve_cmyk_output"
    os.makedirs(out_dir, exist_ok=True)
    n_done = 0
    for path in imgs[:20]:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        cur = apply_curve_rgb(arr, curves_np)
        cur_pil = Image.fromarray((cur * 255).astype(np.uint8), "RGB")
        cmyk = ImageCms.applyTransform(cur_pil, transform)

        base = os.path.splitext(os.path.basename(path))[0]
        base = base.replace("_input", "")
        out_path = os.path.join(out_dir, f"{base}_curved.tif")  # TIFF 支持 CMYK
        cmyk.save(out_path)
        print(f"[转换] {os.path.basename(path)} → {out_path}")
        n_done += 1

    print(f"\n✅ 完成 {n_done} 张 → {out_dir}/")
    print("   格式为 CMYK TIFF，可直接用于印刷流程。")


if __name__ == "__main__":
    main()