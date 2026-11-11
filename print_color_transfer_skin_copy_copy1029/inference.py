# -*- coding: utf-8 -*-
"""infer_rgb.py — 查看模型输出：对任意图片应用学习到的 RGB 曲线。

用法：
  python infer_rgb.py --image some_input.jpg
  python infer_rgb.py --dir ./input_samples --save_dir ./infer_output --dump_curve

说明：
  - 模型是逐像素 1D 曲线，可处理任意分辨率，无需缩放
  - target 若为 CMYK，会用 utils/PSOcoated_v3.icc 转 sRGB 再评估（与训练一致）
  - 输出 pred 的 RGB 预览图和 CMYK TIFF 印刷稿
"""
import os
import glob
import argparse
import numpy as np
import torch
from PIL import Image, ImageCms
from skimage.color import rgb2lab, deltaE_ciede2000

from config import Config
from models.curve_1d import Curve1D


CMS_INTENT = ImageCms.Intent.RELATIVE_COLORIMETRIC
CMS_FLAGS = ImageCms.Flags.BLACKPOINTCOMPENSATION


def build_icc_transforms(cmyk_icc_path):
    """返回 (cmyk_to_rgb, rgb_to_cmyk) 两个 transform，与训练/交付一致。"""
    srgb = ImageCms.createProfile("sRGB")
    cmyk = ImageCms.getOpenProfile(cmyk_icc_path)
    cmyk_to_rgb = ImageCms.buildTransformFromOpenProfiles(
        cmyk, srgb, "CMYK", "RGB",
        renderingIntent=CMS_INTENT, flags=CMS_FLAGS)
    rgb_to_cmyk = ImageCms.buildTransformFromOpenProfiles(
        srgb, cmyk, "RGB", "CMYK",
        renderingIntent=CMS_INTENT, flags=CMS_FLAGS)
    return cmyk_to_rgb, rgb_to_cmyk


def load_rgb(path, cmyk_to_rgb):
    """读取图片，若为 CMYK 则 ICC 转 sRGB，返回 float32 [0,1] (H,W,3)。"""
    img = Image.open(path)
    if img.mode == "CMYK":
        img = ImageCms.applyTransform(img, cmyk_to_rgb)
    else:
        img = img.convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


@torch.no_grad()
def apply_curve_tensor(img_np, curve):
    """用训练好的 Curve1D 对整张图做前向，返回 [0,1] float32 (H,W,3)。"""
    t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
    out = curve(t).clamp(0, 1)
    return out.squeeze(0).permute(1, 2, 0).numpy()


def evaluate_rgb(pred_np, ref_np):
    """计算 ΔE00 和 MAE（输入均为 float32 [0,1]）。"""
    p = (pred_np * 255).astype(np.uint8)
    r = (ref_np * 255).astype(np.uint8)
    de = deltaE_ciede2000(rgb2lab(p), rgb2lab(r)).mean()
    mae = np.abs(pred_np - ref_np).mean()
    return float(de), float(mae)


def process_one(img_path, tgt_path, curve, cmyk_to_rgb, rgb_to_cmyk, out_dir, dump_curve=False):
    name = os.path.splitext(os.path.basename(img_path))[0]
    name = name.replace("_input", "")
    print(f"\n=== {os.path.basename(img_path)} ===")

    # 读取 input 并应用曲线
    inp = load_rgb(img_path, cmyk_to_rgb)
    pred = apply_curve_tensor(inp, curve)

    # 保存 RGB 预览
    pred_pil = Image.fromarray((pred * 255).astype(np.uint8), "RGB")
    pred_rgb_path = os.path.join(out_dir, f"{name}_pred_rgb.jpg")
    pred_pil.save(pred_rgb_path, quality=95)
    print(f"[RGB预览] {pred_rgb_path}")

    # 保存 CMYK 印刷稿
    cmyk_pil = ImageCms.applyTransform(pred_pil, rgb_to_cmyk)
    cmyk_path = os.path.join(out_dir, f"{name}_pred_cmyk.tif")
    cmyk_pil.save(cmyk_path)
    print(f"[CMYK印刷] {cmyk_path}")

    # 如果有 target，计算指标并保存对比图
    if tgt_path and os.path.exists(tgt_path):
        tgt = load_rgb(tgt_path, cmyk_to_rgb)
        de, mae = evaluate_rgb(pred, tgt)
        print(f"[指标] ΔE00={de:.3f}  MAE={mae:.4f}")

        tgt_pil = Image.fromarray((tgt * 255).astype(np.uint8), "RGB")
        inp_pil = Image.fromarray((inp * 255).astype(np.uint8), "RGB")

        # 并排对比：input | pred | target
        w, h = inp_pil.size
        canvas = Image.new("RGB", (w * 3, h), "white")
        canvas.paste(inp_pil, (0, 0))
        canvas.paste(pred_pil, (w, 0))
        canvas.paste(tgt_pil, (w * 2, 0))
        cmp_path = os.path.join(out_dir, f"{name}_compare.jpg")
        canvas.save(cmp_path, quality=90)
        print(f"[对比图] {cmp_path}")
    else:
        print("[指标] 未找到 target，跳过评估")

    if dump_curve:
        curves = curve.get_curves().detach().cpu().numpy()  # (3,33)
        np.save(os.path.join(out_dir, "rgb_curve.npy"), curves)
        print("[曲线] 已保存 rgb_curve.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None, help="单张图片路径")
    ap.add_argument("--dir", type=str, default=None, help="批量处理目录（找 *_input.jpg）")
    ap.add_argument("--save_dir", type=str, default="infer_output")
    ap.add_argument("--dump_curve", action="store_true", help="同时导出曲线")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 加载配置和模型
    cfg = Config()
    cfg.setup_dirs()
    ckpt = os.path.join(cfg.CHECKPOINT_DIR, "curve_rgb_best.pth")
    if not os.path.exists(ckpt):
        print(f"❌ 找不到 {ckpt}")
        return
    curve = Curve1D(channels=3, n_bins=cfg.LUT_DIM)
    curve.load_state_dict(torch.load(ckpt, map_location="cpu"))
    curve.eval()
    print(f"[模型] 加载自 {ckpt}")

    # ICC 转换器
    cmyk_icc = os.path.join("utils", "PSOcoated_v3.icc")
    if not os.path.exists(cmyk_icc):
        print(f"❌ 找不到 ICC: {cmyk_icc}")
        return
    cmyk_to_rgb, rgb_to_cmyk = build_icc_transforms(cmyk_icc)

    # 收集输入图片
    if args.image:
        targets = [args.image]
    elif args.dir:
        targets = sorted(glob.glob(os.path.join(args.dir, "*_input.jpg")) +
                         glob.glob(os.path.join(args.dir, "*_input.JPG")) +
                         glob.glob(os.path.join(args.dir, "*_input.png")))
    else:
        targets = sorted(glob.glob(os.path.join(cfg.DATA_DIR, "*_input.jpg")))
    if not targets:
        print("❌ 没找到图片，请用 --image 或 --dir 指定")
        return

    print(f"[输入] 共 {len(targets)} 张图片\n")

    for img_path in targets[:20]:  # 限制最多 20 张，避免一次太多
        stem = os.path.splitext(os.path.basename(img_path))[0].replace("_input", "")
        tgt_path = None
        for suf in ("_target.jpg", "_target.JPG", "_target.tif", "_target.png"):
            cand = os.path.join(os.path.dirname(img_path), f"{stem}{suf}")
            if os.path.exists(cand):
                tgt_path = cand
                break
        process_one(img_path, tgt_path, curve, cmyk_to_rgb, rgb_to_cmyk,
                    args.save_dir, args.dump_curve)

    print(f"\n✅ 完成，输出目录：{args.save_dir}/")
    print("   *_pred_rgb.jpg    曲线后的 RGB 预览")
    print("   *_pred_cmyk.tif   曲线后的 CMYK 印刷稿")
    print("   *_compare.jpg     input | pred | target 对比")


if __name__ == "__main__":
    main()