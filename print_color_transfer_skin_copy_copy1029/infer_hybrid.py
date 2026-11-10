# -*- coding: utf-8 -*-
"""infer_hybrid.py — 查看混合模型(Curve1D + 残差U-Net)输出，支持三路对比。

用法:
  python infer_hybrid.py --image xxx_input.jpg [--use_unet]
  python infer_hybrid.py --dir ./samples --save_dir infer_output --use_unet

对比图布局(4列): input | curve_only | curve+residual(UNet) | target
默认 use_unet=True; 加 --no_unet 可只看曲线基线作对照。
"""
import os
import glob
import argparse
import numpy as np
import torch
from PIL import Image, ImageCms
from skimage.color import rgb2lab, deltaE_ciede2000

from config import Config
from models.residual_unet import HybridColorModel


CMS_INTENT = ImageCms.Intent.RELATIVE_COLORIMETRIC
CMS_FLAGS = ImageCms.Flags.BLACKPOINTCOMPENSATION


def build_icc_transforms(cmyk_icc_path):
    srgb = ImageCms.createProfile("sRGB")
    cmyk = ImageCms.getOpenProfile(cmyk_icc_path)
    cmyk_to_rgb = ImageCms.buildTransformFromOpenProfiles(cmyk, srgb, "CMYK", "RGB",
                                                          renderingIntent=CMS_INTENT, flags=CMS_FLAGS)
    rgb_to_cmyk = ImageCms.buildTransformFromOpenProfiles(srgb, cmyk, "RGB", "CMYK",
                                                          renderingIntent=CMS_INTENT, flags=CMS_FLAGS)
    return cmyk_to_rgb, rgb_to_cmyk


def load_rgb(path, cmyk_to_rgb):
    img = Image.open(path)
    if img.mode == "CMYK":
        img = ImageCms.applyTransform(img, cmyk_to_rgb)
    else:
        img = img.convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


@torch.no_grad()
def apply_model(img_np, model, use_unet):
    t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
    out = model(t, use_unet=use_unet).clamp(0, 1)
    return out.squeeze(0).permute(1, 2, 0).numpy()


def evaluate_rgb(pred_np, ref_np):
    p = (pred_np * 255).astype(np.uint8)
    r = (ref_np * 255).astype(np.uint8)
    de = deltaE_ciede2000(rgb2lab(p), rgb2lab(r)).mean()
    return float(de), float(np.abs(pred_np - ref_np).mean())


def process_one(img_path, tgt_path, model, cmyk_to_rgb, rgb_to_cmyk, out_dir, use_unet):
    name = os.path.splitext(os.path.basename(img_path))[0].replace("_input", "")
    print(f"\n=== {os.path.basename(img_path)} ===")

    inp = load_rgb(img_path, cmyk_to_rgb)
    curve_out = apply_model(inp, model, use_unet=False)   # 仅曲线基线
    pred = apply_model(inp, model, use_unet=use_unet)       # 曲线 + 残差

    # 保存 RGB 预览
    Image.fromarray((pred * 255).astype(np.uint8), "RGB").save(
        os.path.join(out_dir, f"{name}_pred_rgb.jpg"), quality=95)
    Image.fromarray((curve_out * 255).astype(np.uint8), "RGB").save(
        os.path.join(out_dir, f"{name}_curve_only.jpg"), quality=95)
    # CMYK 印刷稿
    ImageCms.applyTransform(Image.fromarray((pred * 255).astype(np.uint8), "RGB"), rgb_to_cmyk).save(
        os.path.join(out_dir, f"{name}_pred_cmyk.tif"))

    if tgt_path and os.path.exists(tgt_path):
        tgt = load_rgb(tgt_path, cmyk_to_rgb)
        de_curve, mae_c = evaluate_rgb(curve_out, tgt)
        de_full, mae_f = evaluate_rgb(pred, tgt)
        print(f"[仅曲线] ΔE00={de_curve:.3f} MAE={mae_c:.4f}")
        print(f"[+残差UNet] ΔE00={de_full:.3f} MAE={mae_f:.4f}")

        # 4列对比图: input | curve_only | curve+residual | target
        inp_pil = Image.fromarray((inp * 255).astype(np.uint8), "RGB")
        tgt_pil = Image.fromarray((tgt * 255).astype(np.uint8), "RGB")
        pred_pil = Image.fromarray((pred * 255).astype(np.uint8), "RGB")
        curve_pil = Image.fromarray((curve_out * 255).astype(np.uint8), "RGB")
        w, h = inp_pil.size
        canvas = Image.new("RGB", (w * 4, h), "white")
        canvas.paste(inp_pil, (0, 0))
        canvas.paste(curve_pil, (w, 0))
        canvas.paste(pred_pil, (w * 2, 0))
        canvas.paste(tgt_pil, (w * 3, 0))
        # 顶部标注
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        labels = ["input", f"curve only\nΔE={de_curve:.2f}", f"+residual\nΔE={de_full:.2f}", "target"]
        for i, lab in enumerate(labels):
            draw.text((w * i + 10, 10), lab, fill="red")
        canvas.save(os.path.join(out_dir, f"{name}_compare.jpg"), quality=90)
        print(f"[对比图] {name}_compare.jpg (4列: input|curve|curve+residual|target)")
    else:
        print("[指标] 未找到 target，跳过评估")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--dir", type=str, default=None)
    ap.add_argument("--save_dir", type=str, default="infer_output")
    ap.add_argument("--no_unet", action="store_true", help="只看曲线基线")
    args = ap.parse_args()
    use_unet = not args.no_unet
    os.makedirs(args.save_dir, exist_ok=True)

    cfg = Config(); cfg.setup_dirs()
    ckpt = os.path.join(cfg.CHECKPOINT_DIR, "hybrid_best.pth")
    if not os.path.exists(ckpt):
        print(f"❌ 找不到 {ckpt}，请先跑 train_hybrid.py")
        return
    model = HybridColorModel(n_bins=cfg.LUT_DIM, unet_base=16).to("cpu")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    print(f"[模型] {ckpt} (use_unet={use_unet})")

    cmyk_icc = os.path.join("utils", "PSOcoated_v3.icc")
    if not os.path.exists(cmyk_icc):
        print(f"❌ 找不到 ICC: {cmyk_icc}"); return
    cmyk_to_rgb, rgb_to_cmyk = build_icc_transforms(cmyk_icc)

    if args.image:
        targets = [args.image]
    elif args.dir:
        targets = sorted(glob.glob(os.path.join(args.dir, "*_input.*")))
    else:
        targets = sorted(glob.glob(os.path.join(cfg.DATA_DIR, "*_input.jpg")))
    if not targets:
        print("❌ 没找到图片"); return

    for img_path in targets[:20]:
        stem = os.path.splitext(os.path.basename(img_path))[0].replace("_input", "")
        tgt_path = None
        for suf in ("_target.jpg", "_target.JPG", "_target.tif", "_target.png"):
            cand = os.path.join(os.path.dirname(img_path), f"{stem}{suf}")
            if os.path.exists(cand):
                tgt_path = cand; break
        process_one(img_path, tgt_path, model, cmyk_to_rgb, rgb_to_cmyk, args.save_dir, use_unet)

    print(f"\n✅ 完成 → {args.save_dir}/")


if __name__ == "__main__":
    main()
