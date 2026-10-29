# -*- coding: utf-8 -*-
"""
单张/批量推理与交付导出：
- 预测 3 条曲线并应用 -> 调色后 sRGB JPG
- 保存曲线参数 (.npy)
- 可选：用 PSOcoated_v3.icc 转回 CMYK 交付
"""
import os
import sys
import numpy as np
import torch
from PIL import Image, ImageCms
from torchvision.transforms.functional import to_tensor

from config import Config
from models.curve_predictor import CurvePredictor


def load_image(path, target_size=None):
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((target_size, target_size), Image.BILINEAR)
    tensor = to_tensor(img).unsqueeze(0)  # (1,3,H,W)
    return tensor, img


@torch.no_grad()
def process(model, input_path, output_dir, back_to_cmyk=False):
    cfg = Config()
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(input_path))[0]

    img_tensor, pil_in = load_image(input_path)
    out_tensor, curves = model.apply(img_tensor, resize_size=cfg.PRED_INPUT_SIZE)

    out_np = (out_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    out_pil = Image.fromarray(out_np)

    srgb_path = os.path.join(output_dir, name + "_converted.jpg")
    out_pil.save(srgb_path, quality=95)

    # 曲线参数
    curve_arr = curves.squeeze(0).cpu().numpy()  # (3, N)
    np.save(os.path.join(output_dir, name + "_curve.npy"), curve_arr)

    # 可选：转回 CMYK 交付
    if back_to_cmyk and os.path.exists(cfg.CMYK_ICC_PATH):
        cmyk_profile = ImageCms.getOpenProfile(cfg.CMYK_ICC_PATH)
        srgb_profile = ImageCms.createProfile("sRGB")
        to_cmyk = ImageCms.buildTransformFromOpenProfiles(
            srgb_profile, cmyk_profile, "RGB", "CMYK",
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
        )
        cmyk_pil = ImageCms.applyTransform(out_pil, to_cmyk)
        cmyk_pil.save(os.path.join(output_dir, name + "_delivery_cmyk.jpg"), quality=95)

    print(f"✅ {name}: sRGB={srgb_path}, curve_shape={curve_arr.shape}")
    return srgb_path


def main():
    if len(sys.argv) < 2:
        print("用法: python export_curve_pred.py <input.jpg|input_dir> [--cmyk]")
        return
    input_path = sys.argv[1]
    back_cmyk = "--cmyk" in sys.argv

    cfg = Config()
    device = torch.device("cpu")
    model = CurvePredictor(channels=3, n_bins=cfg.LUT_DIM, base=cfg.PRED_BASE).to(device)
    ckpt = os.path.join(cfg.CHECKPOINT_DIR, "curve_pred_best.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        print(f"⚠ 未找到 {ckpt}，使用恒等初始化（结果≈原图）")
    model.eval()

    output_dir = "output"
    if os.path.isdir(input_path):
        for f in sorted(os.listdir(input_path)):
            if f.endswith(("_input.jpg", "_input.jpeg")):
                process(model, os.path.join(input_path, f), output_dir, back_cmyk)
    else:
        process(model, input_path, output_dir, back_cmyk)


if __name__ == "__main__":
    main()