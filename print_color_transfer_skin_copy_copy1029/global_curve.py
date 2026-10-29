# -*- coding: utf-8 -*-
"""
全局曲线拟合工具
用法: python global_curve.py --data <DATA_DIR> [--out <CURVE_PATH>] [--bins 33] [--img-size 256]
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image, ImageCms
from tqdm import tqdm

from dataset import discover_pairs, _preprocess_image


def build_cmyk_to_rgb_transform(cmyk_icc_path="utils/PSOcoated_v3.icc"):
    """构建 CMYK -> sRGB 的 ICC Transform（不是 Profile，是 Transform 对象）"""
    if not os.path.exists(cmyk_icc_path):
        raise FileNotFoundError(f"ICC 文件不存在: {cmyk_icc_path}")
    cmyk_profile = ImageCms.getOpenProfile(cmyk_icc_path)
    srgb_profile = ImageCms.createProfile("sRGB")
    transform = ImageCms.buildTransformFromOpenProfiles(
        cmyk_profile, srgb_profile, "CMYK", "RGB",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
    )
    return transform


def fit_global_curve(data_dir, bins=33, img_size=256, cmyk_to_rgb_transform=None):
    pairs = discover_pairs(data_dir)
    if not pairs:
        raise RuntimeError(f"在 {data_dir} 未找到任何 *_input.* / *_target.* 配对")
    print(f"[全局拟合] 共 {len(pairs)} 对，分箱 {bins}，img_size={img_size}，通道独立")

    # 直方图累计
    counts = np.zeros((3, bins), dtype=np.float64)
    sum_out = np.zeros((3, bins), dtype=np.float64)

    for inp_path, tgt_path in tqdm(pairs, desc="统计分箱"):
        try:
            inp = _preprocess_image(inp_path, is_target=False,
                                    img_size=img_size, cmyk_to_rgb_transform=None)
            tgt = _preprocess_image(tgt_path, is_target=True,
                                    img_size=img_size, cmyk_to_rgb_transform=cmyk_to_rgb_transform)
        except Exception as e:
            print(f"  ⚠ 跳过 {os.path.basename(inp_path)}: {e}")
            continue

        # 将输入图像值映射到 bin 索引
        inp_flat = inp.transpose(2, 0, 1).reshape(3, -1)   # (3, H*W)
        tgt_flat = tgt.transpose(2, 0, 1).reshape(3, -1)

        edges = np.linspace(0, 1, bins + 1)
        idx = np.searchsorted(edges, inp_flat, side="right") - 1
        idx = np.clip(idx, 0, bins - 1)

        for c in range(3):
            counts[c] += np.bincount(idx[c], minlength=bins)
            sum_out[c] += np.bincount(idx[c], weights=tgt_flat[c], minlength=bins)

    # 防止除零
    safe_counts = np.where(counts > 0, counts, 1.0)
    curve = sum_out / safe_counts  # (3, bins)

    # 无样本的 bin 线性插值
    for c in range(3):
        valid = counts[c] > 0
        x = np.arange(bins)
        if valid.sum() == 0:
            curve[c] = x / (bins - 1)
        elif valid.sum() < bins:
            curve[c] = np.interp(x, x[valid], curve[c, valid])

    # 高斯平滑（避免轻微噪声）
    from scipy.ndimage import gaussian_filter1d
    curve = gaussian_filter1d(curve, sigma=1.5, axis=-1)

    # 强制单调递增并归一化
    curve = np.maximum.accumulate(curve, axis=-1)
    curve = (curve - curve[:, :1]) / (curve[:, -1:] - curve[:, :1] + 1e-8)
    curve = np.clip(curve, 0, 1)

    return torch.from_numpy(curve.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="数据目录")
    ap.add_argument("--out", default="checkpoints/global_curve.pt")
    ap.add_argument("--bins", type=int, default=33)
    ap.add_argument("--img-size", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("[ICC] 加载 CMYK ICC 并构建 Transform ...")
    transform = build_cmyk_to_rgb_transform()

    curve = fit_global_curve(args.data, args.bins, args.img_size, transform)

    torch.save(curve, args.out)
    print(f"[全局拟合] 已保存 -> {args.out}  形状: {tuple(curve.shape)}")


if __name__ == "__main__":
    main()