#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键训练：生成mask → 训练整图曲线 → 训练肤色曲线
用法:
    python train_all.py --data /path/to/rgb_pairs
"""
import os, sys, argparse, subprocess

def run(cmd):
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--mask-cache", default="mask_cache")
    ap.add_argument("--skin-epochs", type=int, default=100)
    args = ap.parse_args()

    data = args.data or "data/rgb_pairs"  # 按需修改

    # 1. 生成 person/face mask
    run(f"python generate_masks.py --data-dir {data} --mask-cache {args.mask_cache}")

    # 2. 训练整图曲线 (CurvePredictor)
    run(f"python train_curve_pred.py")

    # 3. 训练肤色曲线（依赖整图曲线）
    run(f"python train_skin_curve.py --data {data} "
        f"--curve-model checkpoints/curve_pred_best.pth "
        f"--epochs {args.skin_epochs}")

    print("\n✅ 全部训练完成！")
    print("   checkpoints/curve_pred_best.pth  → 整图曲线")
    print("   checkpoints/skin_curve_best.pth  → 肤色曲线")

if __name__ == "__main__":
    main()