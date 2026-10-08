# -*- coding: utf-8 -*-
"""评估参数化曲线预测器：ΔE00 / MAE + 可视化对比图"""
import os
import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from torch.utils.data import DataLoader

from config import Config
from dataset import make_rgb_split, RGBPairDataset
from models.curve_predictor import CurvePredictor


@torch.no_grad()
def evaluate(model, loader, device, save_dir="runs/eval_pred", max_save=5):
    model.eval()
    de_sum, mae_sum, n = 0.0, 0.0, 0
    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, batch in enumerate(loader):
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        bs = tgt.size(0)

        out, curves = model.apply(inp, resize_size=Config.PRED_INPUT_SIZE)
        out = out.clamp(0, 1)
        mae_sum += (out - tgt).abs().mean().item() * bs

        pred_np = (out.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        ref_np = (tgt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        inp_np = (inp.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

        for i in range(bs):
            de_sum += deltaE_ciede2000(rgb2lab(pred_np[i]), rgb2lab(ref_np[i])).mean()
            if batch_idx < max_save:
                idx = batch_idx * bs + i
                Image.fromarray(inp_np[i]).save(os.path.join(save_dir, f"{idx:04d}_inp.png"))
                Image.fromarray(pred_np[i]).save(os.path.join(save_dir, f"{idx:04d}_pred.png"))
                Image.fromarray(ref_np[i]).save(os.path.join(save_dir, f"{idx:04d}_ref.png"))
                np.save(os.path.join(save_dir, f"{idx:04d}_curves.npy"), curves[i].cpu().numpy())
        n += bs

    return {"de": de_sum / max(n, 1), "mae": mae_sum / max(n, 1)}


def main():
    cfg = Config()
    device = torch.device("cpu")

    pairs, _, val_idx = make_rgb_split(cfg.DATA_DIR, train_ratio=cfg.TRAIN_RATIO, seed=cfg.SEED)
    val_ds = RGBPairDataset([pairs[i] for i in val_idx], cfg.CACHE_DIR, cfg.IMG_SIZE)
    v_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    model = CurvePredictor(channels=3, n_bins=cfg.LUT_DIM, base=cfg.PRED_BASE).to(device)
    ckpt = os.path.join(cfg.CHECKPOINT_DIR, "curve_pred_best.pth")
    if not os.path.exists(ckpt):
        print(f"❌ 找不到 {ckpt}，请先运行 train_curve_pred.py")
        return

    # 加载全局曲线（与训练一致）
    from train_curve_pred import load_global_curve
    gcurve = load_global_curve(cfg)
    if gcurve is not None:
        model.set_global_curve(gcurve)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"已加载: {ckpt}")

    res = evaluate(model, v_loader, device)
    print(f"\n验证集 (n={len(val_idx)}):")
    print(f"  ΔE00 = {res['de']:.3f}")
    print(f"  MAE  = {res['mae']:.5f}")
    print("对比图 -> runs/eval_pred/")


if __name__ == "__main__":
    main()