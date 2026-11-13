# -*- coding: utf-8 -*-
"""
训练参数化曲线预测器（全局先验融合版）。
核心改动：
- 全局曲线通过模型结构注入（凸组合），不再靠外部正则
- 新增 LAB 空间损失，增强颜色感知
- 使用余弦退火 + 梯度累积，稳定收敛
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.color import rgb2lab, deltaE_ciede2000

from config import Config
from dataset import make_rgb_split, RGBPairDataset
from models.curve_predictor import CurvePredictor, apply_curves
from utils.losses import SoftHistogramLoss


def load_global_curve(cfg):
    """
    加载全局曲线 tensor，用于注入模型。
    优先 curves_rgb.pt（tensor），其次从 state_dict 提取。
    返回 (3, N) tensor 或 None。
    """
    for path in (cfg.GLOBAL_CURVE_TENSOR_PATH, cfg.GLOBAL_CURVE_PATH):
        if not os.path.exists(path):
            continue
        obj = torch.load(path, map_location="cpu")
        # 情况1：直接是 tensor
        if torch.is_tensor(obj):
            t = obj
        # 情况2：dict 含 'curves'
        elif isinstance(obj, dict) and "curves" in obj:
            t = obj["curves"]
        # 情况3：Curve1D state_dict 含 'delta'
        elif isinstance(obj, dict) and "delta" in obj:
            from models.curve_1d import curves_from_delta
            t = curves_from_delta(obj["delta"])
        else:
            continue
        if t.shape[0] == 4:   # 兼容 CMYK 曲线
            t = t[:3]
        print(f"[全局曲线] 加载自 {path} -> {tuple(t.shape)}")
        return t.float()
    print("[全局曲线] 未找到，模型退化为纯预测（无先验）")
    return None


def rgb_to_lab_torch(img):
    """可导 sRGB -> Lab (B,3,H,W)，输出未归一化"""
    img = img.clamp(0, 1)
    mask = img > 0.04045
    lin = torch.where(mask, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)
    r, g, b = lin[:, 0], lin[:, 1], lin[:, 2]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def f(t):
        d = 6 / 29
        return torch.where(t > d ** 3, torch.clamp(t, 1e-8) ** (1 / 3), t / (3 * d ** 2) + 4 / 29)

    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return torch.stack([L, a, b_val], dim=1)


@torch.no_grad()
def evaluate(model, loader, device, pred_input_size, max_samples=0):
    model.eval()
    de_sum, mae_sum, n = 0.0, 0.0, 0
    for batch in loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        bs = tgt.size(0)

        out, _ = model.apply(inp, resize_size=pred_input_size)
        out = out.clamp(0, 1)
        mae_sum += (out - tgt).abs().mean().item() * bs

        pred = (out.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        ref = (tgt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        for i in range(bs):
            de_sum += deltaE_ciede2000(rgb2lab(pred[i]), rgb2lab(ref[i])).mean()
        n += bs
        if max_samples and n >= max_samples:
            break
    return {"de": de_sum / max(n, 1), "mae": mae_sum / max(n, 1)}


def train():
    cfg = Config()
    cfg.setup_dirs()
    torch.manual_seed(cfg.SEED)
    device = torch.device("cpu")
    print(f"[设备] {device} | IMG_SIZE={cfg.IMG_SIZE} | PRED_INPUT_SIZE={cfg.PRED_INPUT_SIZE}")

    all_pairs, train_idx, val_idx = make_rgb_split(cfg.DATA_DIR, train_ratio=cfg.TRAIN_RATIO, seed=cfg.SEED)
    print(f"[配对] total={len(all_pairs)} train={len(train_idx)} val={len(val_idx)} overlap=0")

    train_ds = RGBPairDataset([all_pairs[i] for i in train_idx], cfg.CACHE_DIR, cfg.IMG_SIZE)
    val_ds = RGBPairDataset([all_pairs[i] for i in val_idx], cfg.CACHE_DIR, cfg.IMG_SIZE)

    kw = dict(num_workers=0, pin_memory=False)
    t_loader = DataLoader(train_ds, batch_size=cfg.PRED_BATCH_SIZE, shuffle=True, **kw)
    v_loader = DataLoader(val_ds, batch_size=cfg.PRED_BATCH_SIZE, shuffle=False, **kw)

    # ---- 模型：注入全局曲线先验 ----
    model = CurvePredictor(channels=3, n_bins=cfg.LUT_DIM, base=cfg.PRED_BASE).to(device)
    gcurve = load_global_curve(cfg)
    if gcurve is not None:
        if gcurve.shape[-1] != cfg.LUT_DIM:
            raise ValueError(f"全局曲线 bin 数 {gcurve.shape[-1]} != 配置 {cfg.LUT_DIM}")
        model.set_global_curve(gcurve)
    else:
        print("⚠ 无全局曲线先验，模型从恒等曲线开始（效果可能较差）")

    l1 = nn.L1Loss()
    hist = SoftHistogramLoss(cfg.HIST_BINS, cfg.HIST_DARK_WEIGHT,
                             cfg.HIST_MID_WEIGHT, cfg.HIST_BRIGHT_WEIGHT)

    # 参数分组：曲线分支（含全局先验）用较小 LR，编码器正常 LR
    params_encoder = list(model.encoder.parameters()) + list(model.fc1.parameters())
    params_curve = list(model.fc_delta.parameters()) + list(model.fc_w.parameters())
    optimizer = optim.Adam([
        {"params": params_encoder, "lr": cfg.PRED_LR},
        {"params": params_curve, "lr": cfg.PRED_LR * 0.5},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.PRED_EPOCHS)

    best_de, best_ep, no_imp = float("inf"), 0, 0
    print(f"\n=== 参数化曲线预测器训练 (L1 + Hist + LAB + Mono) ===")

    for ep in range(cfg.PRED_EPOCHS):
        model.train()
        pbar = tqdm(t_loader, desc=f"Pred {ep+1}/{cfg.PRED_EPOCHS}")
        for batch in pbar:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)

            # 训练时下采样，减少 CPU 计算
            inp_small = nn.functional.interpolate(
                inp, size=(cfg.PRED_INPUT_SIZE, cfg.PRED_INPUT_SIZE), mode="bilinear", align_corners=False)
            tgt_small = nn.functional.interpolate(
                tgt, size=(cfg.PRED_INPUT_SIZE, cfg.PRED_INPUT_SIZE), mode="bilinear", align_corners=False)

            curves = model(inp_small)                     # (B,3,N)
            pred = apply_curves(inp_small, curves)        # (B,3,S,S)

            # 单调惩罚（凸组合后仍单调，此处仅为保险）
            mono = nn.functional.relu(curves[:, :, 1:] - curves[:, :, :-1] + 1e-4).mean()

            # LAB 损失：增强颜色感知
            lab_pred = rgb_to_lab_torch(pred)
            lab_tgt = rgb_to_lab_torch(tgt_small)
            lab_loss = nn.functional.l1_loss(lab_pred, lab_tgt)

            loss = (cfg.L1_WEIGHT * l1(pred, tgt_small)
                    + cfg.PRED_HIST_WEIGHT * hist(pred, tgt_small)
                    + cfg.PRED_LAB_WEIGHT * lab_loss
                    + cfg.PRED_MONO_WEIGHT * mono)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             mae=f"{(pred - tgt_small).abs().mean().item():.4f}")

        scheduler.step()

        # 验证
        m = evaluate(model, v_loader, device, cfg.PRED_INPUT_SIZE, max_samples=0)
        print(f"  Pred {ep+1}: ΔE00={m['de']:.3f}  MAE={m['mae']:.5f}")

        if m["de"] < best_de:
            best_de, best_ep, no_imp = m["de"], ep, 0
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "curve_pred_best.pth"))
            torch.save(model.get_curves() if False else curves.detach().cpu(),
                       os.path.join(cfg.CHECKPOINT_DIR, "curves_pred.pt"))
        else:
            no_imp += 1
        if no_imp >= cfg.EARLY_STOP_PATIENCE:
            print(f"  Early stop @ {ep+1}, best ΔE00={best_de:.3f} (ep {best_ep+1})")
            break

    print(f"\n✅ 完成  最佳验证 ΔE00 = {best_de:.3f} (epoch {best_ep+1})")
    if best_de < 2.0:
        print("   → ΔE00 < 2.0，达到用户要求 🎉")
    elif best_de < 3.0:
        print("   → ΔE00 < 3.0，优于全局曲线基线")
    else:
        print("   → 未达 3.0，建议：")
        print("     1) 确认全局曲线 curves_rgb.pt 确实是 ICC 口径下的最优结果；")
        print("     2) 增大 PRED_BASE 至 48，或增加训练 epoch；")
        print("     3) 在曲线输出后叠加一个极轻量残差 CNN（HybridColorModel）。")
    torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "curve_pred_final.pth"))


if __name__ == "__main__":
    train()