# -*- coding: utf-8 -*-
"""
训练 CurvePredictor v3（残差式 + 弱耦合）
核心：
- 曲线 = global + residual（初始化=global，基线稳）
- 损失：L1 + LAB + Hist + 残差幅度正则
- 监控：打印当前残差均值，确认模型在"学习偏离"而非躺平

【修补】
- 补齐 config 字段引用（PRED_BATCH_SIZE / *_WEIGHT / GRAD_CLIP / EARLY_STOP_PATIENCE）
- 新增 fit_global_curve_if_missing：若全局曲线缺失，先全量拟合一条再训练
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


def _fit_global_curve_if_missing(cfg, data_dir=None):
    """
    若全局曲线文件不存在，则用全部数据拟合一条，保存到 GLOBAL_CURVE_TENSOR_PATH。
    返回加载好的全局曲线 tensor (3, LUT_DIM)。
    """
    data_dir = data_dir or cfg.DATA_DIR
    if os.path.exists(cfg.GLOBAL_CURVE_TENSOR_PATH):
        obj = torch.load(cfg.GLOBAL_CURVE_TENSOR_PATH, map_location="cpu")
        t = obj if torch.is_tensor(obj) else obj.get("curves", obj)
        if torch.is_tensor(t) and t.shape[0] >= 3:
            if t.shape[0] == 4:
                t = t[:3]
            if t.shape[-1] == cfg.LUT_DIM:
                print(f"[全局曲线] 已存在 → {cfg.GLOBAL_CURVE_TENSOR_PATH}")
                return t.float()

    print("=" * 60)
    print("[全局曲线] 未找到有效全局曲线，先用全部数据拟合一条 ...")
    print("=" * 60)
    from global_curve import fit_global_curve, discover_pairs
    icc = "utils/PSOcoated_v3.icc"
    icc = icc if os.path.exists(icc) else None
    if icc:
        from icc_color import init_icc
        init_icc(icc)
    pairs = discover_pairs(data_dir)
    if len(pairs) == 0:
        raise RuntimeError(f"未在 {data_dir} 发现任何 *_input.* / *_target.* 配对，无法拟合全局曲线")
    curve = fit_global_curve(pairs, n_bins=cfg.LUT_DIM, img_size=256,
                             shared=False, icc_path=icc)
    os.makedirs(os.path.dirname(cfg.GLOBAL_CURVE_TENSOR_PATH), exist_ok=True)
    torch.save(curve, cfg.GLOBAL_CURVE_TENSOR_PATH)
    print(f"[全局曲线] 拟合完成并保存 → {cfg.GLOBAL_CURVE_TENSOR_PATH}  shape {tuple(curve.shape)}")
    return curve.float()


def load_global_curve(cfg):
    """尝试从多种可能路径加载全局曲线。"""
    for path in (cfg.GLOBAL_CURVE_TENSOR_PATH, cfg.GLOBAL_CURVE_PATH):
        if not os.path.exists(path):
            continue
        obj = torch.load(path, map_location="cpu")
        if torch.is_tensor(obj):
            t = obj
        elif isinstance(obj, dict) and "curves" in obj:
            t = obj["curves"]
        elif isinstance(obj, dict) and "delta" in obj:
            from models.curve_1d import curves_from_delta
            t = curves_from_delta(obj["delta"])
        else:
            continue
        if t.shape[0] == 4:
            t = t[:3]
        print(f"[全局曲线] 加载自 {path} -> {tuple(t.shape)}")
        return t.float()
    print("⚠ 未找到全局曲线，模型退化为纯预测")
    return None


def rgb_to_lab_torch(img):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device} | IMG_SIZE={cfg.IMG_SIZE} | PRED_INPUT_SIZE={cfg.PRED_INPUT_SIZE}")

    # ★ 若没有全局曲线，先全量拟合一条
    _fit_global_curve_if_missing(cfg)

    all_pairs, train_idx, val_idx = make_rgb_split(cfg.DATA_DIR, train_ratio=cfg.TRAIN_RATIO, seed=cfg.SEED)
    print(f"[配对] total={len(all_pairs)} train={len(train_idx)} val={len(val_idx)} overlap=0")

    train_ds = RGBPairDataset([all_pairs[i] for i in train_idx], cfg.CACHE_DIR, cfg.IMG_SIZE)
    val_ds = RGBPairDataset([all_pairs[i] for i in val_idx], cfg.CACHE_DIR, cfg.IMG_SIZE)
    kw = dict(num_workers=0, pin_memory=False)
    t_loader = DataLoader(train_ds, batch_size=cfg.PRED_BATCH_SIZE, shuffle=True, **kw)
    v_loader = DataLoader(val_ds, batch_size=cfg.PRED_BATCH_SIZE, shuffle=False, **kw)

    model = CurvePredictor(channels=3, n_bins=cfg.LUT_DIM, base=cfg.PRED_BASE,
                           residual_max=cfg.PRED_RESIDUAL_MAX, use_coupling=cfg.PRED_USE_COUPLING).to(device)
    gcurve = load_global_curve(cfg)
    if gcurve is not None:
        if gcurve.shape[-1] != cfg.LUT_DIM:
            raise ValueError(f"全局曲线 bin 数 {gcurve.shape[-1]} != {cfg.LUT_DIM}")
        model.set_global_curve(gcurve)
    else:
        print("⚠ 无全局先验，将用恒等曲线初始化")

    l1 = nn.L1Loss()
    hist = SoftHistogramLoss(cfg.HIST_BINS, cfg.HIST_DARK_WEIGHT, cfg.HIST_MID_WEIGHT, cfg.HIST_BRIGHT_WEIGHT)

    optimizer = optim.Adam(model.parameters(), lr=cfg.PRED_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.PRED_EPOCHS)

    best_de, best_ep, no_imp = float("inf"), 0, 0
    print(f"\n=== CurvePredictor v3 训练 (L1 + LAB + Hist + 残差正则) ===")

    for ep in range(cfg.PRED_EPOCHS):
        model.train()
        pbar = tqdm(t_loader, desc=f"Pred {ep+1}/{cfg.PRED_EPOCHS}")
        for batch in pbar:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)

            inp_small = nn.functional.interpolate(
                inp, size=(cfg.PRED_INPUT_SIZE, cfg.PRED_INPUT_SIZE), mode="bilinear", align_corners=False)
            tgt_small = nn.functional.interpolate(
                tgt, size=(cfg.PRED_INPUT_SIZE, cfg.PRED_INPUT_SIZE), mode="bilinear", align_corners=False)

            curves = model(inp_small)                 # (B,3,N)
            pred = apply_curves(inp_small, curves)    # (B,3,S,S)

            lab_pred = rgb_to_lab_torch(pred)
            lab_tgt = rgb_to_lab_torch(tgt_small)
            lab_loss = nn.functional.l1_loss(lab_pred, lab_tgt)

            # 残差幅度正则（软约束，避免偏离全局曲线太远）
            res_reg = (curves - model.global_curve.unsqueeze(0)).abs().mean()

            loss = (cfg.L1_WEIGHT * l1(pred, tgt_small)
                    + cfg.PRED_LAB_WEIGHT * lab_loss
                    + cfg.PRED_HIST_WEIGHT * hist(pred, tgt_small)
                    + 0.5 * res_reg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             mae=f"{(pred - tgt_small).abs().mean().item():.4f}",
                             res=f"{res_reg.item():.4f}")

        scheduler.step()
        m = evaluate(model, v_loader, device, cfg.PRED_INPUT_SIZE, max_samples=0)
        print(f"  Pred {ep+1}: ΔE00={m['de']:.3f}  MAE={m['mae']:.5f}")

        if m["de"] < best_de:
            best_de, best_ep, no_imp = m["de"], ep, 0
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "curve_pred_best.pth"))
            torch.save(model.global_curve, os.path.join(cfg.CHECKPOINT_DIR, "global_curve.pt"))
        else:
            no_imp += 1
        if no_imp >= cfg.EARLY_STOP_PATIENCE:
            print(f"  Early stop @ {ep+1}, best ΔE00={best_de:.3f} (ep {best_ep+1})")
            break

    print(f"\n✅ 完成  最佳验证 ΔE00 = {best_de:.3f} (epoch {best_ep+1})")
    if best_de < 2.0:
        print("   → ΔE00 < 2.0，达标 🎉")
    elif best_de < 3.0:
        print("   → ΔE00 < 3.0，优于全局基线")
    else:
        print("   → 若仍卡住，可：1) 增大 residual_max 到 0.12；2) 确认 curves_rgb.pt 是 ICC 口径最优；")
        print("     3) 叠加极轻量残差 U-Net（HybridColorModel）做最后 0.2~0.4 的突破。")
    torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "curve_pred_final.pth"))


if __name__ == "__main__":
    train()
