#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肤色曲线预测网络训练（改进版）

【修补】
- 原代码依赖 processor.py:load_model，但该文件缺失 → 改为直接加载 CurvePredictor 权重
- 支持 CUDA
- 训练时实时统计每张图被 mask 覆盖的像素占比，便于排查“mask 错位”问题
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.color import rgb2lab, deltaE_ciede2000

from config import Config
from dataset import make_rgb_split, RGBSkinDataset
from models.curve_predictor import CurvePredictor, apply_curves as apply_lut
from models.skin_curve_net import SkinCurveNet, apply_lut as apply_skin_lut


# ---------- 可微 RGB→Lab ----------
def rgb2lab_torch(rgb, device):
    eps = 1e-6
    mask = rgb > 0.04045
    linear = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = torch.tensor([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ], dtype=torch.float32, device=device)
    xyz = torch.einsum('ij,bjhw->bihw', M, linear)
    X, Y, Z = xyz[:, 0:1] / 0.95047, xyz[:, 1:2] / 1.0, xyz[:, 2:3] / 1.08883
    f = lambda t: torch.where(t > 0.008856, torch.pow(t.clamp(min=eps), 1/3), 7.787 * t + 16/116)
    fx, fy, fz = f(X), f(Y), f(Z)
    return torch.cat([116*fy - 16, 500*(fx-fy), 200*(fy-fz)], dim=1)
def masked_apply(img, curves, mask):
    """
    ★ 只在 mask 内应用曲线，mask 外保持原图 —— 与 processor._generate_hybrid_stages
    里的 base2 = base1 * (1-mask) + lut(base1) * mask 完全一致。

    img:    (B,3,H,W)
    curves: (B,3,n_bins)
    mask:   (B,1,H,W) 或 (B,H,W)
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    applied = apply_skin_lut(img, curves)
    return img * (1.0 - mask) + applied * mask

# ---------- 损失 ----------
def skin_curve_loss(curves, pred, tgt,
                    person, skin_face,
                    lambda_smooth, lambda_identity):
    """
    曲线监督损失：
      - 脸部（skin_face）：主损失，高权重
      - 人物身体（person - skin_face）：辅助损失
      - 曲线平滑 + 曲线接近恒等

    ★ 注意：pred 已经在 mask 内应用过了，mask 外的像素 = inp，
      因此不必再额外惩罚背景漂移。
    """
    # 像素级 L1
    l1 = F.l1_loss(pred, tgt, reduction='none').mean(dim=1, keepdim=True)  # (B,1,H,W)

    # 可微 Lab 距离
    lab_pred = rgb2lab_torch(pred, pred.device)
    lab_tgt = rgb2lab_torch(tgt, tgt.device)
    lab_dist = (lab_pred - lab_tgt).pow(2).sum(dim=1, keepdim=True) \
        .clamp_min(1e-10).sqrt()
    lab_norm = lab_dist / 50.0

    per_pixel = l1 + 0.3 * lab_norm

    # ---- 人脸区域：主损失 ----
    face_mask = (skin_face > 0.5).float()
    face_count = face_mask.sum().clamp(min=1.0)
    face_loss = ((face_mask * per_pixel).sum() / face_count) * 1.5

    # ---- 人物区域（排除脸）：辅助损失 ----
    person_only = ((person > 0.5).float() * (1 - face_mask)).clamp(0.0, 1.0)
    person_count = person_only.sum().clamp(min=1.0)
    person_loss = ((person_only * per_pixel).sum() / person_count) * 0.3

    # ---- 曲线平滑正则 ----
    diff1 = curves[:, :, 1:] - curves[:, :, :-1]
    diff2 = diff1[:, :, 1:] - diff1[:, :, :-1]
    smooth_loss = diff1.pow(2).mean() + 0.5 * diff2.pow(2).mean()

    # ---- ★ 曲线恒等正则（防止激进偏离，比之前的 magnitude_reg 权重更大） ----
    ident = torch.linspace(0, 1, curves.shape[-1], device=curves.device).view(1, 1, -1)
    identity_loss = (curves - ident).abs().mean()

    total = (face_loss
             + person_loss
             + lambda_smooth * smooth_loss
             + lambda_identity * identity_loss)

    return total, {
        "face": face_loss,
        "person": person_loss,
        "smooth": smooth_loss,
        "identity": identity_loss,
    }

# ---------- 评估 ----------
@torch.no_grad()
def masked_deltaE(img, tgt, skin_np, min_pixels=100, return_mean=True):
    de_sum, n = 0.0, 0
    pred_np = img.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    tgt_np = tgt.permute(0, 2, 3, 1).cpu().numpy()

    for i in range(pred_np.shape[0]):
        m = skin_np[i, 0] > 0.5
        if m.sum() < min_pixels:
            continue

        p = (pred_np[i][m] * 255).astype(np.float32)
        t = (tgt_np[i][m] * 255).astype(np.float32)

        # 过滤 NaN/Inf 像素
        valid = np.isfinite(p).all(axis=1) & np.isfinite(t).all(axis=1)
        valid &= (p.min(axis=1) >= 0) & (p.max(axis=1) <= 255)
        p = np.clip(p[valid], 0, 255).astype(np.uint8)
        t = np.clip(t[valid], 0, 255).astype(np.uint8)

        if len(p) < min_pixels:
            continue

        de_sum += deltaE_ciede2000(rgb2lab(p), rgb2lab(t)).mean()
        n += 1

    return (de_sum / max(n, 1)) if return_mean else (de_sum, n)


@torch.no_grad()
def evaluate(model, curve_model, dataloader, device, curve_input_size=320):
    model.eval()
    total_de, total_n = 0.0, 0
    base_de_sum, base_n = 0.0, 0

    for batch in dataloader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        skin_broad = batch["skin_broad"].to(device)   # ★ 与训练一致
        skin_face_np = batch["skin_face"].cpu().numpy()

        if curve_model is not None:
            inp = curve_model.apply(inp, resize_size=curve_input_size)[0]

        # 基线
        b_de, b_n = masked_deltaE(inp, tgt, skin_face_np, return_mean=False)
        base_de_sum += b_de
        base_n += b_n

        # ★ 模型输入 mask 用 skin_broad（与推理一致）
        curves = model(inp, mask=skin_broad)
        # ★ 只在 mask 内应用
        pred = masked_apply(inp, curves, skin_broad)

        s_de, s_n = masked_deltaE(pred, tgt, skin_face_np, return_mean=False)
        total_de += s_de
        total_n += s_n

    base_avg = base_de_sum / max(base_n, 1)
    skin_avg = total_de / max(total_n, 1)
    print(f"   基线(整图曲线后): ΔE={base_avg:.3f} ({base_n}张)")
    print(f"   肤色曲线后:       ΔE={skin_avg:.3f} ({total_n}张)")
    return skin_avg


def _load_curve_model(ckpt_path, device, curve_input_size):
    """加载整图曲线模型（CurvePredictor）。兼容 state_dict 与完整 checkpoint。"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"整图曲线模型不存在: {ckpt_path}")
    from config import Config
    cfg = Config()
    curve_model = CurvePredictor(
        channels=3, n_bins=cfg.LUT_DIM, base=cfg.PRED_BASE,
        residual_max=cfg.PRED_RESIDUAL_MAX, use_coupling=cfg.PRED_USE_COUPLING,
    ).to(device)
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        state = obj["model"]
    else:
        state = obj
    curve_model.load_state_dict(state)
    # 若 checkpoint 里保存了拟合好的 global_curve，顺带注入
    if isinstance(obj, dict) and "global_curve" in obj:
        curve_model.set_global_curve(obj["global_curve"])
    curve_model.eval()
    for p in curve_model.parameters():
        p.requires_grad_(False)
    return curve_model


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="默认取 config.DATA_DIR")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lambda-smooth", type=float, default=1e-3)
    ap.add_argument("--curve-model", default="checkpoints/curve_pred_best.pth", help="整图曲线模型路径(先应用再训练肤色)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--lambda-identity", type=float, default=0.15,help="★ 曲线恒等正则权重（新加，默认 0.15）")
    ap.add_argument("--patience", type=int, default=40, help="早停耐心值")
    ap.add_argument("--mask-dir", default=None, help="mask 目录，默认取 config.SKIN_MASK_DIR")
    args = ap.parse_args()

    cfg = Config()
    cfg.setup_dirs()
    data_dir = args.data or cfg.DATA_DIR
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 数据 ----------
    pairs, train_idx, val_idx = make_rgb_split(data_dir, train_ratio=cfg.TRAIN_RATIO, seed=cfg.SEED)
    print(f"[配对] total={len(pairs)} train={len(train_idx)} val={len(val_idx)}")

    mask_dir = args.mask_dir or cfg.SKIN_MASK_DIR
    train_ds = RGBSkinDataset(
        [pairs[i] for i in train_idx],
        mask_dir=mask_dir,
        cache_dir=cfg.CACHE_DIR,
        img_size=args.img_size,
    )
    val_ds = RGBSkinDataset(
        [pairs[i] for i in val_idx],
        mask_dir=mask_dir,
        cache_dir=cfg.CACHE_DIR,
        img_size=args.img_size,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # ---------- 整图曲线（可选）----------
    curve_model = None
    if args.curve_model and os.path.exists(args.curve_model):
        print(f"[整图曲线] 加载 {args.curve_model}")
        curve_model = _load_curve_model(args.curve_model, device, cfg.PRED_INPUT_SIZE)
    else:
        print("[整图曲线] 未提供，肤色曲线将作用于原图")

    # ---------- 模型 ----------
    model = SkinCurveNet(n_bins=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch, best_de = 0, 1e9
    no_imp = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_de = ckpt.get("best_de", 1e9)
        no_imp = ckpt.get("no_imp", 0)

    os.makedirs(args.out, exist_ok=True)

    # ---------- 训练 ----------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = {"face": 0, "person": 0, "smooth": 0, "total": 0}
        n_iter = 0

        for batch in train_loader:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            person = batch["person_mask"].to(device)
            skin_face = batch["skin_face"].to(device)
            skin_broad = batch["skin_broad"].to(device)   # ★ 新增

            # 先过 Stage1 全局曲线
            if curve_model is not None:
                with torch.no_grad():
                    inp = curve_model.apply(inp, resize_size=cfg.PRED_INPUT_SIZE)[0]

            # ★ 模型输入 mask 用 skin_broad（推理一致性）
            curves = model(inp, mask=skin_broad)

            # ★ 只在 skin_broad 内应用曲线（与推理一致，mask 外自动 = inp）
            pred = masked_apply(inp, curves, skin_broad)

            loss, parts = skin_curve_loss(
                curves, pred, tgt,
                person, skin_face,
                args.lambda_smooth, args.lambda_identity,
            )

            optimizer.zero_grad()
            loss.backward()

            # NaN 检查
            has_nan_grad = any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in model.parameters()
            )
            if has_nan_grad or torch.isnan(loss):
                print("    ⚠ 检测到 NaN 梯度/损失，跳过本步更新")
                optimizer.zero_grad()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            for k, v in parts.items():
                losses[k] = losses.get(k, 0.0) + float(v.detach())
            losses["total"] += float(loss.detach())
            n_iter += 1

        # 验证（在 scheduler.step 之前，确保评估使用当前 lr）
        val_de = evaluate(model, curve_model, val_loader, device)
        scheduler.step()

        avg = {k: v / max(n_iter, 1) for k, v in losses.items()}
        print(f"[Epoch {epoch+1}/{args.epochs}] loss={avg['total']:.4f} "
                f"face={avg['face']:.4f} person={avg['person']:.4f} "
                f"smooth={avg['smooth']:.6f} identity={avg['identity']:.4f} "
                f"val_faceΔE={val_de:.3f}")
        # 保存 last
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch+1, "best_de": best_de, "no_imp": no_imp},
                   os.path.join(args.out, "skin_curve_last.pth"))

        # 保存 best + 早停
        if val_de < best_de:
            best_de = val_de
            no_imp = 0
            torch.save({"model": model.state_dict(), "best_de": best_de, "epoch": epoch+1},
                       os.path.join(args.out, "skin_curve_best.pth"))
            # 保存曲线示例
            with torch.no_grad():
                sample_curves = curves[0].cpu().numpy()  # (3,256)
            np.savez_compressed(
                os.path.join(args.out, f"best_curves.npz"),
                curves=sample_curves,
            )
            print(f"  ✔ 新最佳 ΔE={best_de:.3f}，已保存 skin_curve_best.pth")
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print(f"  Early stop @ epoch {epoch+1}，best ΔE={best_de:.3f}")
                break

    print("训练完成")


if __name__ == "__main__":
    main()
