# -*- coding: utf-8 -*-
"""
阶段3 v6：曲线(冻结) + 肤色修正(冻结，新训练版) + 残差 U-Net（空间门控）

关键改动（相比 v5）：
  ★ Stage2 调用与推理严格对齐：用 skin_broad 做输入 mask + 只在 mask 内应用
  ★ 强制使用 RGBSkinDataset（必须有 skin_broad 字段）
  ★ 支持 U-Net 新 spatial gate：返回 (output, gate) 用于正则
  ★ 损失新增 easy_res_penalty：base 已接近 target 的像素，惩罚 U-Net 乱动
  ★ gate_reg 从标量改为对空间门控均值做约束

★ 若 checkpoint 存在 skin_curve_best.pth，则自动接入肤色修正
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.color import rgb2lab, deltaE_ciede2000
import numpy as np

from config import Config
from dataset import make_rgb_split, RGBSkinDataset
from models.curve_predictor import CurvePredictor
from models.residual_unet import ResidualUNet
from models.skin_curve_net import SkinCurveNet, apply_lut


# ============ 可调超参 ============
UNET_LR = 1e-4
UNET_WEIGHT_DECAY = 1e-6
RES_TARGET = 0.05            # 残差 RMS 目标
GATE_TARGET = 0.05           # 门控平均值目标（超过才惩罚）
GATE_REG_WEIGHT = 0.5
EASY_RES_WEIGHT = 0.5        # easy 区域残差惩罚权重
EASY_THRESHOLD = 0.05        # |base-tgt| 小于此值视为 easy
BATCH_SIZE = 8
VAL_BATCH_SIZE = 4
EPOCHS = 200
EARLY_STOP_PATIENCE = 30
# ==================================


def rgb_to_lab_torch(img):
    img = img.clamp(0, 1)
    mask = img > 0.04045
    lin = torch.where(mask, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)
    r, g, b = lin[:, 0], lin[:, 1], lin[:, 2]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    def f(t):
        d = 6 / 29
        return torch.where(t > d ** 3,
                           torch.clamp(t, 1e-8) ** (1 / 3),
                           t / (3 * d ** 2) + 4 / 29)

    fx, fy, fz = f(x / 0.95047), f(y), f(z / 1.08883)
    return torch.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], dim=1)


def smooth_residual(res, sigma=2.0):
    device = res.device
    ks = 5
    offset = torch.arange(ks, dtype=torch.float32, device=device) - (ks // 2)
    y, x = torch.meshgrid(offset, offset, indexing="ij")
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, ks, ks).repeat(res.shape[1], 1, 1, 1)
    return torch.nn.functional.conv2d(res, kernel, padding=ks // 2, groups=res.shape[1])


def load_skin_model(path, device):
    """加载已训练的肤色曲线模型并冻结。不存在则返回 None。"""
    if not path or not os.path.exists(path):
        print("[肤色曲线] 未找到，本阶段将仅使用全局曲线 + U-Net")
        return None
    model = SkinCurveNet(n_bins=256).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[肤色曲线] 加载 {path} 并冻结 ✅")
    return model


def _build_base_with_skin(curve_model, skin_model, inp, batch, device):
    """
    冻结的 Stage1 + Stage2 前向：
      - Stage1: curve_model.apply(inp) → base
      - Stage2: skin_model(base, mask=skin_broad) → 只在 skin_broad 内应用
    返回 base (B,3,H,W)
    """
    base = curve_model.apply(inp, resize_size=Config.PRED_INPUT_SIZE)[0]
    if skin_model is not None:
        skin_broad = batch["skin_broad"].to(device)          # (B,1,H,W)
        curves = skin_model(base, mask=skin_broad)
        applied = apply_lut(base, curves)
        # ★ 与 processor._generate_hybrid_stages 一致：只在 mask 内应用
        base = base * (1.0 - skin_broad) + applied * skin_broad
    return base


@torch.no_grad()
def evaluate(curve_model, skin_model, unet, loader, device, max_samples=0):
    curve_model.eval()
    if skin_model is not None:
        skin_model.eval()
    unet.eval()
    de_sum, mae_sum, n = 0.0, 0.0, 0
    for batch in loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        bs = tgt.size(0)

        # ★ 冻结阶段用统一的辅助函数
        base = _build_base_with_skin(curve_model, skin_model, inp, batch, device)

        base_small = torch.nn.functional.interpolate(
            base, size=(256, 256), mode="bilinear", align_corners=False)
        inp_small = torch.nn.functional.interpolate(
            inp, size=(256, 256), mode="bilinear", align_corners=False)

        # ★ U-Net 新接口：return_gate=True，evaluate 里不需要 gate，用 _ 接
        residual, _ = unet(torch.cat([inp_small, base_small], dim=1), return_gate=True)
        residual = smooth_residual(residual)
        residual = torch.nn.functional.interpolate(
            residual, size=inp.shape[-2:], mode="bilinear", align_corners=False)

        out = (base + residual).clamp(0, 1)

        mae_sum += (out - tgt).abs().mean().item() * bs
        pred = (out.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        ref = (tgt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        for i in range(bs):
            de_sum += deltaE_ciede2000(rgb2lab(pred[i]), rgb2lab(ref[i])).mean()
        n += bs
        if max_samples and n >= max_samples:
            break
    return {"de": de_sum / max(n, 1), "mae": mae_sum / max(n, 1)}


def main():
    cfg = Config()
    cfg.setup_dirs()
    torch.manual_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    # ===== 1. 冻结全局曲线 =====
    curve_model = CurvePredictor(
        3, cfg.LUT_DIM, cfg.PRED_BASE, cfg.PRED_RESIDUAL_MAX, cfg.PRED_USE_COUPLING
    ).to(device)
    curve_model.set_global_curve(
        torch.load(cfg.GLOBAL_CURVE_TENSOR_PATH, map_location="cpu").float()
    )
    pred_ckpt = os.path.join(cfg.CHECKPOINT_DIR, "curve_pred_best.pth")
    if os.path.exists(pred_ckpt):
        curve_model.load_state_dict(torch.load(pred_ckpt, map_location=device))
    curve_model.eval()
    for p in curve_model.parameters():
        p.requires_grad_(False)
    print(f"[全局曲线] 冻结 ✅")

    # ===== 2. 冻结肤色曲线（新训练的）=====
    skin_model = load_skin_model(
        os.path.join(cfg.CHECKPOINT_DIR, "skin_curve_best.pth"), device
    )
    if skin_model is None:
        raise RuntimeError(
            "本脚本要求提供已训练的 skin_curve_best.pth（新版本）。"
            "请先运行 train_skin_curve.py 训练 Stage2。"
        )

    # ===== 3. U-Net（空间门控）=====
    unet = ResidualUNet(in_ch=6, base=48, out_ch=3).to(device)
    n_unet = sum(p.numel() for p in unet.parameters())
    print(f"[U-Net] 参数量: {n_unet:,}")

    # 初始化自检：初始残差 std 应 ≈ 0.0067
    with torch.no_grad():
        test_in = torch.rand(1, 6, 64, 64, device=device)
        test_out, test_gate = unet(test_in, return_gate=True)
        test_std = test_out.std().item()
        gate_mean = test_gate.mean().item()
        assert test_std < 0.02, f"初始残差 std={test_std}, 必须 < 0.02"
        print(f"[U-Net] 初始残差 std={test_std:.6f}, gate_mean={gate_mean:.4f} → 安全接入 ✅")

    # ===== 4. 数据（强制 RGBSkinDataset，必须有 skin_broad）=====
    pairs, train_idx, val_idx = make_rgb_split(cfg.DATA_DIR, cfg.TRAIN_RATIO, cfg.SEED)
    train_ds = RGBSkinDataset(
        [pairs[i] for i in train_idx], cfg.SKIN_MASK_DIR, cfg.CACHE_DIR, cfg.IMG_SIZE
    )
    val_ds = RGBSkinDataset(
        [pairs[i] for i in val_idx], cfg.SKIN_MASK_DIR, cfg.CACHE_DIR, cfg.IMG_SIZE
    )
    print(f"[数据] RGBSkinDataset | train={len(train_ds)} val={len(val_ds)}")

    kw = dict(num_workers=0, pin_memory=False)
    t_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **kw)
    v_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, **kw)

    # ===== 5. 基线评估 =====
    base_m = evaluate(curve_model, skin_model, unet, v_loader, device)
    base_de = base_m["de"]
    print(f"\n[基线] 纯曲线(全局+新肤色) ΔE00 = {base_de:.3f}  MAE={base_m['mae']:.5f}")

    # ===== 6. 优化器 =====
    opt = optim.Adam(unet.parameters(), lr=UNET_LR, weight_decay=UNET_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    l1 = nn.L1Loss()

    best_de, best_ep, no_imp = float("inf"), 0, 0
    print(f"\n=== Stage3: 曲线(冻结) + 肤色(冻结) + 残差 U-Net (空间门控) ===")

    for ep in range(EPOCHS):
        unet.train()
        pbar = tqdm(t_loader, desc=f"Hybrid {ep+1}/{EPOCHS}")

        for batch in pbar:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)

            # ---- 冻结的 Stage1 + Stage2 ----
            with torch.no_grad():
                base = _build_base_with_skin(curve_model, skin_model, inp, batch, device)

            # ---- U-Net 前向（拿 gate）----
            inp_small = torch.nn.functional.interpolate(
                inp, size=(256, 256), mode="bilinear", align_corners=False
            )
            base_small = torch.nn.functional.interpolate(
                base, size=(256, 256), mode="bilinear", align_corners=False
            )
            tgt_small = torch.nn.functional.interpolate(
                tgt, size=(256, 256), mode="bilinear", align_corners=False
            )

            residual_raw, gate = unet(
                torch.cat([inp_small, base_small], dim=1), return_gate=True
            )
            residual = smooth_residual(residual_raw)
            out_small = (base_small + residual).clamp(0, 1)

            # ---- 损失 ----
            lab_loss = torch.nn.functional.l1_loss(
                rgb_to_lab_torch(out_small), rgb_to_lab_torch(tgt_small)
            )

            # 残差 RMS 正则（只惩罚超过 target 的部分）
            res_rms = (residual ** 2).mean().sqrt()
            res_reg = (res_rms - RES_TARGET).clamp(min=0) ** 2

            # 空间门控正则：希望 gate 平均不超过 GATE_TARGET
            gate_reg = (gate.mean() - GATE_TARGET).clamp(min=0) ** 2

            # ★ easy 区域残差惩罚：base 已接近 target 的地方，U-Net 不该乱动
            base_err = (base_small - tgt_small).abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
            easy_mask = (base_err < EASY_THRESHOLD).float()
            easy_count = easy_mask.sum().clamp(min=1.0)
            easy_res_penalty = (residual.abs() * easy_mask).sum() / easy_count

            loss = (
                1.0 * l1(out_small, tgt_small)
                + 0.3 * lab_loss
                + 1.0 * res_reg
                + GATE_REG_WEIGHT * gate_reg
                + EASY_RES_WEIGHT * easy_res_penalty
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 3.0)
            opt.step()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                res=f"{res_rms.item():.4f}",
                gate=f"{gate.mean().item():.4f}",
                easy=f"{easy_res_penalty.item():.4f}",
            )

        scheduler.step()
        m = evaluate(curve_model, skin_model, unet, v_loader, device)

        tag = ""
        if m["de"] < best_de:
            best_de, best_ep, no_imp = m["de"], ep, 0
            torch.save(unet.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "unet_best.pth"))
            tag = " ★best"
        else:
            no_imp += 1

        # 从 v_loader 里再算一次 gate 均值（评估）
        with torch.no_grad():
            sample_batch = next(iter(v_loader))
            s_in = sample_batch["input"].to(device)
            s_base = _build_base_with_skin(curve_model, skin_model, s_in, sample_batch, device)
            s_in_small = torch.nn.functional.interpolate(s_in, size=(256, 256), mode="bilinear", align_corners=False)
            s_base_small = torch.nn.functional.interpolate(s_base, size=(256, 256), mode="bilinear", align_corners=False)
            _, s_gate = unet(torch.cat([s_in_small, s_base_small], dim=1), return_gate=True)
            val_gate = s_gate.mean().item()

        print(
            f"  Hybrid {ep+1}: ΔE00={m['de']:.3f}  MAE={m['mae']:.5f}  "
            f"gate={val_gate:.4f}{tag}"
        )

        if no_imp >= EARLY_STOP_PATIENCE:
            print(f"  Early stop @ {ep+1}, best ΔE00={best_de:.3f} (ep {best_ep+1})")
            break

    print(f"\n✅ 完成  最佳 ΔE00 = {best_de:.3f} (epoch {best_ep+1})")
    print(f"   基线(纯曲线+肤色) = {base_de:.3f}")
    if best_de < base_de - 0.1:
        print(f"   → U-Net 有效！比基线低 {base_de - best_de:.3f}")
    else:
        print(f"   ⚠ 未明显优于基线")
    torch.save(unet.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "unet_final.pth"))


if __name__ == "__main__":
    main()