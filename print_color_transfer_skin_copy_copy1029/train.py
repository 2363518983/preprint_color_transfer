# -*- coding: utf-8 -*-
"""阶段1 v3: CPU 优化 —— 训练L1(快), 验证ΔE(准), 向量化曲线, 缓存预热。"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import make_split, PrintDataset
from models.curve_1d import Curve1D
from utils.icc_color import init_icc, cmyk_to_rgb_icc
from utils.losses import SoftHistogramLoss, MonotoneLoss, ciede2000_rgb


def to_float32(batch):
    return {k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def cmyk_to_rgb_batch(cmyk_t):
    arr = cmyk_t.detach().permute(0, 2, 3, 1).cpu().numpy()
    return torch.from_numpy(cmyk_to_rgb_icc(arr)).permute(0, 3, 1, 2)


def evaluate(curve, loader, device, cfg, max_samples=32):
    curve.eval()
    mae_sum, de_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = to_float32(batch)
            inp_cmyk = batch["input_cmyk"].to(device)
            tgt = batch["target"].to(device)
            bs = tgt.size(0); n += bs
            cout = curve(inp_cmyk)
            mae_sum += (cout - tgt).abs().mean().item() * bs

            if cfg.COMPUTE_DE_IN_EVAL:
                scale = cfg.EVAL_DOWNSAMPLE
                def _ds(x): return nn.functional.interpolate(x, size=(scale, scale), mode="bilinear", align_corners=False)
                pred_rgb = cmyk_to_rgb_batch(_ds(cout))
                tgt_rgb = cmyk_to_rgb_batch(_ds(tgt))
                de_sum += ciede2000_rgb(pred_rgb, tgt_rgb).item() * bs
            if n >= max_samples:
                break
    out = {"cmyk_mae": mae_sum / max(n, 1)}
    out["delta_e"] = de_sum / max(n, 1) if cfg.COMPUTE_DE_IN_EVAL else float("nan")
    return out


def train():
    cfg = Config()
    cfg.setup_dirs()
    init_icc(cfg.CMYK_ICC_PATH, cfg.sRGB_ICC_PATH, intent=cfg.RENDERING_INTENT, bpc=cfg.BLACK_POINT_COMPENSATION)
    device = torch.device("cpu")   # ★ CPU 专用，强制明确
    print(f"[设备] {device} (PyTorch {torch.__version__})")

    icc_md5 = None
    if cfg.CMYK_ICC_PATH and os.path.exists(cfg.CMYK_ICC_PATH):
        import hashlib
        icc_md5 = hashlib.md5(open(cfg.CMYK_ICC_PATH, "rb").read()).hexdigest()

    all_pairs, train_idx, val_idx = make_split(cfg.DATA_DIR, train_ratio=cfg.TRAIN_RATIO, seed=cfg.SEED)
    print(f"[split] total={len(all_pairs)} train={len(train_idx)} val={len(val_idx)} (overlap={len(set(train_idx)&set(val_idx))})")

    common = dict(cfg=cfg, cache_dir=cfg.CACHE_DIR, cmyk_icc_md5=icc_md5, all_pairs=all_pairs)
    train_ds = PrintDataset(cfg.DATA_DIR, "train", augment=False, indices=train_idx, **common)
    val_ds = PrintDataset(cfg.DATA_DIR, "val", augment=False, indices=val_idx, **common)

    kw = dict(num_workers=0, pin_memory=False)   # ★ CPU 下 num_workers=0 反而快（避免进程开销）
    t_loader = DataLoader(train_ds, batch_size=cfg.CURVE_BATCH_SIZE, shuffle=True, **kw)
    v_loader = DataLoader(val_ds, batch_size=cfg.CURVE_BATCH_SIZE, shuffle=False, **kw)

    l1 = nn.L1Loss()
    hist = SoftHistogramLoss(cfg.HIST_BINS, cfg.HIST_DARK_WEIGHT, cfg.HIST_MID_WEIGHT, cfg.HIST_BRIGHT_WEIGHT)
    mono = MonotoneLoss()
    curve = Curve1D(channels=4, n_bins=cfg.LUT_DIM)
    opt = optim.Adam(curve.parameters(), lr=cfg.CURVE_LR)

    best_de, best_ep, no_imp = float("inf"), 0, 0
    print(f"\n=== 阶段1: 1D 曲线 (CPU, 训练L1, 验证ΔE; eval_de={cfg.COMPUTE_DE_IN_EVAL}) ===")

    for ep in range(cfg.CURVE_EPOCHS):
        curve.train()
        pbar = tqdm(t_loader, desc=f"Curve {ep+1}/{cfg.CURVE_EPOCHS}")
        for batch in pbar:
            batch = to_float32(batch)
            inp_cmyk = batch["input_cmyk"]
            tgt = batch["target"]

            opt.zero_grad(set_to_none=True)
            cout = curve(inp_cmyk)
            loss = (cfg.L1_WEIGHT * l1(cout, tgt)
                    + cfg.HIST_WEIGHT * hist(cout, tgt)
                    + cfg.MONOTONE_WEIGHT * mono(curve.get_curves()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(curve.parameters(), cfg.GRAD_CLIP)
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}", mae=f"{(cout-tgt).abs().mean().item():.4f}")

        m = evaluate(curve, v_loader, device, cfg)
        de_str = f"ΔE={m['delta_e']:.3f}" if cfg.COMPUTE_DE_IN_EVAL else "ΔE=skip"
        print(f"  Curve {ep+1}: {de_str}  MAE={m['cmyk_mae']:.5f}")

        score = m["delta_e"] if cfg.COMPUTE_DE_IN_EVAL else m["cmyk_mae"]
        if score < best_de:
            best_de, best_ep, no_imp = score, ep, 0
            torch.save(curve.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "curve_best.pth"))
            torch.save(curve.get_curves().detach(), os.path.join(cfg.CHECKPOINT_DIR, "best_curves.pt"))
        else:
            no_imp += 1
        if no_imp >= cfg.EARLY_STOP_PATIENCE:
            print(f"  Early stop @ {ep+1}, best={'ΔE' if cfg.COMPUTE_DE_IN_EVAL else 'MAE'}={best_de:.3f} (ep {best_ep+1})")
            break

    curve.load_state_dict(torch.load(os.path.join(cfg.CHECKPOINT_DIR, "curve_best.pth")))
    final = evaluate(curve, v_loader, device, cfg)
    print(f"\n✅ 阶段1 完成  最佳={'ΔE' if cfg.COMPUTE_DE_IN_EVAL else 'MAE'}={best_de:.3f} (ep {best_ep+1})")
    print(f"   最终验证 ΔE={final['delta_e']:.3f}  MAE={final['cmyk_mae']:.5f}")

    if cfg.COMPUTE_DE_IN_EVAL:
        if best_de <= 3.0: print("   → ΔE≤3.0, 1D 曲线天花板达标, 可交付纯曲线")
        elif best_de <= 3.5: print("   → 优于 CNN64(3.50), 1D 曲线已足够")
        else: print("   → 未达 3.5, 需通道联合(3D LUT)/残差CNN, 设 ENABLE_STAGE2=True")
    torch.save(curve.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "curve_final.pth"))


if __name__ == "__main__":
    train()