# -*- coding: utf-8 -*-
"""train_rgb.py v5 — RGB 空间 1D 曲线，target 用 ICC 转 sRGB，配对与诊断一致。"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageCms
from tqdm import tqdm

from config import Config
from models.curve_1d import Curve1D
from utils.losses import SoftHistogramLoss, MonotoneLoss


# ---------- ICC：CMYK → sRGB（与最终交付一致） ----------
CMYK_ICC_PATH = os.path.join("utils", "PSOcoated_v3.icc")  # 可改为 cfg.CMYK_ICC_PATH

_srgb_profile = ImageCms.createProfile("sRGB")
_cmyk_profile = ImageCms.getOpenProfile(CMYK_ICC_PATH)
_cmyk_to_rgb = ImageCms.buildTransformFromOpenProfiles(
    _cmyk_profile, _srgb_profile, "CMYK", "RGB",
    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
    flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
)


def discover_pairs(data_dir, input_suffixes=("_input.jpg", "_input.JPG", "_input.jpeg", "_input.png")):
    """健壮配对：与诊断脚本 661 对一致。返回 [(input_path, target_path), ...]"""
    pairs, seen = [], set()
    for f in sorted(os.listdir(data_dir)):
        for suf in input_suffixes:
            if not f.endswith(suf):
                continue
            stem = f[: -len(suf)]
            if stem in seen:
                break
            # 尝试多种 target 后缀
            found = None
            for t_suf in ("_target.jpg", "_target.JPG", "_target.jpeg", "_target.png", "_target.tif", "_target.TIF"):
                tp = os.path.join(data_dir, stem + t_suf)
                if os.path.exists(tp):
                    found = tp
                    break
            if found:
                seen.add(stem)
                pairs.append((os.path.join(data_dir, f), found))
            break
    return pairs


class RGBPairDataset(Dataset):
    def __init__(self, pairs, cache_dir=None, img_size=512):
        self.pairs = pairs
        self.cache_dir = cache_dir
        self.img_size = img_size

    def _load(self, path, is_target):
        cpath = None
        if self.cache_dir and self.img_size < 4000:
            tag = "target_icc" if is_target else "input_rgb"
            cpath = os.path.join(self.cache_dir, f"{os.path.basename(path)}_{self.img_size}_{tag}.npy")
            if os.path.exists(cpath):
                return np.load(cpath).astype(np.float32)

        img = Image.open(path)

        if is_target and img.mode == "CMYK":
            # ★ 关键：用 ICC 把 CMYK 转成 sRGB，与交付链路一致
            img = ImageCms.applyTransform(img, _cmyk_to_rgb)
        else:
            img = img.convert("RGB")

        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.shape[0] != self.img_size or arr.shape[1] != self.img_size:
            pil = Image.fromarray((arr * 255).astype(np.uint8))
            w, h = pil.size
            s = min(w, h)
            left, top = (w - s) // 2, (h - s) // 2
            pil = pil.crop((left, top, left + s, top + s))
            arr = np.array(pil.resize((self.img_size, self.img_size)), dtype=np.float32) / 255.0

        if cpath:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(cpath, arr.astype(np.float16))
        return arr.astype(np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_p, tgt_p = self.pairs[idx]
        inp = self._load(inp_p, is_target=False)
        tgt = self._load(tgt_p, is_target=True)
        return {
            "input": torch.from_numpy(inp).permute(2, 0, 1).float(),
            "target": torch.from_numpy(tgt).permute(2, 0, 1).float(),
        }


@torch.no_grad()
def eval_rgb(curve, loader, device, max_samples=0):
    """max_samples=0 表示验证全部。ΔE00 用 skimage，与基线同口径。"""
    from skimage.color import rgb2lab, deltaE_ciede2000
    curve.eval()
    de_sum, mae_sum, n = 0.0, 0.0, 0
    for batch in loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        bs = tgt.size(0)
        cout = curve(inp).clamp(0, 1)
        if torch.isnan(cout).any():
            print("❌ NaN in output, abort eval"); break
        mae_sum += (cout - tgt).abs().mean().item() * bs

        pred = (cout.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        ref = (tgt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        for i in range(len(pred)):
            de_sum += deltaE_ciede2000(rgb2lab(pred[i]), rgb2lab(ref[i])).mean()
        n += bs
        if max_samples and n >= max_samples:
            break
    return {"de": de_sum / max(n, 1), "mae": mae_sum / max(n, 1)}


def train():
    cfg = Config()
    cfg.setup_dirs()
    torch.manual_seed(cfg.SEED)   # ★ 可复现
    device = torch.device("cpu")
    img_size = getattr(cfg, "IMG_SIZE", 512)
    print(f"[设备] {device} | IMG_SIZE={img_size}")

    # ★ 用诊断一致的配对，得到 661 对
    all_pairs = discover_pairs(cfg.DATA_DIR)
    print(f"[配对] total={len(all_pairs)}")

    # 划分
    rng = np.random.default_rng(cfg.SEED)
    idxs = np.arange(len(all_pairs))
    rng.shuffle(idxs)
    n_train = int(len(idxs) * cfg.TRAIN_RATIO)
    train_idx, val_idx = idxs[:n_train], idxs[n_train:]
    print(f"[split] train={len(train_idx)} val={len(val_idx)} overlap=0")

    train_ds = RGBPairDataset([all_pairs[i] for i in train_idx], cache_dir=cfg.CACHE_DIR, img_size=img_size)
    val_ds = RGBPairDataset([all_pairs[i] for i in val_idx], cache_dir=cfg.CACHE_DIR, img_size=img_size)

    kw = dict(num_workers=0, pin_memory=False)
    t_loader = DataLoader(train_ds, batch_size=cfg.CURVE_BATCH_SIZE, shuffle=True, **kw)
    v_loader = DataLoader(val_ds, batch_size=cfg.CURVE_BATCH_SIZE, shuffle=False, **kw)

    curve = Curve1D(channels=3, n_bins=cfg.LUT_DIM).to(device)
    l1 = nn.L1Loss()
    hist = SoftHistogramLoss(cfg.HIST_BINS, cfg.HIST_DARK_WEIGHT,
                             cfg.HIST_MID_WEIGHT, cfg.HIST_BRIGHT_WEIGHT)
    mono = MonotoneLoss()
    opt = optim.Adam(curve.parameters(), lr=cfg.CURVE_LR)

    print(f"\n=== RGB 1D 曲线 (L1 + Hist + Mono; 验证真ΔE00, 全量) ===")

    best_de, best_ep, no_imp = float("inf"), 0, 0
    for ep in range(cfg.CURVE_EPOCHS):
        curve.train()
        pbar = tqdm(t_loader, desc=f"Curve {ep+1}/{cfg.CURVE_EPOCHS}")
        for batch in pbar:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            opt.zero_grad(set_to_none=True)
            cout = curve(inp)
            loss = (cfg.L1_WEIGHT * l1(cout, tgt)
                    + cfg.HIST_WEIGHT * hist(cout, tgt)
                    + cfg.MONOTONE_WEIGHT * mono(curve.get_curves()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(curve.parameters(), cfg.GRAD_CLIP)
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}", mae=f"{(cout-tgt).abs().mean().item():.4f}")

        m = eval_rgb(curve, v_loader, device, max_samples=0)  # ★ 全量验证
        print(f"  Curve {ep+1}: ΔE00={m['de']:.3f}  MAE={m['mae']:.5f}")

        if m["de"] < best_de:
            best_de, best_ep, no_imp = m["de"], ep, 0
            torch.save(curve.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "curve_rgb_best.pth"))
            torch.save(curve.get_curves().detach().cpu(), os.path.join(cfg.CHECKPOINT_DIR, "curves_rgb.pt"))
        else:
            no_imp += 1
        if no_imp >= cfg.EARLY_STOP_PATIENCE:
            print(f"  Early stop @ {ep+1}, best ΔE00={best_de:.3f} (ep {best_ep+1})")
            break

    print(f"\n✅ 完成  最佳验证 ΔE00 = {best_de:.3f} (epoch {best_ep+1})")
    if best_de <= 3.0:
        print("   → ΔE00≤3.0，1D RGB 曲线达标")
    elif best_de <= 3.5:
        print("   → 优于 CNN64(3.50)")
    else:
        print("   → 未达 3.5，考虑通道耦合")


if __name__ == "__main__":
    train()