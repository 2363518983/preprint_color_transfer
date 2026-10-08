# -*- coding: utf-8 -*-
"""train_hybrid.py v3 — 阶段2: Curve1D(冻结) + 残差U-Net。

修复:
  - eval_rgb 返回 dict，调用处不再对其取 ["de"]（避免 scalar index 报错）
  - 全程 model.train()，BN 行为一致；缓存 base，避免两次前向污染
  - grad_clip 只对可训练参数
  - U-Net 残差初始化为零（BN zero_init + Tanh ≈0），保证 ep1 残差==基线
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageCms
from tqdm import tqdm

from config import Config
from models.residual_unet import HybridColorModel
from utils.losses import SoftHistogramLoss


# ---------- ICC ----------
CMYK_ICC_PATH = os.path.join("utils", "PSOcoated_v3.icc")
_srgb_profile = ImageCms.createProfile("sRGB")
_cmyk_profile = ImageCms.getOpenProfile(CMYK_ICC_PATH)
_cmyk_to_rgb = ImageCms.buildTransformFromOpenProfiles(
    _cmyk_profile, _srgb_profile, "CMYK", "RGB",
    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
    flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
)


def discover_pairs(data_dir, input_suffixes=("_input.jpg", "_input.JPG", "_input.jpeg", "_input.png")):
    pairs, seen = [], set()
    for f in sorted(os.listdir(data_dir)):
        for suf in input_suffixes:
            if not f.endswith(suf):
                continue
            stem = f[: -len(suf)]
            if stem in seen:
                break
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
            img = ImageCms.applyTransform(img, _cmk_to_rgb if False else _cmyk_to_rgb)
        else:
            img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.shape[0] != self.img_size or arr.shape[1] != self.img_size:
            pil = Image.fromarray((arr * 255).astype(np.uint8))
            w, h = pil.size
            s = min(w, h)
            pil = pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            arr = np.array(pil.resize((self.img_size, self.img_size)), dtype=np.float32) / 255.0
        if cpath:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(cpath, arr.astype(np.float16))
        return arr.astype(np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_p, tgt_p = self.pairs[idx]
        inp = self._load(inp_p, False)
        tgt = self._load(tgt_p, True)
        return {
            "input": torch.from_numpy(inp).permute(2, 0, 1).float(),
            "target": torch.from_numpy(tgt).permute(2, 0, 1).float(),
        }


# ---------- Lab 损失 ----------
def rgb_to_lab_torch(img):
    img = img.clamp(0, 1)
    mask = img > 0.04045
    linear = torch.where(mask, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)
    m = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]],
                     device=img.device, dtype=img.dtype)
    xyz = torch.einsum("ij,bjhw->bihw", m, linear)
    xyz = xyz / torch.tensor([0.95047, 1.0, 1.08883], device=img.device).view(1, 3, 1, 1)
    eps = 1e-8

    def f(t):
        delta = 6 / 29
        return torch.where(t > delta ** 3, torch.clamp(t, min=eps) ** (1 / 3),
                           t / (3 * delta ** 2) + 4 / 29)

    fx, fy, fz = f(xyz[:, 0]), f(xyz[:, 1]), f(xyz[:, 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return torch.stack([L, a, b_val], dim=1)


@torch.no_grad()
def evaluate(model, loader, device):
    """单次前向同时得 base 和 full，避免 BN 被两次前向污染。
    返回 dict: {de_base, de_full, mae_full}"""
    from skimage.color import rgb2lab, deltaE_ciede2000
    de_base, de_full, mae_full, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        bs = tgt.size(0)

        base = model.forward_base(inp)                # (B,3,H,W)
        full = model(inp, use_unet=True).clamp(0, 1) # (B,3,H,W)  ★ 单次前向

        mae_full += (full - tgt).abs().mean().item() * bs

        for out, acc in ((base, de_base), (full, de_full)):
            pass
        # base ΔE
        pb = (base.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        rf = (tgt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        for i in range(bs):
            de_base += deltaE_ciede2000(rgb2lab(pb[i]), rgb2lab(rf[i])).mean()
        # full ΔE
        pf = (full.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        for i in range(bs):
            de_full += deltaE_ciede2000(rgb2lab(pf[i]), rgb2lab(rf[i])).mean()
        n += bs
    return {"de_base": de_base / max(n, 1), "de_full": de_full / max(n, 1),
            "mae": mae_full / max(n, 1)}


def train():
    cfg = Config()
    cfg.setup_dirs()
    torch.manual_seed(cfg.SEED)
    device = torch.device("cpu")
    img_size = getattr(cfg, "IMG_SIZE", 512)
    print(f"[设备] {device} | IMG_SIZE={img_size}")

    curve_ckpt = os.path.join(cfg.CHECKPOINT_DIR, "curve_rgb_best.pth")
    if not os.path.exists(curve_ckpt):
        print(f"❌ 找不到 {curve_ckpt}，请先跑 train_rgb.py (阶段1)")
        return
    print(f"[Curve1D] 加载基线: {curve_ckpt}")

    all_pairs = discover_pairs(cfg.DATA_DIR)
    print(f"[配对] total={len(all_pairs)}")
    idxs = np.arange(len(all_pairs))
    rng = np.random.default_rng(cfg.SEED)
    rng.shuffle(idxs)
    n_train = int(len(idxs) * cfg.TRAIN_RATIO)
    train_idx = idxs[:n_train].tolist()
    val_idx = idxs[n_train:].tolist()
    print(f"[split] train={len(train_idx)} val={len(val_idx)} overlap=0")

    kw = dict(num_workers=0, pin_memory=False)
    train_ds = RGBPairDataset([all_pairs[i] for i in train_idx], cfg.CACHE_DIR, img_size)
    val_ds = RGBPairDataset([all_pairs[i] for i in val_idx], cfg.CACHE_DIR, img_size)
    t_loader = DataLoader(train_ds, batch_size=cfg.CURVE_BATCH_SIZE, shuffle=True, **kw)
    v_loader = DataLoader(val_ds, batch_size=cfg.CURVE_BATCH_SIZE, shuffle=False, **kw)

    model = HybridColorModel(n_bins=cfg.LUT_DIM, unet_base=16, residual_scale=0.03).to(device)
    model.curve.load_state_dict(torch.load(curve_ckpt, map_location=device))
    model.freeze_curve()

    # ★ U-Net 残差初始化为零（BN zero_init + Tanh ≈0）
    # for m in model.unet.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         nn.init.zeros_(m.weight)   # gamma=0 → 输出恒为 beta(=0)
    #         nn.init.zeros_(m.bias)

    l1 = nn.L1Loss()
    hist = SoftHistogramLoss(cfg.HIST_BINS, cfg.HIST_DARK_WEIGHT, cfg.HIST_MID_WEIGHT, cfg.HIST_BRIGHT_WEIGHT)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(trainable, lr=1e-4)
    lab_weight = getattr(cfg, "LAB_WEIGHT", 0.1)

    print(f"\n=== 阶段2: Curve1D(冻结) + 残差U-Net (L1 + {lab_weight}·LabL1 + Hist) ===")
    print("★ 全程 model.train()，单次前向得 base+full，BN 行为一致")

    best_de, best_ep, no_imp = float("inf"), 0, 0
    for ep in range(getattr(cfg, "UNET_EPOCHS", 60)):
        model.train()   # ★ train 模式（曲线已 freeze，BN 用 batch stats）
        pbar = tqdm(t_loader, desc=f"Unet {ep+1}/{getattr(cfg,'UNET_EPOCHS',60)}")
        for batch in pbar:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            opt.zero_grad(set_to_none=True)
            cout = model(inp, use_unet=True)         # (B,3,H,W)

            lab_pred = rgb_to_lab_torch(cout)
            lab_tgt = rgb_to_lab_torch(tgt)
            loss = (cfg.L1_WEIGHT * l1(cout, tgt)
                    + lab_weight * nn.L1Loss()(lab_pred, lab_tgt)
                    + cfg.HIST_WEIGHT * hist(cout, tgt))
            loss.backward()
            # ★ grad_clip 只对可训练参数，避免动到冻结的 curve
            torch.nn.utils.clip_grad_norm_(trainable, cfg.GRAD_CLIP)
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}", mae=f"{(cout-tgt).abs().mean().item():.4f}")

        # 验证：单次前向同时得 base 与 full
        m = evaluate(model, v_loader, device)
        print(f"  Ep {ep+1}: [仅曲线]ΔE00={m['de_base']:.3f}  [+残差]ΔE00={m['de_full']:.3f}  MAE={m['mae']:.5f}")

        # ★ m 是 dict，m['de_full'] 是标量，best_de 也是标量——不再混用
        score = m["de_full"]
        if score < best_de:
            best_de, best_ep, no_imp = score, ep, 0
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "hybrid_best.pth"))
        else:
            no_imp += 1
        if no_imp >= cfg.EARLY_STOP_PATIENCE:
            print(f"  Early stop @ {ep+1}, best ΔE00={best_de:.3f} (ep {best_ep+1})")
            break

    print(f"\n✅ 阶段2 完成  最佳 ΔE00 = {best_de:.3f} (epoch {best_ep+1})")
    print(f"   基线(Curve1D 3.69) → 混合模型 {best_de:.3f}")
    if best_de < 3.69:
        print(f"   → U-Net 残差带来 {3.69 - best_de:.2f} ΔE 提升 🎉")
    else:
        print("   → 未提升: 检查 BN/初始化，或考虑解冻 curve 联合微调")
    torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "hybrid_final.pth"))


if __name__ == "__main__":
    train()