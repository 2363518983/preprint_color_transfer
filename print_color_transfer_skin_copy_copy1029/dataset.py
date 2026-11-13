# -*- coding: utf-8 -*-
"""
统一数据加载：
- PrintDataset: CMYK 4通道（阶段1）
- RGBPairDataset + discover_pairs + make_rgb_split: RGB 输入 + ICC 转换 sRGB 目标（阶段2）
"""
import os
import re
import glob
import hashlib
import random
import numpy as np
from PIL import Image, ImageCms
from torch.utils.data import Dataset

from utils.icc_color import read_rgb, read_cmyk_from_jpg, rgb_to_cmyk_icc

try:
    import torch
except ImportError:
    torch = None


# ==================== 通用工具 ====================

def center_crop_square(pil):
    w, h = pil.size
    if w == h:
        return pil
    if w > h:
        left = (w - h) // 2
        return pil.crop((left, 0, left + h, h))
    top = (h - w) // 2
    return pil.crop((0, top, w, top + w))


def _clean_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = stem.replace("_input", "").replace("_target", "")
    stem = re.sub(r"\(\d+\)\s*$", "", stem)
    stem = re.sub(r"\s+\d+\s*$", "", stem)
    return stem.strip()


def _file_content_hash(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return hashlib.md5(path_or_bytes).hexdigest()
    if isinstance(path_or_bytes, np.ndarray):
        return hashlib.md5(path_or_bytes.tobytes()).hexdigest()
    h = hashlib.md5()
    with open(path_or_bytes, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resize(arr, size, mode):
    pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode)
    pil = center_crop_square(pil).resize((size, size), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


# ==================== PrintDataset（阶段1，保持原样） ====================

def _discover_pairs(data_dir):
    input_files = sorted(glob.glob(os.path.join(data_dir, "*_input.jpg")))
    input_files += sorted(glob.glob(os.path.join(data_dir, "*_input.jpeg")))
    pairs, missing, seen = [], 0, set()
    for inp in input_files:
        stem = _clean_stem(inp)
        tgt = os.path.join(data_dir, stem + "_target.jpg")
        if not os.path.exists(tgt):
            tgt = os.path.join(data_dir, stem + "_target.jpeg")
        if not os.path.exists(tgt):
            missing += 1
            continue
        if stem in seen:
            continue
        seen.add(stem)
        pairs.append((inp, tgt))
    if missing > 0:
        print(f"  ⚠ {missing} 张 input 找不到 target，已跳过")
    return pairs


def make_split(data_dir, train_ratio=0.8, seed=42):
    pairs = _discover_pairs(data_dir)
    n = len(pairs)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    sp = int(n * train_ratio)
    return pairs, idxs[:sp], idxs[sp:]


class PrintDataset(Dataset):
    def __init__(self, data_dir, split="train", augment=True, cfg=None,
                 train_ratio=0.8, seed=42, cache_dir=None, cmyk_icc_md5=None,
                 full_res=False, indices=None, all_pairs=None):
        self.data_dir = data_dir
        self.split = split
        self.augment = augment
        self.seed = seed
        self.cache_dir = cache_dir
        self.full_res = full_res
        self.img_size = cfg.IMG_SIZE if cfg is not None else 512
        self.cmyk_icc_md5 = cmyk_icc_md5 or "noicc"

        if all_pairs is not None:
            full = all_pairs
        else:
            full = _discover_pairs(data_dir)

        if indices is not None:
            self.pairs = [full[i] for i in indices]
        else:
            random.seed(seed)
            idxs = list(range(len(full)))
            random.shuffle(idxs)
            sp = int(len(idxs) * train_ratio)
            self.pairs = [full[i] for i in (idxs[:sp] if split == "train" else idxs[sp:])]

        print(f"  [{split}] {len(self.pairs)} pairs (from {data_dir})")

    def _cache_key(self, path, suffix):
        base = _file_content_hash(path)
        return f"{base}_{self.img_size}_{self.cmyk_icc_md5}_{suffix}.npy"

    def _cache_path(self, path, suffix):
        key = self._cache_key(path, suffix)
        return os.path.join(self.cache_dir, key) if self.cache_dir else None

    def _prepare(self, path, suffix, reader, mode):
        cpath = self._cache_path(path, suffix)
        if cpath and os.path.exists(cpath):
            return np.load(cpath).astype(np.float32)
        arr = reader(path)
        if not self.full_res:
            arr = _resize(arr, self.img_size, mode)
        if cpath:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(cpath, arr.astype(np.float16))
        return arr.astype(np.float32)

    def _prepare_cmyk_from_rgb(self, rgb_arr, source_path, suffix="input_cmyk"):
        cpath = self._cache_path(source_path, suffix) if self.cache_dir else None
        if cpath and os.path.exists(cpath):
            return np.load(cpath).astype(np.float32)
        cmyk = rgb_to_cmyk_icc(rgb_arr)
        if cpath:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(cpath, cmyk.astype(np.float16))
        return cmyk.astype(np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_path, tgt_path = self.pairs[idx]
        inp = self._prepare(inp_path, "input_rgb", read_rgb, "RGB")
        tgt = self._prepare(tgt_path, "target_cmyk", read_cmyk_from_jpg, "CMYK")
        inp_cmyk = self._prepare_cmyk_from_rgb(inp, inp_path, "input_cmyk")
        if torch is None:
            raise ImportError("dataset 需要 torch")
        return {
            "input":      torch.from_numpy(inp).permute(2, 0, 1).float(),
            "input_cmyk": torch.from_numpy(inp_cmyk).permute(2, 0, 1).float(),
            "target":     torch.from_numpy(tgt).permute(2, 0, 1).float(),
        }


# ==================== RGB 配对 Dataset（阶段2） ====================

def discover_pairs(data_dir, input_suffixes=("_input.jpg", "_input.JPG", "_input.jpeg", "_input.png")):
    """健壮配对：与诊断脚本一致"""
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


def make_rgb_split(data_dir, train_ratio=0.8, seed=42):
    """返回 (pairs, train_indices, val_indices)"""
    pairs = discover_pairs(data_dir)
    idxs = np.arange(len(pairs))
    rng = np.random.default_rng(seed)
    rng.shuffle(idxs)
    n_train = int(len(idxs) * train_ratio)
    return pairs, idxs[:n_train], idxs[n_train:]


class RGBPairDataset(Dataset):
    """input RGB + target(CMYK 经 ICC 转 sRGB)"""

    def __init__(self, pairs, cache_dir=None, img_size=512):
        self.pairs = pairs
        self.cache_dir = cache_dir
        self.img_size = img_size

        cmyk_icc_path = os.path.join("utils", "PSOcoated_v3.icc")
        if os.path.exists(cmyk_icc_path):
            cmyk_profile = ImageCms.getOpenProfile(cmyk_icc_path)
            srgb_profile = ImageCms.createProfile("sRGB")
            self._cmyk_to_rgb = ImageCms.buildTransformFromOpenProfiles(
                cmyk_profile, srgb_profile, "CMYK", "RGB",
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
            )
        else:
            print("⚠ 未找到 PSOcoated_v3.icc，target 按普通 RGB 读取（可能偏色）")
            self._cmyk_to_rgb = None

    def _load(self, path, is_target):
        cpath = None
        if self.cache_dir and self.img_size < 4000:
            tag = "target_icc" if is_target else "input_rgb"
            cpath = os.path.join(self.cache_dir, f"{os.path.basename(path)}_{self.img_size}_{tag}.npy")
            if os.path.exists(cpath):
                return np.load(cpath).astype(np.float32)

        img = Image.open(path)
        if is_target and img.mode == "CMYK":
            if self._cmyk_to_rgb is None:
                img = img.convert("RGB")
            else:
                img = ImageCms.applyTransform(img, self._cmyk_to_rgb)
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
            "input":  torch.from_numpy(inp).permute(2, 0, 1).float(),
            "target": torch.from_numpy(tgt).permute(2, 0, 1).float(),
        }


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    pairs, tr, va = make_rgb_split(data_dir)
    print(f"配对总数: {len(pairs)}, train: {len(tr)}, val: {len(va)}")