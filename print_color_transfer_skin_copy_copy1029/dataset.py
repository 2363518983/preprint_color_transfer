# -*- coding: utf-8 -*-
"""
统一数据加载（优化版 v3）：
- PrintDataset: CMYK 4通道（阶段1）
- RGBPairDataset: RGB 输入 + ICC 转换 sRGB 目标（阶段2）
- RGBSkinDataset: RGB + person/face mask（阶段3 肤色调整，带 npy 缓存）

【修补】
- RGBSkinDataset：mask 现在与图像执行相同的中心裁剪 + resize，避免拉伸错位
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


def _preprocess_image(path, is_target, img_size, cmyk_to_rgb_transform):
    """加载并预处理单张图：转RGB + 中心裁剪 + resize，返回 float32 [0,1]"""
    img = Image.open(path)
    if is_target and img.mode == "CMYK":
        if cmyk_to_rgb_transform is None:
            img = img.convert("RGB")
        else:
            img = ImageCms.applyTransform(img, cmyk_to_rgb_transform)
    else:
        img = img.convert("RGB")

    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.shape[0] != img_size or arr.shape[1] != img_size:
        pil = Image.fromarray((arr * 255).astype(np.uint8), "RGB")
        w, h = pil.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        pil = pil.crop((left, top, left + s, top + s))
        arr = np.array(pil.resize((img_size, img_size)), dtype=np.float32) / 255.0
    return arr.astype(np.float32)


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

        arr = _preprocess_image(path, is_target, self.img_size, self._cmyk_to_rgb)

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


# ==================== 肤色调整 Dataset（阶段3，带 npy 缓存） ====================

def _resize_mask(mask, img_size):
    """mask (H,W) float32 -> resize 到 (img_size, img_size)"""
    if mask.shape[0] == img_size and mask.shape[1] == img_size:
        return mask
    pil = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), "L")
    return np.array(pil.resize((img_size, img_size), Image.BILINEAR), dtype=np.float32) / 255.0


def _center_crop_mask(mask, img_size):
    """
    【新增】对 mask 执行与图像相同的中心正方形裁剪，再 resize。
    避免原图非方形时 mask 被拉伸、与图像内容错位。
    mask: (H,W) 或 (H,W,...) ndarray
    返回: (img_size, img_size) float32
    """
    pil = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), "L")
    w, h = pil.size
    s = min(w, h)
    if w > h:
        left = (w - h) // 2
        pil = pil.crop((left, 0, left + h, h))
    else:
        top = (h - w) // 2
        pil = pil.crop((0, top, w, top + w))
    return np.array(pil.resize((img_size, img_size), Image.BILINEAR), dtype=np.float32) / 255.0


class RGBSkinDataset(Dataset):
    """
    输入: RGB(input) + target(RGB, 由CMYK经ICC转来)
    附加: person_mask, face_mask (来自 mask_cache)
    支持 npy 缓存：首次读取时保存 {stem}_cache.npz，后续直接读取加速

    缓存文件命名规则（避免与 mask 目录混淆）：
        {cache_dir}/{stem}_size{img_size}.npz  含 input/target/person/face
    """

    def __init__(self, pairs, mask_dir, cache_dir=None, img_size=512):
        self.pairs = pairs
        self.mask_dir = mask_dir
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
            print("⚠ 未找到 PSOcoated_v3.icc，target 按普通 RGB 读取")
            self._cmyk_to_rgb = None

        # 预检查 mask 文件是否齐全
        missing = []
        for inp_p, _ in self.pairs[:5]:  # 只抽查前5个
            stem = os.path.splitext(os.path.basename(inp_p))[0].replace("_input", "")
            p = os.path.join(mask_dir, f"{stem}_person.npy")
            f = os.path.join(mask_dir, f"{stem}_face.npy")
            if not os.path.exists(p) or not os.path.exists(f):
                missing.append(stem)
        if missing:
            print(f"⚠ mask 缺失示例: {missing[:3]} ... 请先运行 generate_masks.py")

    def _cache_path(self, stem):
        """返回该 stem 对应的缓存文件路径"""
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{stem}_size{self.img_size}.npz")

    @staticmethod
    def _compute_skin_face(face_mask, rgb_arr):
        """YuNet人脸 ∩ YCrCb肤色检测，返回 0/1 mask"""
        import cv2
        img = (np.clip(rgb_arr, 0, 1) * 255).astype(np.uint8)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        skin = ((ycrcb[..., 1] >= 133) & (ycrcb[..., 1] <= 173) &
                (ycrcb[..., 2] >= 77)  & (ycrcb[..., 2] <= 127)).astype(np.float32)
        skin_face = skin * face_mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_face = cv2.morphologyEx(skin_face, cv2.MORPH_OPEN, kernel)
        return skin_face
    @staticmethod
    def _compute_skin_broad(rgb_arr):
        r = rgb_arr[..., 0]
        g = rgb_arr[..., 1]
        b = rgb_arr[..., 2]
        mx = np.max(rgb_arr, axis=-1)
        mn = np.min(rgb_arr, axis=-1)
        denom = np.maximum(mx, 1e-6)              # ★ 改为 maximum 而不是 +eps
        sat = (mx - mn) / denom
        sat = np.nan_to_num(sat, nan=0.0, posinf=1.0, neginf=0.0)   # ★ 处理 NaN/Inf
        mask = (r > 0.35) & (g > 0.15) & (b > 0.05) & (r > g) & (g > b) & (sat > 0.15)
        return mask.astype(np.float32)

    def _load_cached_or_build(self, stem, inp_p, tgt_p):
        """
        首次加载：处理 input/target/person/face/skin_face/skin_broad 并保存 npz 缓存
        后续加载：直接读取 npz（老缓存缺 skin_broad 会补算并回写）
        返回: (input, target, person, face, skin_face, skin_broad)
        """
        cpath = self._cache_path(stem)

        # ---------- 命中缓存 ----------
        if cpath and os.path.exists(cpath):
            data = np.load(cpath)
            inp = data["input"]
            tgt = data["target"]
            person = data["person"]
            face = data["face"]

            if "skin_face" in data:
                skin_face = data["skin_face"]
            else:
                skin_face = self._compute_skin_face(face, inp).astype(np.float16)

            # ★ 新增：老缓存缺 skin_broad 时实时计算
            if "skin_broad" in data:
                skin_broad = data["skin_broad"]
            else:
                skin_broad = self._compute_skin_broad(inp).astype(np.float16)
                # 回写升级
                np.savez_compressed(
                    cpath,
                    input=data["input"], target=data["target"],
                    person=data["person"], face=data["face"],
                    skin_face=skin_face, skin_broad=skin_broad,
                )
            return inp, tgt, person, face, skin_face, skin_broad

        # ---------- 未命中：从头构建 ----------
        inp = _preprocess_image(inp_p, is_target=False, img_size=self.img_size,
                                cmyk_to_rgb_transform=self._cmyk_to_rgb)
        tgt = _preprocess_image(tgt_p, is_target=True, img_size=self.img_size,
                                cmyk_to_rgb_transform=self._cmyk_to_rgb)

        person = np.load(os.path.join(self.mask_dir, f"{stem}_person.npy")).astype(np.float32)
        face = np.load(os.path.join(self.mask_dir, f"{stem}_face.npy")).astype(np.float32)
        person = _center_crop_mask(person, self.img_size)
        face = _center_crop_mask(face, self.img_size)

        skin_face = _center_crop_mask(self._compute_skin_face(face, inp), self.img_size)
        # ★ 新增：宽色彩 mask
        skin_broad = self._compute_skin_broad(inp)

        if cpath:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.savez_compressed(
                cpath,
                input=inp.astype(np.float16),
                target=tgt.astype(np.float16),
                person=person.astype(np.float16),
                face=face.astype(np.float16),
                skin_face=skin_face.astype(np.float16),
                skin_broad=skin_broad.astype(np.float16),
            )
        return inp, tgt, person, face, skin_face, skin_broad

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_p, tgt_p = self.pairs[idx]
        stem = os.path.splitext(os.path.basename(inp_p))[0].replace("_input", "")

        inp, tgt, person, face, skin_face, skin_broad = \
            self._load_cached_or_build(stem, inp_p, tgt_p)

        return {
            "input":       torch.from_numpy(inp).permute(2, 0, 1).float(),
            "target":      torch.from_numpy(tgt).permute(2, 0, 1).float(),
            "person_mask": torch.from_numpy(person).unsqueeze(0).float(),
            "face_mask":   torch.from_numpy(face).unsqueeze(0).float(),
            "skin_face":   torch.from_numpy(skin_face).unsqueeze(0).float(),
            "skin_broad":  torch.from_numpy(skin_broad).unsqueeze(0).float(),  # ★ 新增
        }


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    pairs, tr, va = make_rgb_split(data_dir)
    print(f"配对总数: {len(pairs)}, train: {len(tr)}, val: {len(va)}")
