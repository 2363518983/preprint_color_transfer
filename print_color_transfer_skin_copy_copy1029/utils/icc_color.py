# -*- coding: utf-8 -*-
"""CMYK 四通道直读 + ICC 转换（渲染意图 & 黑点补偿，每次调用显式传入）。"""
import os
import hashlib
import warnings
import numpy as np
from PIL import Image, ImageCms


_cmyk_profile = None
_rgb_profile = None
_cmyk_icc_md5 = None
_cmyk_icc_path = None      # 新增：记录路径，供懒初始化复用
_intent_default = "relative"
_bpc_default = True

_INTENT_MAP = {
    "perceptual":  ImageCms.Intent.PERCEPTUAL,
    "relative":    ImageCms.Intent.RELATIVE_COLORIMETRIC,
    "saturation":  ImageCms.Intent.SATURATION,
    "absolute":    ImageCms.Intent.ABSOLUTE_COLORIMETRIC,
}


def _resolve_flags(bpc):
    flags = ImageCms.Flags.NONE
    if bpc:
        flags |= getattr(ImageCms.Flags, "BLACKPOINTCOMPENSATION", 0)
    return flags


def init_icc(cmyk_icc_path, srgb_icc_path=None, intent="relative", bpc=True):
    """加载 ICC，并打印 MD5 供与调图师 PS 配置核对。"""
    global _cmyk_profile, _rgb_profile, _cmyk_icc_md5, _cmyk_icc_path
    if not os.path.exists(cmyk_icc_path):
        raise FileNotFoundError(f"CMYK ICC 不存在: {cmyk_icc_path}")

    _cmyk_profile = ImageCms.ImageCmsProfile(open(cmyk_icc_path, "rb"))
    _cmyk_icc_md5 = hashlib.md5(open(cmyk_icc_path, "rb").read()).hexdigest()
    _cmyk_icc_path = cmyk_icc_path   # 记录路径

    if srgb_icc_path and os.path.exists(srgb_icc_path):
        _rgb_profile = ImageCms.ImageCmsProfile(open(srgb_icc_path, "rb"))
    else:
        _rgb_profile = ImageCms.createProfile("sRGB")

    print(f"[ICC] CMYK profile : {cmyk_icc_path}")
    print(f"[ICC] MD5          : {_cmyk_icc_md5}")
    print(f"[ICC] Intent       : {intent}  (per-call)")
    print(f"[ICC] BPC          : {bpc}  (per-call)")
    print(f"[ICC] Pillow ver   : PIL {Image.__version__}")


def get_cmyk_profile():
    """懒初始化：未显式 init_icc 时，尝试用默认路径自动加载（仅用于自检/调试）。"""
    global _cmyk_profile, _rgb_profile
    if _cmyk_profile is not None:
        return _cmyk_profile
    # 尝试默认路径（与 config 约定一致）
    candidate = _cmyk_icc_path or os.path.join("utils", "PSOcoated_v3.icc")
    if os.path.exists(candidate):
        warnings.warn(
            f"[ICC] 未显式调用 init_icc()，自动懒加载: {candidate}\n"
            f"       生产训练请务必在 train.py 中显式 init_icc() 以固定 Intent/BPC。",
            RuntimeWarning,
        )
        _cmyk_profile = ImageCms.ImageCmsProfile(open(candidate, "rb"))
        if _rgb_profile is None:
            _rgb_profile = ImageCms.createProfile("sRGB")
        return _cmyk_profile
    raise RuntimeError(
        "ICC 未初始化。请先调用 init_icc(cmyk_icc_path)，或在 utils/ 放置 PSOcoated_v3.icc。"
    )


def _intent_enum(intent):
    return _INTENT_MAP.get(intent, ImageCms.Intent.RELATIVE_COLORIMETRIC)


def read_rgb(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def read_cmyk_from_jpg(path):
    img = Image.open(path)
    if img.mode == "CMYK":
        return np.array(img, dtype=np.float32) / 255.0   # (H,W,4)
    return np.array(img.convert("RGB"), dtype=np.float32) / 255.0


def rgb_to_cmyk_icc(rgb_np, intent=None, bpc=None):
    """RGB (H,W,3) → CMYK (H,W,4)。intent/bpc 缺省时使用默认值。"""
    if get_cmyk_profile() is None:   # 触发懒初始化
        pass
    intent = intent or _intent_default
    bpc = _bpc_default if bpc is None else bpc
    if rgb_np.dtype != np.uint8:
        rgb_np = (np.clip(rgb_np, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(rgb_np, "RGB")
    out = ImageCms.profileToProfile(
        img, _rgb_profile, get_cmyk_profile(),
        renderingIntent=_intent_enum(intent),
        outputMode="CMYK",
        flags=_resolve_flags(bpc),
    )
    return np.clip(np.array(out, dtype=np.float32) / 255.0, 0, 1)


def cmyk_to_rgb_icc(cmyk_np, intent=None, bpc=None):
    """CMYK → RGB，用于 ΔE 评估。支持批量 (B,H,W,4)。"""
    if cmyk_np.ndim == 4:
        return np.stack([cmyk_to_rgb_icc(c, intent, bpc) for c in cmyk_np], axis=0)
    intent = intent or _intent_default
    bpc = _bpc_default if bpc is None else bpc
    cmyk_u8 = (np.clip(cmyk_np, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(cmyk_u8, "CMYK")
    out = ImageCms.profileToProfile(
        img, get_cmyk_profile(), _rgb_profile,
        renderingIntent=_intent_enum(intent),
        outputMode="RGB",
        flags=_resolve_flags(bpc),
    )
    return np.array(out, dtype=np.float32) / 255.0


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python icc_color.py <cmyk.icc> [some_input.jpg]")
        sys.exit(0)
    init_icc(sys.argv[1])
    if len(sys.argv) > 2:
        rgb = read_rgb(sys.argv[2])
        cmyk = rgb_to_cmyk_icc(rgb)
        back = cmyk_to_rgb_icc(cmyk)
        print("RGB", rgb.shape, "CMYK", cmyk.shape, "back RGB", back.shape)
        print("round-trip 均值差:", round(float(np.abs(rgb - back).mean()), 4))