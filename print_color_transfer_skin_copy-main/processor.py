# -*- coding: utf-8 -*-
"""processor.py — RGB 曲线推理后端（支持 CurvePredictor / Curve1D / HybridPipeline）。"""
import io
import os
import json
import shutil
import traceback
import zlib
import inspect
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageCms
import PIL.JpegImagePlugin
from pathlib import Path

Image.preinit()

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None


# ============================================================
# ★★★ 提高 pypdf 的安全上限（必须在 import pypdf 之前/之后尽早调用） ★★★
# ============================================================

def _raise_pypdf_limits():
    """印刷 PDF 的流经常超过 pypdf 默认 75MB，统一抬到 1.5GB。"""
    try:
        from pypdf._configuration import overwrite_configuration
        import pypdf  # noqa: F401
    except ImportError:
        return
    _GB = 1_500_000_000
    overwrite_configuration(
        maximum_declared_stream_length=_GB,
        array_based_stream_maximum_output_length=_GB,
        zlib_maximum_output_length=_GB,
        image_maximum_buffer_size=_GB,
        jbig2_maximum_output_length=_GB,
        lzw_maximum_output_length=_GB,
        run_length_maximum_output_length=_GB,
    )


_raise_pypdf_limits()

from config import Config
from models.curve_1d import Curve1D, apply_curves
from models.curve_predictor import CurvePredictor
from models.skin_curve_net import SkinCurveNet, apply_lut
from models.residual_unet import ResidualUNet


# ============================================================
# 基础工具
# ============================================================

def _smooth_residual(res, sigma=2.0):
    device = res.device
    ks = 5
    offset = torch.arange(ks, dtype=torch.float32, device=device) - (ks // 2)
    y, x = torch.meshgrid(offset, offset, indexing="ij")
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, ks, ks).repeat(res.shape[1], 1, 1, 1)
    return F.conv2d(res, kernel, padding=ks // 2, groups=res.shape[1])


def _skin_mask_batch(color_tensor):
    r = color_tensor[:, 0:1, :, :]
    g = color_tensor[:, 1:2, :, :]
    b = color_tensor[:, 2:3, :, :]
    mx, _ = torch.max(color_tensor, dim=1, keepdim=True)
    mn, _ = torch.min(color_tensor, dim=1, keepdim=True)
    sat = (mx - mn) / (mx + 1e-9)
    mask = (r > 0.35) & (g > 0.15) & (b > 0.05) & (r > g) & (g > b) & (sat > 0.15)
    return mask.float()


def _amplify_lut(curves, gain):
    if curves is None or gain == 1.0:
        return curves
    n = curves.shape[-1]
    identity = torch.linspace(0, 1, n, device=curves.device, dtype=curves.dtype)
    identity = identity.view(1, 1, -1)
    return (identity + (curves - identity) * gain).clamp(0, 1)


# ============================================================
# 分块推理参数
# ============================================================

TILE_TRIGGER_PIXELS = 4000 * 4000
TILE_SIZE = 2048
TILE_OVERLAP = 256


def _tile_ranges(total, tile_size, overlap):
    if total <= tile_size:
        return [(0, total)]
    step = tile_size - overlap
    ranges = []
    pos = 0
    while pos < total:
        end = min(pos + tile_size, total)
        ranges.append((pos, end))
        if end >= total:
            break
        pos += step
    return ranges


def _edge_ramp(n, overlap):
    if overlap <= 0 or n <= 2 * overlap:
        return np.ones(n, dtype=np.float32)
    w = np.ones(n, dtype=np.float32)
    ramp = np.linspace(0, 1, overlap, endpoint=False, dtype=np.float32)
    w[:overlap] = ramp
    w[-overlap:] = ramp[::-1]
    return w


# ============================================================
# 混合推理
# ============================================================

class HybridPipeline:
    def __init__(self, curve_model, skin_model, unet_model, device):
        self.curve_model = curve_model
        self.skin_model = skin_model
        self.unet_model = unet_model
        self.device = device

    def eval(self):
        for m in (self.curve_model, self.skin_model, self.unet_model):
            if m is not None:
                m.eval()
        return self


@torch.no_grad()
def _generate_hybrid_stages(img_np, pipeline,
                            curve_gain=1.0, skin_gain=1.0, residual_gain=1.0,
                            use_amp=False):
    device = pipeline.device
    t = None
    base1_t = base2_t = out_t = None
    try:
        t = torch.from_numpy(np.ascontiguousarray(img_np)).permute(2, 0, 1).unsqueeze(0).float().to(device)

        size = Config.PRED_INPUT_SIZE
        resized = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        curves1 = pipeline.curve_model.forward(resized)
        del resized
        if curves1 is not None:
            curves1 = _amplify_lut(curves1, curve_gain)
            base1_t = apply_curves(t, curves1)
            del curves1
        else:
            base1_t, _ = pipeline.curve_model.apply(t, resize_size=size)

        if pipeline.skin_model is not None:
            mask = _skin_mask_batch(base1_t)
            curves2 = pipeline.skin_model(base1_t, mask=mask)
            del mask
            curves2 = _amplify_lut(curves2, skin_gain)
            base2_t = apply_lut(base1_t, curves2)
            del curves2
        else:
            base2_t = base1_t

        if pipeline.unet_model is not None:
            amp_enabled = use_amp and device.type == "cuda"
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                inp_small = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
                base_small = F.interpolate(base2_t, size=(256, 256), mode="bilinear", align_corners=False)
                cat_in = torch.cat([inp_small, base_small], dim=1)
                del inp_small, base_small
                residual = pipeline.unet_model(cat_in)
                del cat_in
            residual = residual.float()
            residual = _smooth_residual(residual) * residual_gain
            residual = F.interpolate(residual, size=t.shape[-2:], mode="bilinear", align_corners=False)
            out_t = (base2_t + residual).clamp(0, 1)
            del residual
        else:
            out_t = base2_t

        s1 = base1_t.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        s2 = base2_t.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_np = out_t.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return s1, s2, out_np
    finally:
        del t, base1_t, base2_t, out_t
        if device.type == "cuda":
            torch.cuda.empty_cache()


@torch.no_grad()
def _generate_hybrid_stages_tiled(img_np, pipeline,
                                  curve_gain=1.0, skin_gain=1.0, residual_gain=1.0,
                                  use_amp=False,
                                  tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    device = pipeline.device
    h, w = img_np.shape[:2]

    full_t = torch.from_numpy(np.ascontiguousarray(img_np)).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # 阶段 1：整图预测全局曲线
    size = Config.PRED_INPUT_SIZE
    resized = F.interpolate(full_t, size=(size, size), mode="bilinear", align_corners=False)
    curves1 = pipeline.curve_model.forward(resized)
    del resized
    if curves1 is not None:
        curves1 = _amplify_lut(curves1, curve_gain)
        base1_t = apply_curves(full_t, curves1)
        del curves1
    else:
        base1_t, _ = pipeline.curve_model.apply(full_t, resize_size=size)

    # 阶段 2：整图肤色 mask + 肤色曲线
    if pipeline.skin_model is not None:
        mask = _skin_mask_batch(base1_t)
        curves2 = pipeline.skin_model(base1_t, mask=mask)
        del mask
        curves2 = _amplify_lut(curves2, skin_gain)
        base2_t = apply_lut(base1_t, curves2)
        del curves2
    else:
        base2_t = base1_t

    if pipeline.unet_model is None:
        s1 = base1_t.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        s2 = base2_t.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        del full_t, base1_t, base2_t
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return s1, s2, s2

    # 阶段 3：U-Net 残差分块
    amp_enabled = use_amp and device.type == "cuda"
    yr = _tile_ranges(h, tile_size, overlap)
    xr = _tile_ranges(w, tile_size, overlap)

    acc_res = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)

    try:
        for y0, y1 in yr:
            for x0, x1 in xr:
                t_tile = full_t[:, :, y0:y1, x0:x1]
                b_tile = base2_t[:, :, y0:y1, x0:x1]
                with torch.autocast(device_type="cuda", enabled=amp_enabled):
                    inp_small = F.interpolate(t_tile, size=(256, 256), mode="bilinear", align_corners=False)
                    b_small = F.interpolate(b_tile, size=(256, 256), mode="bilinear", align_corners=False)
                    cat_in = torch.cat([inp_small, b_small], dim=1)
                    del inp_small, b_small
                    res = pipeline.unet_model(cat_in)
                    del cat_in
                res = res.float()
                res = _smooth_residual(res) * residual_gain
                th, tw = y1 - y0, x1 - x0
                res = F.interpolate(res, size=(th, tw), mode="bilinear", align_corners=False)
                res_np = res.squeeze(0).permute(1, 2, 0).cpu().numpy()

                wy = _edge_ramp(th, overlap)
                wx = _edge_ramp(tw, overlap)
                w_tile = (wy[:, None] * wx[None, :])[:, :, None]

                acc_res[y0:y1, x0:x1] += res_np * w_tile
                weight[y0:y1, x0:x1] += w_tile

                del t_tile, b_tile, res, res_np, w_tile, wy, wx
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        weight = np.maximum(weight, 1e-6)
        residual_map = acc_res / weight

        s1 = base1_t.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        s2 = base2_t.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_np = np.clip(s2 + residual_map, 0, 1)
        return s1, s2, out_np
    finally:
        del full_t, base1_t, base2_t, acc_res, weight
        if device.type == "cuda":
            torch.cuda.empty_cache()


def process_with_stages(img_np, model, gains=None):
    gains = gains or {}
    if isinstance(model, HybridPipeline):
        h, w = img_np.shape[:2]
        gains_kw = dict(
            curve_gain=gains.get("curve", 1.0),
            skin_gain=gains.get("skin", 1.0),
            residual_gain=gains.get("residual", 1.0),
        )
        if h * w > TILE_TRIGGER_PIXELS:
            print(f"   [分块推理] {w}×{h} 超过阈值，按 {TILE_SIZE}px 分块处理")
            stage1, stage2, final = _generate_hybrid_stages_tiled(img_np, model, **gains_kw)
        else:
            stage1, stage2, final = _generate_hybrid_stages(img_np, model, **gains_kw)
        return final, find_content_boxes(img_np), stage1, stage2

    boxes = find_content_boxes(img_np)
    final, _ = _apply_curve_content(img_np, model, gains=gains)
    return final, boxes, None, None


def _save_stage_previews(base1, base2, page_index, stage_dir, stem="page"):
    if stage_dir is None:
        return
    os.makedirs(stage_dir, exist_ok=True)
    for name, arr in (("stage1", base1), ("stage2", base2)):
        if arr is None:
            continue
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        img.save(os.path.join(stage_dir, f"{stem}_{name}_{page_index}.jpg"),
                 quality=92, optimize=True)
        img.close()


# ============================================================
# 模型加载
# ============================================================

def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ('state_dict', 'model_state', 'model', 'net'):
            if key in ckpt and isinstance(ckpt[key], dict):
                if any(k.endswith('.weight') or k.endswith('.bias') for k in ckpt[key]):
                    return ckpt[key]
        if any(k.endswith('.weight') or k.endswith('.bias') for k in ckpt):
            return ckpt
    return ckpt


def _make_model(model_cls, **preferred_kwargs):
    sig = inspect.signature(model_cls.__init__)
    params = sig.parameters
    supported = {k: v for k, v in preferred_kwargs.items() if k in params}
    if 'input_channels' in params and 'input_channels' not in supported:
        supported['input_channels'] = preferred_kwargs.get('channels', 3)
    return model_cls(**supported)


def _load_skin_curve(path, device):
    if not os.path.exists(path):
        print("⚠️ 未找到肤色曲线模型，跳过肤色修正")
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = _extract_state_dict(ckpt)
    if 'fc.2.weight' not in state:
        print("⚠️ 肤色曲线模型结构异常，跳过肤色修正")
        return None
    fc_out = state['fc.2.weight'].shape[0]
    channels = 3
    n_bins = fc_out // channels
    model = _make_model(SkinCurveNet, channels=channels, n_bins=n_bins).to(device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"✅ 肤色曲线模型加载成功: {path} (输出 {channels}×{n_bins})")
    return model


def _load_unet(path, device):
    if not os.path.exists(path):
        print("⚠️ 未找到残差 U-Net 模型，仅使用曲线链路")
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = _extract_state_dict(ckpt)
    model = _make_model(ResidualUNet, in_channels=6, out_channels=3).to(device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"✅ 残差 U-Net 模型加载成功: {path}")
    return model


def _is_curve_predictor_ckpt(obj):
    if isinstance(obj, dict):
        keys = obj.keys()
        if any(("encoder" in k or "fc_delta" in k or "fc_w" in k or "fc1" in k) for k in keys):
            return True
        for inner_key in ("state_dict", "model"):
            if inner_key in obj and isinstance(obj[inner_key], dict):
                return _is_curve_predictor_ckpt(obj[inner_key])
    return False


def _load_global_curve(cfg):
    for path in (cfg.GLOBAL_CURVE_TENSOR_PATH, cfg.GLOBAL_CURVE_PATH):
        if not os.path.exists(path):
            continue
        obj = torch.load(path, map_location="cpu", weights_only=False)
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
    print("[全局曲线] 未找到，模型退化为纯预测")
    return None


def load_model(model_path: str, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[模型] 使用设备 {device}")

    cfg = Config()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    obj = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        for inner_key in ("state_dict", "model"):
            if inner_key in obj and isinstance(obj[inner_key], dict):
                obj = obj[inner_key]
                break

    if _is_curve_predictor_ckpt(obj):
        curve_model = CurvePredictor(channels=3, n_bins=cfg.LUT_DIM, base=cfg.PRED_BASE).to(device)
        curve_model.load_state_dict(obj, strict=False)
        gcurve = _load_global_curve(cfg)
        if gcurve is not None:
            if gcurve.shape[-1] != cfg.LUT_DIM:
                raise ValueError(f"全局曲线 bin 数 {gcurve.shape[-1]} != 配置 {cfg.LUT_DIM}")
            curve_model.set_global_curve(gcurve)
        else:
            raise RuntimeError(
                "未找到全局曲线先验。请将 GLOBAL_CURVE_TENSOR_PATH 指向正确的 .pt 文件，"
                "或在 config.py 中关闭此检查。")
        for p in curve_model.parameters():
            p.requires_grad_(False)

        checkpoint_dir = Path(model_path).parent
        skin_model = _load_skin_curve(checkpoint_dir / "skin_curve_best.pth", device)
        unet_model = _load_unet(checkpoint_dir / "unet_best.pth", device)
        print(f"✅ CurvePredictor 加载成功: {model_path}")
        if skin_model is not None or unet_model is not None:
            print(f"🎯 启用混合推理管线：全局曲线 + "
                  f"{'肤色曲线 + ' if skin_model is not None else ''}"
                  f"{'残差U-Net' if unet_model is not None else '纯曲线'}")
            model = HybridPipeline(curve_model, skin_model, unet_model, device)
        else:
            model = curve_model
    else:
        model = Curve1D(channels=3, n_bins=cfg.LUT_DIM).to(device)
        model.load_state_dict(obj)
        print(f"✅ Curve1D 加载成功: {model_path}")

    model.eval()
    return model


# ============================================================
# ICC 转换
# ============================================================

_CMYK_ICC = os.path.join("utils", "PSOcoated_v3.icc")
_INTENT = ImageCms.Intent.RELATIVE_COLORIMETRIC
_FLAGS = ImageCms.Flags.BLACKPOINTCOMPENSATION

_srgb_profile = None
_cmyk_profile = None
_rgb_to_cmyk = None
_cmyk_to_rgb = None


def _init_icc(cmyk_icc_path=_CMYK_ICC):
    global _srgb_profile, _cmyk_profile, _rgb_to_cmyk, _cmyk_to_rgb
    if _rgb_to_cmyk is not None:
        return
    if not os.path.exists(cmyk_icc_path):
        raise FileNotFoundError(f"CMYK ICC 不存在: {cmyk_icc_path}")
    _srgb_profile = ImageCms.createProfile("sRGB")
    _cmyk_profile = ImageCms.getOpenProfile(cmyk_icc_path)
    _rgb_to_cmyk = ImageCms.buildTransformFromOpenProfiles(
        _srgb_profile, _cmyk_profile, "RGB", "CMYK", _INTENT, _FLAGS)
    _cmyk_to_rgb = ImageCms.buildTransformFromOpenProfiles(
        _cmyk_profile, _srgb_profile, "CMYK", "RGB", _INTENT, _FLAGS)


def _to_cmyk_pil(rgb_pil):
    _init_icc()
    return ImageCms.applyTransform(rgb_pil, _rgb_to_cmyk)


def _icc_bytes():
    if os.path.exists(_CMYK_ICC):
        with open(_CMYK_ICC, "rb") as f:
            return f.read()
    return None


def cmyk_to_srgb(cmyk_pil):
    if cmyk_pil.mode != "CMYK":
        return cmyk_pil.convert("RGB")
    _init_icc()
    return ImageCms.applyTransform(cmyk_pil, _cmyk_to_rgb)


def save_cmyk_image(cmyk_pil, path, fmt=None, lossless=False):
    path = str(path)
    fmt = (fmt or format_from_suffix(Path(path).suffix)).upper()
    icc = _icc_bytes()
    kwargs = {"icc_profile": icc} if icc else {}
    if fmt == "JPEG":
        cmyk_pil.save(path, format="JPEG", quality=95, subsampling=0, **kwargs)
    elif fmt == "TIFF" and lossless:
        cmyk_pil.save(path, format="TIFF", compression="tiff_lzw", **kwargs)
    else:
        cmyk_pil.save(path, **kwargs)
    return path


def _save_cmyk(rgb_pil, path, fmt=None):
    return save_cmyk_image(_to_cmyk_pil(rgb_pil), path, fmt=fmt)


# ============================================================
# 内容框检测
# ============================================================

CONTENT_FULLPAGE_RATIO = 0.88
CONTENT_MIN_AREA_RATIO = 0.008
CONTENT_PAD_RATIO = 0.008


def paper_mask_rgb(rgb):
    x = np.asarray(rgb, dtype=np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    mx = x.max(axis=2)
    mn = x.min(axis=2)
    return (mx >= 0.93) & ((mx - mn) <= 0.08)


def _flag_spans(flag):
    spans, in_run, start = [], False, 0
    for i, value in enumerate(flag):
        if value and not in_run:
            start, in_run = i, True
        elif not value and in_run:
            spans.append((start, i))
            in_run = False
    if in_run:
        spans.append((start, len(flag)))
    return spans


def _dilate_1d(flag, radius):
    out = np.asarray(flag, dtype=bool).copy()
    for shift in range(1, radius + 1):
        out[:-shift] |= flag[shift:]
        out[shift:] |= flag[:-shift]
    return out


def _boxes_from_mask(content, min_area):
    h, w = content.shape
    gap = max(3, min(25, int(min(h, w) * 0.012)))
    boxes = []
    for y0, y1 in _flag_spans(_dilate_1d(content.any(axis=1), gap)):
        band = content[y0:y1]
        for x0, x1 in _flag_spans(_dilate_1d(band.any(axis=0), gap)):
            sub = content[y0:y1, x0:x1]
            if not sub.any():
                continue
            ys, xs = np.where(sub)
            xx0, yy0 = x0 + int(xs.min()), y0 + int(ys.min())
            xx1, yy1 = x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1
            if (xx1 - xx0) * (yy1 - yy0) >= min_area:
                boxes.append((xx0, yy0, xx1, yy1))
    return boxes


def find_content_boxes(rgb, min_area_ratio=CONTENT_MIN_AREA_RATIO,
                       fullpage_ratio=CONTENT_FULLPAGE_RATIO,
                       pad_ratio=CONTENT_PAD_RATIO):
    h, w = rgb.shape[:2]
    content = ~paper_mask_rgb(rgb)
    fill = float(content.mean())
    if fill >= fullpage_ratio or fill < 0.002:
        return [(0, 0, w, h)]
    min_area = h * w * min_area_ratio
    boxes = _boxes_from_mask(content, min_area)
    if not boxes:
        return [(0, 0, w, h)]
    out = []
    for x0, y0, x1, y1 in boxes:
        if pad_ratio <= 0:
            pad_x = pad_y = 0
        else:
            pad_x = max(2, int((x1 - x0) * pad_ratio))
            pad_y = max(2, int((y1 - y0) * pad_ratio))
        x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
        x1, y1 = min(w, x1 + pad_x), min(h, y1 + pad_y)
        if (x1 - x0) * (y1 - y0) >= h * w * fullpage_ratio:
            return [(0, 0, w, h)]
        out.append((x0, y0, x1, y1))
    return out or [(0, 0, w, h)]


# ============================================================
# 曲线推理
# ============================================================

def _image_to_tensor(img_np, device):
    return torch.from_numpy(np.ascontiguousarray(img_np)).permute(2, 0, 1).unsqueeze(0).float().to(device)


@torch.no_grad()
def _apply_curve(img_np, model):
    if isinstance(model, HybridPipeline):
        _, _, out = _generate_hybrid_stages(img_np, model)
        return out
    device = next(model.parameters()).device
    t = _image_to_tensor(img_np, device)
    try:
        if isinstance(model, CurvePredictor):
            out, _ = model.apply(t, resize_size=Config.PRED_INPUT_SIZE)
        else:
            out = model(t)
        out = out.clamp(0, 1)
        return out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    finally:
        del t
        if device.type == "cuda":
            torch.cuda.empty_cache()


@torch.no_grad()
def _predict_curves(img_np, model):
    device = next(model.parameters()).device
    t = _image_to_tensor(img_np, device)
    try:
        if isinstance(model, CurvePredictor):
            size = Config.PRED_INPUT_SIZE
            resized = torch.nn.functional.interpolate(
                t, size=(size, size), mode="bilinear", align_corners=False)
            try:
                return model.forward(resized)
            finally:
                del resized
        return None
    finally:
        del t


@torch.no_grad()
def _apply_predicted_curves(img_np, model, curves):
    device = next(model.parameters()).device
    t = _image_to_tensor(img_np, device)
    try:
        out = model(t) if curves is None else apply_curves(t, curves)
        return out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    finally:
        del t
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _apply_curve_content(img_np, model, gains=None):
    gains = gains or {}
    if isinstance(model, HybridPipeline):
        h, w = img_np.shape[:2]
        gains_kw = dict(
            curve_gain=gains.get("curve", 1.0),
            skin_gain=gains.get("skin", 1.0),
            residual_gain=gains.get("residual", 1.0),
        )
        if h * w > TILE_TRIGGER_PIXELS:
            print(f"   [分块推理] {w}×{h} 超过阈值，按 {TILE_SIZE}px 分块处理")
            _s1, _s2, out = _generate_hybrid_stages_tiled(img_np, model, **gains_kw)
        else:
            _s1, _s2, out = _generate_hybrid_stages(img_np, model, **gains_kw)
        return out, find_content_boxes(img_np)

    h, w = img_np.shape[:2]
    boxes = find_content_boxes(img_np)
    if boxes == [(0, 0, w, h)]:
        return _apply_curve(img_np, model), boxes
    tight_boxes = find_content_boxes(img_np, pad_ratio=0)
    pairs = (list(zip(tight_boxes, boxes)) if len(tight_boxes) == len(boxes)
             else [(box, box) for box in boxes])
    out = img_np.copy()
    for (tx0, ty0, tx1, ty1), (x0, y0, x1, y1) in pairs:
        pred_src = img_np[ty0:ty1, tx0:tx1]
        crop = img_np[y0:y1, x0:x1]
        if pred_src.size == 0 or crop.size == 0:
            continue
        curves = _predict_curves(pred_src, model)
        out[y0:y1, x0:x1] = _apply_predicted_curves(crop, model, curves)
    return out, boxes


# ============================================================
# CMYK 手工曲线
# ============================================================

CMYK_CHANNELS = ("C", "M", "Y", "K")
CURVE_KEYS = ("master",) + CMYK_CHANNELS
IDENTITY_POINTS = [[0, 0], [255, 255]]
_IDENTITY_LUT = np.arange(256, dtype=np.uint8)


def _clean_points(points):
    pts = {}
    for p in points or []:
        try:
            x, y = int(round(float(p[0]))), int(round(float(p[1])))
        except (TypeError, ValueError, IndexError):
            continue
        pts[max(0, min(255, x))] = max(0, min(255, y))
    if not pts:
        return [(0, 0), (255, 255)]
    items = sorted(pts.items())
    if items[0][0] != 0:
        items.insert(0, (0, items[0][1]))
    if items[-1][0] != 255:
        items.append((255, items[-1][1]))
    return items


def _monotone_cubic(xs, ys, grid):
    n = len(xs)
    h = np.diff(xs)
    delta = np.diff(ys) / h
    m = np.empty(n)
    m[0], m[-1] = delta[0], delta[-1]
    for i in range(1, n - 1):
        m[i] = 0.0 if delta[i - 1] * delta[i] <= 0 else (delta[i - 1] + delta[i]) / 2.0
    for i in range(n - 1):
        if delta[i] == 0:
            m[i] = m[i + 1] = 0.0
            continue
        a, b = m[i] / delta[i], m[i + 1] / delta[i]
        s = a * a + b * b
        if s > 9.0:
            t = 3.0 / np.sqrt(s)
            m[i], m[i + 1] = t * a * delta[i], t * b * delta[i]
    idx = np.clip(np.searchsorted(xs, grid, side="right") - 1, 0, n - 2)
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    m0, m1 = m[idx], m[idx + 1]
    hh = x1 - x0
    t = (grid - x0) / hh
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * hh * m0
            + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * hh * m1)


def curve_points_to_lut(points):
    items = _clean_points(points)
    xs = np.array([p[0] for p in items], dtype=np.float64)
    ys = np.array([p[1] for p in items], dtype=np.float64)
    grid = np.arange(256, dtype=np.float64)
    vals = np.interp(grid, xs, ys) if len(xs) < 3 else _monotone_cubic(xs, ys, grid)
    return np.clip(np.rint(vals), 0, 255).astype(np.uint8)


def normalize_curves(curves):
    curves = curves if isinstance(curves, dict) else {}
    return {k: [list(p) for p in _clean_points(curves.get(k))] for k in CURVE_KEYS}


def build_cmyk_luts(curves):
    normalized = normalize_curves(curves)
    master = curve_points_to_lut(normalized["master"])
    return {ch: curve_points_to_lut(normalized[ch])[master] for ch in CMYK_CHANNELS}


def curves_are_identity(curves):
    luts = build_cmyk_luts(curves)
    return all(np.array_equal(luts[ch], _IDENTITY_LUT) for ch in CMYK_CHANNELS)


def apply_cmyk_curves(cmyk_pil, curves, content_only=True):
    if cmyk_pil.mode != "CMYK":
        cmyk_pil = _to_cmyk_pil(cmyk_pil.convert("RGB"))
    if curves_are_identity(curves):
        return cmyk_pil
    luts = build_cmyk_luts(curves)
    w, h = cmyk_pil.size
    boxes = [(0, 0, w, h)]
    if content_only:
        boxes = find_content_boxes(np.asarray(cmyk_to_srgb(cmyk_pil)))
    if boxes == [(0, 0, w, h)]:
        bands = [band.point(luts[ch].tolist())
                 for band, ch in zip(cmyk_pil.split(), CMYK_CHANNELS)]
        return Image.merge("CMYK", bands)
    result = cmyk_pil.copy()
    for x0, y0, x1, y1 in boxes:
        crop = cmyk_pil.crop((x0, y0, x1, y1))
        bands = [band.point(luts[ch].tolist())
                 for band, ch in zip(crop.split(), CMYK_CHANNELS)]
        result.paste(Image.merge("CMYK", bands), (x0, y0))
    return result


# ============================================================
# 格式与 PDF
# ============================================================

def format_from_suffix(suffix: str) -> str:
    s = (suffix or "").lower()
    if s in {".jpg", ".jpeg"}:
        return "JPEG"
    if s == ".png":
        return "PNG"
    if s in {".tif", ".tiff"}:
        return "TIFF"
    if s == ".pdf":
        return "PDF"
    raise ValueError(f"不支持的图像格式: {suffix}")


def cmyk_container_name(name: str) -> str:
    path = Path(name)
    if path.suffix.lower() == ".png":
        return str(path.with_suffix(".tif"))
    return name


PDF_DPI = 300
PDF_MAX_PAGES = 60
PDF_MAX_DETECT_DPI = 9600


def is_pdf(path) -> bool:
    p = Path(str(path))
    if p.suffix.lower() == ".pdf":
        return True
    try:
        with open(p, "rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def _require_pdfium():
    if pdfium is None:
        raise ValueError("未安装 pypdfium2，无法处理 PDF。请执行 pip install pypdfium2")


def pdf_page_count(path) -> int:
    _require_pdfium()
    doc = pdfium.PdfDocument(str(path))
    try:
        return len(doc)
    finally:
        doc.close()


def _pdf_embedded_dpi(path):
    doc = pdfium.PdfDocument(str(path))
    try:
        samples = []
        for index in range(len(doc)):
            page = doc[index]
            page_w, page_h = page.get_size()
            page_area = max(page_w * page_h, 1.0)
            for obj in page.get_objects(max_depth=15):
                if not isinstance(obj, pdfium.PdfImage):
                    continue
                px_w, px_h = obj.get_px_size()
                left, bottom, right, top = obj.get_bounds()
                disp_w, disp_h = abs(right - left), abs(top - bottom)
                if px_w < 8 or px_h < 8 or disp_w < 1 or disp_h < 1:
                    continue
                dpi = max(px_w * 72.0 / disp_w, px_h * 72.0 / disp_h)
                if dpi < 36 or dpi > PDF_MAX_DETECT_DPI:
                    continue
                samples.append(((disp_w * disp_h) / page_area, dpi))
        if not samples:
            return None
        significant = [dpi for cover, dpi in samples if cover >= 0.05]
        pool = significant or [dpi for _, dpi in samples]
        return float(round(max(pool)))
    finally:
        doc.close()


def pdf_render_dpi(path, dpi=PDF_DPI):
    _require_pdfium()
    doc = pdfium.PdfDocument(str(path))
    try:
        total = len(doc)
        if total == 0:
            raise ValueError("PDF 不包含任何页面")
        if total > PDF_MAX_PAGES:
            raise ValueError(f"PDF 页数 {total} 超过上限 {PDF_MAX_PAGES}")
    finally:
        doc.close()
    detected = _pdf_embedded_dpi(path)
    chosen = float(detected or dpi or PDF_DPI)
    print(f"[PDF] 栅格化 {chosen:.0f} dpi（{'输入图精度' if detected else '默认'}）")
    return chosen


def iter_pdf_pages(path, dpi=None):
    _require_pdfium()
    dpi = pdf_render_dpi(path) if dpi is None else float(dpi)
    scale = dpi / 72.0
    doc = pdfium.PdfDocument(str(path))
    try:
        for index in range(len(doc)):
            page = doc[index]
            width_pt, height_pt = page.get_size()
            rendered = page.render(scale=scale).to_pil().convert("RGB")
            yield rendered, (width_pt, height_pt), dpi
    finally:
        doc.close()


def iter_source_pages(path, dpi=None):
    if is_pdf(path):
        yield from iter_pdf_pages(path, dpi=dpi)
        return
    _init_icc()
    with Image.open(path) as img:
        img.load()
        rgb = ImageCms.applyTransform(img, _cmyk_to_rgb) if img.mode == "CMYK" else img.convert("RGB")
        info_dpi = img.info.get("dpi")
    page_dpi = float(dpi or PDF_DPI)
    try:
        if info_dpi and float(info_dpi[0]) > 0:
            page_dpi = float(info_dpi[0])
    except (TypeError, ValueError, IndexError):
        pass
    size_pt = (rgb.width / page_dpi * 72.0, rgb.height / page_dpi * 72.0)
    yield rgb, size_pt, page_dpi


class _PdfWriter:
    def __init__(self):
        self.objects = {1: None, 2: None}

    def add(self, body: bytes) -> int:
        oid = max(self.objects) + 1
        self.objects[oid] = body
        return oid

    def add_stream(self, data: bytes, extra: str) -> int:
        header = f"<< {extra} /Length {len(data)} >>\nstream\n".encode("latin-1")
        return self.add(header + data + b"\nendstream")

    def write(self, path):
        with open(path, "wb") as f:
            f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
            offsets = {}
            for oid in range(1, max(self.objects) + 1):
                offsets[oid] = f.tell()
                body = self.objects[oid]
                if not body.endswith(b"\n"):
                    body += b"\n"
                f.write(f"{oid} 0 obj\n".encode("ascii") + body + b"endobj\n")
            xref = f.tell()
            count = max(self.objects) + 1
            f.write(f"xref\n0 {count}\n".encode("ascii"))
            f.write(b"0000000000 65535 f \n")
            for oid in range(1, count):
                f.write(f"{offsets[oid]:010d} 00000 n \n".encode("ascii"))
            f.write(
                f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
                .encode("ascii"))


def save_cmyk_pdf(page_paths, path, dpi=PDF_DPI, page_sizes_pt=None):
    if not page_paths:
        raise ValueError("没有可写入 PDF 的页面")
    writer = _PdfWriter()
    icc = _icc_bytes()
    icc_id = writer.add_stream(icc, "/N 4 /Alternate /DeviceCMYK") if icc else None
    colorspace = f"[/ICCBased {icc_id} 0 R]" if icc_id else "/DeviceCMYK"
    page_ids = []
    dpi = float(dpi or PDF_DPI)
    for index, page_path in enumerate(page_paths):
        with Image.open(str(page_path)) as img:
            img.load()
            if img.mode != "CMYK":
                img = _to_cmyk_pil(img.convert("RGB"))
            width, height = img.size
            raw = img.tobytes()
        packed = zlib.compress(raw, 9)
        if page_sizes_pt and index < len(page_sizes_pt) and page_sizes_pt[index]:
            page_w, page_h = float(page_sizes_pt[index][0]), float(page_sizes_pt[index][1])
        else:
            page_w, page_h = width * 72.0 / dpi, height * 72.0 / dpi
        image_id = writer.add_stream(
            packed,
            f"/Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace {colorspace} /BitsPerComponent 8 /Filter /FlateDecode",
        )
        contents_id = writer.add_stream(
            f"q\n{page_w:.4f} 0 0 {page_h:.4f} 0 0 cm\n/Im0 Do\nQ\n".encode("ascii"),
            "",
        )
        page_ids.append(writer.add(
            (f"<< /Type /Page /Parent 2 0 R "
             f"/MediaBox [0 0 {page_w:.4f} {page_h:.4f}] "
             f"/Resources << /XObject << /Im0 {image_id} 0 R >> >> "
             f"/Contents {contents_id} 0 R >>").encode("ascii")
        ))
    kids = " ".join(f"{oid} 0 R" for oid in page_ids)
    writer.objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    writer.objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
    writer.write(str(path))
    return str(path)


def _require_pypdf():
    try:
        import pypdf  # noqa: F401
    except ImportError as exc:
        raise ValueError("未安装 pypdf，无法保留 PDF 分层。请执行 pip install pypdf") from exc
    _raise_pypdf_limits()


class _SimpleImageRef:
    """轻量 image 适配器，只暴露 pypdf image 对象里我们用到的接口。
    ★ 关键：不触发 pypdf 的解码和 MAX_IMAGE_BUFFER_SIZE 检查。
    """
    __slots__ = ("indirect_reference", "_obj")

    def __init__(self, ref):
        self.indirect_reference = ref
        self._obj = ref.get_object()

    @property
    def is_inline(self):
        return False

    def get(self, key, default=None):
        try:
            return self._obj.get(key, default)
        except Exception:
            return default

    def __getitem__(self, key):
        return self._obj[key]

    @property
    def image(self):
        """★ 兼容非 raw_cmyk 路径的兜底访问（会触发 pypdf 解码）。
        我们已经提高了上限，多数情况能通过；不行还有 _read_embedded_image_raw。"""
        try:
            from pypdf.generic._image_xobject import _xobj_to_image
            ext, data, img = _xobj_to_image(self._obj)
            return img
        except Exception as exc:
            print(f"   [PDF] image 解码失败: {exc}")
            return None

    def __repr__(self):
        try:
            w = int(self._obj.get("/Width", 0) or 0)
            h = int(self._obj.get("/Height", 0) or 0)
        except Exception:
            w = h = 0
        return f"<ImageRef {w}x{h}>"


def _page_xobjects(page):
    """从 page 的 /Resources /XObject 里拿 (name, IndirectObject) 列表。
    不解码图像，只枚举键值。"""
    from pypdf.generic import IndirectObject
    try:
        resources = page.get("/Resources")
    except Exception:
        return []
    if resources is None:
        return []
    try:
        if isinstance(resources, IndirectObject):
            resources = resources.get_object()
    except Exception:
        return []
    if not hasattr(resources, "get"):
        return []
    try:
        xobj = resources.get("/XObject")
    except Exception:
        return []
    if xobj is None:
        return []
    try:
        if isinstance(xobj, IndirectObject):
            xobj = xobj.get_object()
    except Exception:
        return []
    result = []
    try:
        items = xobj.items() if hasattr(xobj, "items") else []
        for name, value in items:
            if isinstance(value, IndirectObject):
                result.append((name, value))
    except Exception:
        pass
    return result

def _collect_writer_images(writer):
    """★ 绕过 page.images：直接遍历每页的 XObject 表拿图像引用。
    不会触发 pypdf 的图像解码，因此不会撞 MAX_IMAGE_BUFFER_SIZE 限制。"""
    images = []
    seen = set()
    total_pages = 0
    for page in writer.pages:
        total_pages += 1
        try:
            xobjs = _page_xobjects(page)
        except Exception as exc:
            print(f"   [PDF] 第 {total_pages} 页 XObject 获取失败: {exc}")
            continue
        for name, ref in xobjs:
            try:
                obj = ref.get_object()
                subtype = obj.get("/Subtype")
                if subtype is None or str(subtype) != "/Image":
                    continue
                if obj.get("/ImageMask"):
                    continue
                ident = getattr(ref, "idnum", None) or id(ref)
                if ident in seen:
                    continue
                seen.add(ident)
                images.append(_SimpleImageRef(ref))
            except Exception as exc:
                print(f"   [PDF] 跳过图像对象 {name}: {exc}")
                continue
    print(f"   [PDF] 扫描 {total_pages} 页，找到 {len(images)} 张嵌入图")
    return images

_icc_to_srgb_cache = {}


def _transform_icc_to_srgb(icc_bytes, in_mode):
    """用图像自带 ICC 建到 sRGB 的转换，同一份配置文件只建一次。"""
    _init_icc()
    key = (icc_bytes[:64], len(icc_bytes), icc_bytes[-64:], in_mode)
    cached = _icc_to_srgb_cache.get(key)
    if cached is not None:
        return cached
    profile = ImageCms.getOpenProfile(io.BytesIO(icc_bytes))
    transform = ImageCms.buildTransformFromOpenProfiles(
        profile, _srgb_profile, in_mode, "RGB", _INTENT, _FLAGS)
    _icc_to_srgb_cache[key] = transform
    return transform


def _pil_icc_bytes(pil):
    icc = pil.info.get("icc_profile") if getattr(pil, "info", None) else None
    if isinstance(icc, str):
        icc = icc.encode("latin-1")
    return bytes(icc) if icc else None


def _pil_to_model_rgb(pil):
    """送到模型前的 sRGB。有自带 ICC 就用它，否则 CMYK 用 PSOcoated_v3。"""
    icc = _pil_icc_bytes(pil)
    mode = pil.mode if pil.mode in ("RGB", "CMYK", "L") else None
    if icc and mode:
        try:
            rgb = ImageCms.applyTransform(pil, _transform_icc_to_srgb(icc, mode))
            return np.asarray(rgb, dtype=np.float32) / 255.0
        except Exception as exc:
            print(f"   自带 ICC 转换失败，改用 PSOcoated_v3: {exc}")
    _init_icc()
    if pil.mode == "CMYK":
        rgb = ImageCms.applyTransform(pil, _cmyk_to_rgb)
    else:
        rgb = pil.convert("RGB")
    return np.asarray(rgb, dtype=np.float32) / 255.0


def _read_embedded_image_raw(image):
    """★ 从 PDF XObject 直接解码嵌入图像，返回 PIL Image。
    完全绕过 pypdf 的 _xobj_to_image / MAX_IMAGE_BUFFER_SIZE 检查。
    支持 8bit FlateDecode + DeviceCMYK / DeviceRGB / DeviceGray / ICCBased(N=1/3/4)。
    其余情况（DCTDecode / JPXDecode / 复合 Filter / Indexed）返回 None。
    """
    from pypdf.generic import IndirectObject
    try:
        obj = image.indirect_reference.get_object()
        w = int(obj["/Width"])
        h = int(obj["/Height"])
        bpc = int(obj.get("/BitsPerComponent", 8))
        if bpc != 8:
            return None

        # ---- 解析 ColorSpace ----
        cs = obj.get("/ColorSpace")
        cs = cs.get_object() if isinstance(cs, IndirectObject) else cs
        n_channels = 0
        mode = None
        if isinstance(cs, list):
            cs_name = str(cs[0])
            if cs_name == "/ICCBased":
                icc_obj = cs[1].get_object() if isinstance(cs[1], IndirectObject) else cs[1]
                n = int(icc_obj.get("/N", 0))
                if n == 4:
                    n_channels, mode = 4, "CMYK"
                elif n == 3:
                    n_channels, mode = 3, "RGB"
                elif n == 1:
                    n_channels, mode = 1, "L"
                else:
                    return None
            else:
                return None
        else:
            cs_str = str(cs)
            if cs_str == "/DeviceCMYK":
                n_channels, mode = 4, "CMYK"
            elif cs_str == "/DeviceRGB":
                n_channels, mode = 3, "RGB"
            elif cs_str == "/DeviceGray":
                n_channels, mode = 1, "L"
            else:
                return None

        # ---- 只支持单一 FlateDecode ----
        filt = obj.get("/Filter")
        filt = filt.get_object() if isinstance(filt, IndirectObject) else filt
        if isinstance(filt, list):
            if [str(f) for f in filt] != ["/FlateDecode"]:
                return None
        elif str(filt) not in ("/FlateDecode", "FlateDecode"):
            return None

        # ---- 取原始字节 + 自己解压 ----
        raw = _raw_stream_bytes(obj)
        if raw is None:
            return None
        try:
            raw = zlib.decompress(raw)
        except Exception:
            return None

        expected = w * h * n_channels
        if len(raw) < expected:
            return None
        img = Image.frombytes(mode, (w, h), raw[:expected])
        if isinstance(cs, list) and str(cs[0]) == "/ICCBased":
            icc_obj = cs[1].get_object() if isinstance(cs[1], IndirectObject) else cs[1]
            icc = icc_obj.get_data() if hasattr(icc_obj, "get_data") else None
            if isinstance(icc, str):
                icc = icc.encode("latin-1")
            if icc:
                img.info["icc_profile"] = bytes(icc)
        return img
    except Exception:
        return None


def _pdf_image_icc_bytes(image):
    """从嵌入图的 /ICCBased 色彩空间取出配置文件。Device* 没有。"""
    from pypdf.generic import IndirectObject
    try:
        obj = image.indirect_reference.get_object()
        cs = obj.get("/ColorSpace")
        cs = cs.get_object() if isinstance(cs, IndirectObject) else cs
        if not isinstance(cs, list) or str(cs[0]) != "/ICCBased":
            return None
        icc_obj = cs[1].get_object() if isinstance(cs[1], IndirectObject) else cs[1]
        data = icc_obj.get_data() if hasattr(icc_obj, "get_data") else None
        if isinstance(data, str):
            data = data.encode("latin-1")
        return bytes(data) if data else None
    except Exception:
        return None


def _raw_stream_bytes(obj) -> bytes | None:
    """
    从 pypdf 的 EncodedStreamObject 里拿原始字节流（未解压）。
    不同 pypdf 版本内部字段名可能不同，逐个尝试。
    """
    # 新版：obj._data 是原始 bytes
    data = getattr(obj, "_data", None)
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    # 有些版本用 _raw_get_data
    try:
        raw = obj._raw_get_data()  # type: ignore[attr-defined]
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)
    except Exception:
        pass
    # 退路：直接从 pdf 流里按偏移读
    try:
        pdf = obj.pdf
        start = obj._data_start  # type: ignore[attr-defined]
        length = int(obj["/Length"])
        stream = getattr(pdf, "stream", None)
        if stream is not None:
            stream.seek(start)
            return stream.read(length)
    except Exception:
        pass
    return None


def _write_pypdf_cmyk_image(image_file, cmyk_pil):
    from pypdf.generic import (NameObject, NumberObject, StreamObject, ArrayObject)
    ref = image_file.indirect_reference
    old = ref.get_object()
    pdf = ref.pdf
    width, height = cmyk_pil.size

    icc_bytes = _icc_bytes()
    if icc_bytes:
        icc_stream = StreamObject()
        icc_stream[NameObject("/N")] = NumberObject(4)
        icc_stream[NameObject("/Alternate")] = NameObject("/DeviceCMYK")
        icc_stream.set_data(icc_bytes)
        icc_ref = pdf._add_object(icc_stream)
        cs = ArrayObject()
        cs.append(NameObject("/ICCBased"))
        cs.append(icc_ref)
        colorspace = cs
    else:
        colorspace = NameObject("/DeviceCMYK")

    new = StreamObject()
    new[NameObject("/Type")] = NameObject("/XObject")
    new[NameObject("/Subtype")] = NameObject("/Image")
    new[NameObject("/Width")] = NumberObject(width)
    new[NameObject("/Height")] = NumberObject(height)
    new[NameObject("/ColorSpace")] = colorspace
    new[NameObject("/BitsPerComponent")] = NumberObject(8)
    new[NameObject("/Filter")] = NameObject("/FlateDecode")
    for key in ("/SMask", "/Interpolate", "/Intent"):
        if key in old:
            new[NameObject(key)] = old[key]
    new.set_data(zlib.compress(cmyk_pil.tobytes(), 9))
    new.indirect_reference = ref
    pdf._objects[ref.idnum - 1] = new


def _embedded_to_print_cmyk(pil, model, curves, identity_curves, gains):
    """嵌入图送入模型后统一写成印刷 CMYK。

    有自带 ICC 时用它转到 sRGB；没有时 CMYK 用 PSOcoated_v3。
    模型输出再按 PSOcoated_v3 写回 CMYK。只套手工曲线时 CMYK 保持原样套 LUT。
    """
    src_mode = pil.mode
    if model is not None:
        rgb_np = _pil_to_model_rgb(pil)
        rgb_np, _ = _apply_curve_content(rgb_np, model, gains=gains)
        rgb_final = Image.fromarray((np.clip(rgb_np, 0, 1) * 255).astype(np.uint8), "RGB")
        try:
            cmyk = _to_cmyk_pil(rgb_final)
        finally:
            rgb_final.close()
    elif src_mode == "CMYK":
        cmyk = pil.copy()
    else:
        rgb_src = pil if src_mode == "RGB" else pil.convert("RGB")
        cmyk = _to_cmyk_pil(rgb_src)
        if rgb_src is not pil:
            rgb_src.close()
    if not identity_curves:
        graded = apply_cmyk_curves(cmyk, curves, content_only=False)
        if graded is not cmyk:
            cmyk.close()
            cmyk = graded
    return cmyk


def process_pdf_keep_structure(src, dest, model=None, curves=None, gains=None,
                               raw_cmyk=False):
    """复制输入 PDF 对象树，只替换嵌入图像素。"""
    _require_pypdf()
    from pypdf import PdfWriter

    src, dest = str(src), str(dest)
    identity_curves = curves is None or curves_are_identity(curves)
    if model is None and identity_curves:
        if os.path.abspath(src) != os.path.abspath(dest):
            shutil.copy2(src, dest)
        return 0

    gains = gains or {}
    writer = PdfWriter(clone_from=src)
    try:
        images = _collect_writer_images(writer)
        if not images:
            # 没有图像可处理，直接原样复制并返回 0（上层不会当成失败）
            writer.write(dest)
            _restore_pdf_private_tail(src, dest)
            print("   PDF 无嵌入位图，原样保留分层结构")
            return 0

        count = 0
        for image in images:
            cmyk = None
            src = None
            try:
                src = _read_embedded_image_raw(image)
                if src is None:
                    try:
                        src = image.image
                    except Exception as exc:
                        print(f"   跳过无法解码的嵌入图: {exc}")
                        continue
                if src is None:
                    print("   跳过 image.image 返回 None 的嵌入图")
                    continue
                if not _pil_icc_bytes(src):
                    icc = _pdf_image_icc_bytes(image)
                    if icc:
                        src.info["icc_profile"] = icc
                src_mode = src.mode
                cmyk = _embedded_to_print_cmyk(src, model, curves, identity_curves, gains)
                orig = image.indirect_reference.get_object()
                orig_w, orig_h = int(orig["/Width"]), int(orig["/Height"])
                if cmyk.size != (orig_w, orig_h):
                    resized = cmyk.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
                    cmyk.close()
                    cmyk = resized
                _write_pypdf_cmyk_image(image, cmyk)
                count += 1
                has_icc = bool(_pil_icc_bytes(src))
                if model is not None and src_mode == "CMYK":
                    route = "自带ICC→RGB→模型→CMYK" if has_icc else "PSOcoated_v3→RGB→模型→CMYK"
                elif model is not None and has_icc:
                    route = f"{src_mode}自带ICC→模型→CMYK"
                elif model is not None:
                    route = f"{src_mode}→模型→CMYK"
                else:
                    route = f"{src_mode}→CMYK"
                print(f"   嵌入图 {orig_w}×{orig_h}（{route}）已调色")
            except Exception as exc:
                print(f"   跳过无法处理的嵌入图: {exc}")
                traceback.print_exc()
            finally:
                for img in (cmyk, src):
                    try:
                        if img is not None:
                            img.close()
                    except Exception:
                        pass

        # ★ 关键：一张图都没成功，抛异常触发栅格化降级
        if count == 0:
            raise RuntimeError(
                f"分层路径未处理任何嵌入图（共发现 {len(images)} 张）")

        writer.write(dest)
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            raise RuntimeError(f"分层输出文件无效: {dest}")
        print(f"   [分层] 处理 {count}/{len(images)} 张嵌入图 → {Path(dest).name}")
    finally:
        writer.close()
    _restore_pdf_private_tail(src, dest)
    return count

def _pdf_private_tail(path):
    with open(path, "rb") as f:
        data = f.read()
    marker = data.rfind(b"%%EOF")
    if marker < 0:
        return b""
    tail = data[marker + 5:]
    if b"8BPS" in tail or b"8BIM" in tail:
        return tail
    return b""


def _restore_pdf_private_tail(src, dest):
    tail = _pdf_private_tail(src)
    if not tail or os.path.abspath(src) == os.path.abspath(dest):
        return
    with open(dest, "ab") as f:
        if not tail.startswith((b"\n", b"\r")):
            f.write(b"\n")
        f.write(tail)


def apply_cmyk_curves_to_pdf(src, dest, curves):
    """在保留分层的 PDF 上套手工 CMYK 曲线。失败时抛出，由上层用整页 TIFF 重组。"""
    src, dest = str(src), str(dest)
    print(f"   [曲线调整] 开始: {Path(src).name} → {Path(dest).name}")
    try:
        if os.path.exists(dest):
            os.remove(dest)
    except Exception:
        pass
    process_pdf_keep_structure(src, dest, model=None, curves=curves, raw_cmyk=True)
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        raise RuntimeError(f"分层路径未生成有效文件: {dest}")
    print(f"   [曲线调整] ✅ 分层完成 ({os.path.getsize(dest)} bytes)")
    return dest


# ============================================================
# 保存与指标
# ============================================================

def save_processed_image(pred_pil, path, as_cmyk=True):
    path = cmyk_container_name(str(path))
    fmt = format_from_suffix(Path(path).suffix)
    _save_cmyk(pred_pil, path, fmt=fmt)
    return path


def pil_to_bytes(pil_img, fmt="JPEG", quality=95):
    buf = io.BytesIO()
    fmt_upper = fmt.upper()
    if fmt_upper == "JPEG":
        save_kw = {"format": "JPEG", "quality": quality, "subsampling": 0}
        if pil_img.mode != "CMYK":
            save_kw["optimize"] = True
        pil_img.save(buf, **save_kw)
        mime = "image/jpeg"
    elif fmt_upper == "PNG":
        pil_img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    elif fmt_upper == "TIFF":
        pil_img.save(buf, format="TIFF")
        mime = "image/tiff"
    else:
        raise ValueError(f"不支持的图像格式: {fmt}")
    return buf.getvalue(), mime


def _calc_metrics(pred_rgb, target_rgb):
    try:
        from skimage.color import rgb2lab, deltaE_ciede2000
    except ImportError:
        mae = float(np.abs(pred_rgb - target_rgb).mean())
        return {"mean_delta_e": None, "mae": mae}
    p = (np.clip(pred_rgb, 0, 1) * 255).astype(np.uint8)
    t = (np.clip(target_rgb, 0, 1) * 255).astype(np.uint8)
    de = float(deltaE_ciede2000(rgb2lab(p), rgb2lab(t)).mean())
    mae = float(np.abs(pred_rgb - target_rgb).mean())
    return {"mean_delta_e": de, "mae": mae}


def _read_target_rgb(target_path, size):
    width, height = size
    page_rgb, _, _ = next(iter_source_pages(target_path))
    if page_rgb.size != (width, height):
        page_rgb = page_rgb.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(page_rgb, dtype=np.float32) / 255.0


def _page_rgb_array(pil_rgb):
    return np.asarray(pil_rgb, dtype=np.float32) / 255.0


MAX_INFERENCE_PIXELS = 8000 * 8000


def _downscale_if_needed(img_rgb, max_pixels=MAX_INFERENCE_PIXELS):
    h, w = img_rgb.shape[:2]
    if h * w <= max_pixels:
        return img_rgb, 1.0
    scale = (max_pixels / (h * w)) ** 0.5
    new_w, new_h = int(w * scale), int(h * scale)
    print(f"   [尺寸限制] {w}×{h} → {new_w}×{new_h}（缩放 {scale:.3f}）")
    img_pil = Image.fromarray((img_rgb * 255).astype(np.uint8))
    img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    arr = np.asarray(img_pil, dtype=np.float32) / 255.0
    img_pil.close()
    return arr, scale


# ============================================================
# 主流程
# ============================================================

def process_file(image_path, model, output_dir, output_filename=None,
                 target_path=None, return_metrics=True,
                 proof_dir=None, input_preview_dir=None, page_dir=None,
                 stage_dir=None, page_stem="page",
                 proof_size=1400, input_preview_size=1600,
                 pdf_dpi=None, gains=None):
    if model is None:
        raise ValueError("必须提供已加载的模型")

    gains = gains or {}
    image_path = str(image_path)
    src_name = Path(image_path).name
    download_name = cmyk_container_name(output_filename or src_name)
    out_fmt = format_from_suffix(Path(download_name).suffix)
    os.makedirs(output_dir, exist_ok=True)
    for d in (proof_dir, input_preview_dir, stage_dir):
        if d:
            os.makedirs(d, exist_ok=True)

    doc_dpi = pdf_render_dpi(image_path, pdf_dpi or PDF_DPI) if is_pdf(image_path) else None
    pages_meta, cmyk_page_paths, metrics = [], [], None
    structured_pdf = False
    out_path = os.path.join(output_dir, download_name)

    if out_fmt == "PDF":
        page_dir = Path(page_dir or Path(output_dir) / "pages")
        page_dir.mkdir(parents=True, exist_ok=True)
        try:
            n_img = process_pdf_keep_structure(image_path, out_path,
                                               model=model, gains=gains)
            structured_pdf = True
            print(f"   输出 CMYK PDF（保留分层）: {Path(out_path).name}，{n_img} 张嵌入图")
        except Exception as exc:
            print(f"   结构保留失败，回退整页栅格: {exc}")

    source_pages = iter_source_pages(image_path, dpi=doc_dpi)
    if structured_pdf:
        source_pages = zip(source_pages, iter_source_pages(out_path, dpi=doc_dpi))

    for index, page_item in enumerate(source_pages):
        if structured_pdf:
            (page_rgb, size_pt, page_dpi), (out_rgb, _, _) = page_item
        else:
            page_rgb, size_pt, page_dpi = page_item
            out_rgb = None

        img_rgb = _page_rgb_array(page_rgb)
        height, width = img_rgb.shape[:2]
        label = f"{src_name} p{index + 1}" if out_fmt == "PDF" else src_name
        print(f"📄 {label}  {width}x{height} @ {page_dpi:.0f}dpi")

        pred_pil = None
        cmyk_page = None
        try:
            if structured_pdf:
                pred = _page_rgb_array(out_rgb)
                pred_pil = out_rgb.convert("RGB")
                boxes = find_content_boxes(img_rgb)
                stage1 = stage2 = None
            else:
                img_rgb, _scale = _downscale_if_needed(img_rgb)
                pred, boxes, stage1, stage2 = process_with_stages(img_rgb, model, gains=gains)
                _save_stage_previews(stage1, stage2, page_index=index,
                                     stage_dir=stage_dir, stem=page_stem)
                pred_pil = Image.fromarray((pred * 255).astype(np.uint8), "RGB")

            paper = paper_mask_rgb(img_rgb)
            content_diff = float(np.abs(pred - img_rgb)[~paper].mean()) if (~paper).any() else 0.0
            page_diff = float(np.abs(pred - img_rgb).mean())
            print(f"   图块 {len(boxes)} 个  内容区作用量 {content_diff:.4f}  整页 {page_diff:.4f}")

            if index == 0 and target_path and os.path.exists(target_path) and return_metrics:
                target_rgb = _read_target_rgb(target_path, (width, height))
                metrics = _calc_metrics(pred, target_rgb)
                print(f"   ✅ ΔE00={metrics['mean_delta_e']:.3f}  MAE={metrics['mae']:.4f}")

            cmyk_page = _to_cmyk_pil(pred_pil)
            page_meta = {"index": index, "width": width, "height": height,
                         "dpi": round(float(page_dpi), 2),
                         "size_pt": [round(float(size_pt[0]), 2), round(float(size_pt[1]), 2)]}

            if proof_dir:
                proof = cmyk_page.copy()
                proof.thumbnail((proof_size, proof_size), Image.Resampling.LANCZOS)
                proof_name = f"{page_stem}_p{index}.tif"
                save_cmyk_image(proof, Path(proof_dir) / proof_name)
                page_meta["proof_file"] = proof_name
                proof.close()

            if input_preview_dir:
                src_preview = page_rgb.copy()
                src_preview.thumbnail((input_preview_size, input_preview_size), Image.Resampling.LANCZOS)
                preview_name = f"{page_stem}_p{index}.jpg"
                src_preview.save(Path(input_preview_dir) / preview_name, quality=88, optimize=True)
                page_meta["input_preview_file"] = preview_name

                try:
                    src_cmyk = _to_cmyk_pil(src_preview)
                    src_proof = cmyk_to_srgb(src_cmyk)
                    src_proof_name = f"{page_stem}_p{index}_srcproof.jpg"
                    src_proof.save(Path(input_preview_dir) / src_proof_name,
                                   quality=88, optimize=True)
                    page_meta["input_proof_file"] = src_proof_name
                    src_cmyk.close()
                    src_proof.close()
                except Exception as exc:
                    print(f"   原图 CMYK 软打样失败: {exc}")
                finally:
                    src_preview.close()

            if out_fmt == "PDF":
                page_name = f"{page_stem}_p{index}.tif"
                save_cmyk_image(cmyk_page, page_dir / page_name, lossless=True)
                cmyk_page_paths.append(page_dir / page_name)
                page_meta["base_page_file"] = page_name
            else:
                save_cmyk_image(cmyk_page, os.path.join(output_dir, download_name))

            pages_meta.append(page_meta)
        finally:
            del img_rgb
            if pred_pil is not None:
                pred_pil.close()
            if cmyk_page is not None:
                cmyk_page.close()
            page_rgb.close()
            if out_rgb is not None and out_rgb is not page_rgb:
                out_rgb.close()

        import gc
        gc.collect()

    if out_fmt == "PDF" and not structured_pdf:
        save_cmyk_pdf(
            cmyk_page_paths, out_path,
            dpi=doc_dpi or PDF_DPI,
            page_sizes_pt=[p["size_pt"] for p in pages_meta],
        )
        print(f"   输出 CMYK PDF: {Path(out_path).name}（{len(pages_meta)} 页，{doc_dpi or PDF_DPI:.0f} dpi，无损）")
    elif out_fmt != "PDF":
        print(f"   输出 CMYK: {Path(out_path).name}")

    if metrics is not None:
        metrics_path = os.path.join(output_dir, f"{Path(download_name).stem}_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return {"output_path": out_path, "metrics": metrics,
            "page_count": len(pages_meta), "pages": pages_meta,
            "dpi": doc_dpi or (pages_meta[0]["dpi"] if pages_meta else None)}


def load_first_page_rgb(path):
    page_rgb, _, _ = next(iter_source_pages(path))
    return np.asarray(page_rgb, dtype=np.uint8)