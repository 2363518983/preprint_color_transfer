# -*- coding: utf-8 -*-
"""ICC 口径诊断：target 用 PSOcoated_v3 转 sRGB，再评估 1D 曲线天花板。"""
import os, glob, numpy as np
from PIL import Image, ImageCms
from skimage.color import rgb2lab, deltaE_ciede2000

DATA = "/home/admin/picture_data/clean_out/clean"
CMYK_ICC = "utils/PSOcoated_v3.icc"
N = 60

_srgb = ImageCms.createProfile("sRGB")
_cmyk_prof = ImageCms.getOpenProfile(CMYK_ICC)
_cmyk_to_rgb = ImageCms.buildTransformFromOpenProfiles(
    _cmyk_prof, _srgb, "CMYK", "RGB",
    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
    flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)

def load_rgb(path):
    img = Image.open(path)
    if img.mode == "CMYK":
        img = ImageCms.applyTransform(img, _cmyk_to_rgb)
    else:
        img = img.convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0

def downsample(arr, s=512):
    pil = Image.fromarray((arr*255).astype(np.uint8))
    w,h = pil.size; side = min(w,h)
    pil = pil.crop(((w-side)//2,(h-side)//2,(w+side)//2,(h+side)//2)).resize((s,s))
    return np.array(pil, dtype=np.float32)/255.0

def de00(rgb1, rgb2):
    lab1 = rgb2lab((rgb1 * 255).astype(np.uint8)); lab2 = rgb2lab((rgb2 * 255).astype(np.uint8))
    return deltaE_ciede2000(lab1, lab2).mean()

pairs = []
for inp in sorted(glob.glob(os.path.join(DATA, "*_input.jpg"))):
    stem = os.path.splitext(os.path.basename(inp))[0].replace("_input","")
    tgt = os.path.join(DATA, stem + "_target.jpg")
    if os.path.exists(tgt): pairs.append((inp,tgt))
print(f"配对: {len(pairs)}")
rng = np.random.default_rng(42); rng.shuffle(pairs); samples = pairs[:N]

de_id, de_1d = [], []
for i,(ip,tp) in enumerate(samples):
    inp = downsample(load_rgb(ip)); tgt = downsample(load_rgb(tp))
    de_id.append(de00(inp,tgt))
    inp_f, tgt_f = inp.reshape(-1,3), tgt.reshape(-1,3)
    pred = np.zeros_like(tgt_f)
    for c in range(3):
        x,y = inp_f[:,c], tgt_f[:,c]
        bins = np.linspace(0,1,34); cy = np.clip(np.digitize(x,bins)-1,0,32)
        means = np.array([y[cy==k].mean() if (cy==k).any() else 0.5 for k in range(33)])
        pred[:,c] = np.clip(means[cy],0,1)
    de_1d.append(de00(pred.reshape(inp.shape), tgt))
    if (i+1)%10==0: print(f"  ...{i+1}/{N}")

de_id, de_1d = np.array(de_id), np.array(de_1d)
print("\n[ICC 口径]")
print(f"恒等   ΔE00: {de_id.mean():.3f} ± {de_id.std():.3f}")
print(f"1D曲线 ΔE00: {de_1d.mean():.3f} ± {de_1d.std():.3f}")
print(f"改进: {(1-de_1d.mean()/de_id.mean())*100:.1f}%")