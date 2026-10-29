# -*- coding: utf-8 -*-
"""
三阶段测试脚本（修复版 v2）

v2 修复：
  ★ _cmyk_to_rgb 导入时机 bug —— 原来 `from processor import _cmyk_to_rgb`
    会把 None 复制到本地命名空间，之后 _init_icc() 更新的是 processor 模块
    里的变量，本地这份永远是 None，导致 ImageCms.applyTransform(img, None)
    触发 `'NoneType' object has no attribute 'apply'`。
    现改为使用 processor.cmyk_to_srgb() 公共接口，每次读取都取最新 transform。

其余同 v1：
  1. Target 走 ICC（PSOcoated_v3.icc → sRGB），与训练口径一致
  2. stage1/stage2 为 None 时不崩溃
  3. OOM 回退 CPU 时同步更新 HybridPipeline.device
  4. 平均统计忽略 None 阶段
  5. 启动时打印阶段可用性
"""
import os
import glob
import json
import gc
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from skimage.color import rgb2lab, deltaE_ciede2000

import processor
from processor import load_model, process_with_stages, HybridPipeline

# ========== 可配置参数 ==========
MAX_INFER_SIZE = 3000
# ================================


def read_image_with_icc(path, is_target=True):
    """
    与训练口径一致的图像读取：
      - target: CMYK → ICC(PSOcoated_v3.icc) → sRGB
      - input : 直接 .convert("RGB")
    ★ 用 processor.cmyk_to_srgb()，它在内部每次 _init_icc() 后再读最新的
      _cmyk_to_rgb，不会踩"模块级变量导入快照"的坑。
    """
    img = Image.open(path)
    if is_target and img.mode == "CMYK":
        img = processor.cmyk_to_srgb(img)   # 内部已调用 _init_icc()
    else:
        img = img.convert("RGB")
    return img


def calc_metrics(pred_rgb, target_rgb):
    p = (np.clip(pred_rgb, 0, 1) * 255).astype(np.uint8)
    t = (np.clip(target_rgb, 0, 1) * 255).astype(np.uint8)
    de_map = deltaE_ciede2000(rgb2lab(p), rgb2lab(t))
    mae = float(np.abs(pred_rgb - target_rgb).mean())
    return {"mean_delta_e": float(de_map.mean()), "mae": mae}


def find_image_pairs(folder):
    input_files = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        for f in glob.glob(os.path.join(folder, ext)):
            base = os.path.basename(f)
            if "_input." in base.lower():
                input_files[base] = f

    pairs = []
    for base, inp_path in input_files.items():
        tgt_base = base.replace("_input.", "_target.")
        tgt_path = os.path.join(folder, tgt_base)
        if os.path.exists(tgt_path):
            pairs.append((inp_path, tgt_path))
        else:
            print(f"[警告] 未找到与 {base} 配对的 target 文件: {tgt_base}")
    return pairs


def smart_resize(np_img, max_size=MAX_INFER_SIZE):
    h, w = np_img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_size:
        return np_img, 1.0
    scale = max_size / float(long_side)
    new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
    pil = Image.fromarray((np_img * 255).astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.float32) / 255.0, scale


def _move_model_to(model, device_str):
    """HybridPipeline 不是 nn.Module，需手动搬子模块 + 更新 .device"""
    target_dev = torch.device(device_str)
    if isinstance(model, HybridPipeline):
        for name in ("curve_model", "skin_model", "unet_model"):
            sub = getattr(model, name, None)
            if sub is not None:
                sub.to(target_dev)
        model.device = target_dev
        model.eval()
    else:
        model.to(target_dev)
        model.eval()
    return model


def infer_with_oom_protection(model, inp_np, device_str, max_retry=2):
    for attempt in range(max_retry):
        try:
            with torch.no_grad():
                final_np, _, stage1_np, stage2_np = process_with_stages(inp_np, model,gains={"curve": 1.0, "skin": 1.0, "residual": 1.0})
            return final_np, stage1_np, stage2_np, device_str
        except RuntimeError as e:
            err_msg = str(e)
            if ("CUDA out of memory" in err_msg) or ("OOM" in err_msg):
                print(f"    ⚠️ CUDA OOM (attempt {attempt+1}/{max_retry})，清理显存...")
                if device_str == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                if attempt == max_retry - 1:
                    print(f"    🔄 回退到 CPU 推理...")
                    _move_model_to(model, "cpu")
                    with torch.no_grad():
                        final_np, _, stage1_np, stage2_np = process_with_stages(inp_np, model)
                    return final_np, stage1_np, stage2_np, "cpu"
            else:
                raise
    raise RuntimeError("推理失败")


def main():
    parser = argparse.ArgumentParser(description="分三阶段测试模型准确性（修复版 v2）")
    parser.add_argument("--folder", type=str,
                        default="/workspace/meng_xf/preprint/preprint_data/negative")
    parser.add_argument("--model", type=str,
                        default="checkpoints/curve_pred_best.pth")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_size", type=int, default=MAX_INFER_SIZE,
                        help=f"推理最大长边像素（默认 {MAX_INFER_SIZE}）")
    parser.add_argument("--output_plot", type=str, default="stage_wise_delta_e.png")
    parser.add_argument("--output_json", type=str, default="stage_wise_report.json")
    args = parser.parse_args()

    # 先初始化 ICC，确保后续 cmyk_to_srgb 直接可用
    processor._init_icc()

    print(f"[初始化] 加载模型: {args.model}")
    model = load_model(args.model, args.device)
    model.eval()

    has_skin, has_unet = False, False
    if isinstance(model, HybridPipeline):
        has_skin = model.skin_model is not None
        has_unet = model.unet_model is not None
    print(f"[初始化] HybridPipeline={isinstance(model, HybridPipeline)} | "
          f"skin={'✅' if has_skin else '❌'} | unet={'✅' if has_unet else '❌'}")
    if not isinstance(model, HybridPipeline):
        print("    ⚠️ 当前加载的是纯曲线模型，Stage1/Stage2 不可用，仅评估 Final")

    pairs = find_image_pairs(args.folder)
    if not pairs:
        raise FileNotFoundError("未找到 input/target 图像对")
    print(f"[初始化] 共找到 {len(pairs)} 对测试图像，最大推理尺寸={args.max_size}px\n")

    results, skipped = [], []

    for idx, (inp_path, tgt_path) in enumerate(pairs):
        print(f"[{idx + 1}/{len(pairs)}] 处理: {os.path.basename(inp_path)}")

        try:
            inp_pil = read_image_with_icc(inp_path, is_target=False)
            tgt_pil = read_image_with_icc(tgt_path, is_target=True)
        except Exception as e:
            print(f"    ❌ 读取失败: {e}")
            skipped.append(os.path.basename(inp_path))
            continue

        orig_h, orig_w = inp_pil.height, inp_pil.width
        inp_np = np.asarray(inp_pil, dtype=np.float32) / 255.0

        inp_np, scale = smart_resize(inp_np, args.max_size)
        if scale < 1.0:
            print(f"    📐 输入图 {orig_w}×{orig_h} 缩放到 "
                  f"{inp_np.shape[1]}×{inp_np.shape[0]}")

        try:
            final_np, stage1_np, stage2_np, used_device = infer_with_oom_protection(
                model, inp_np, args.device
            )
        except Exception as e:
            print(f"    ❌ 推理失败，跳过: {e}")
            skipped.append(os.path.basename(inp_path))
            if args.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            continue

        if args.device == "cuda":
            torch.cuda.empty_cache()

        if final_np.shape[:2] != (tgt_pil.height, tgt_pil.width):
            tgt_pil = tgt_pil.resize((final_np.shape[1], final_np.shape[0]),
                                     Image.Resampling.LANCZOS)
        tgt_np = np.asarray(tgt_pil, dtype=np.float32) / 255.0

        m_s1 = calc_metrics(stage1_np, tgt_np) if stage1_np is not None else None
        m_s2 = calc_metrics(stage2_np, tgt_np) if stage2_np is not None else None
        m_final = calc_metrics(final_np, tgt_np)

        if m_s1 is not None:
            print(f"  ├─ Stage1(全局曲线) ΔE00: {m_s1['mean_delta_e']:.4f}")
        else:
            print(f"  ├─ Stage1(全局曲线) 不可用")
        if m_s2 is not None:
            print(f"  ├─ Stage2(肤色修正) ΔE00: {m_s2['mean_delta_e']:.4f}")
        else:
            print(f"  ├─ Stage2(肤色修正) 不可用")
        print(f"  └─ Final(残差精修)  ΔE00: {m_final['mean_delta_e']:.4f}")

        results.append({
            "file": os.path.basename(inp_path),
            "orig_size": f"{orig_w}×{orig_h}",
            "infer_size": f"{inp_np.shape[1]}×{inp_np.shape[0]}",
            "scale": round(scale, 3),
            "device": used_device,
            "stage1": m_s1,
            "stage2": m_s2,
            "final": m_final,
        })

        del inp_np, final_np, stage1_np, stage2_np, tgt_np
        gc.collect()

    if not results:
        print("❌ 所有图像都处理失败！")
        return

    def _avg(stage_key):
        vals = [r[stage_key]["mean_delta_e"] for r in results if r.get(stage_key)]
        return float(np.mean(vals)) if vals else None

    avg_s1 = _avg("stage1")
    avg_s2 = _avg("stage2")
    avg_final = _avg("final")

    print("\n" + "=" * 60)
    parts = []
    if avg_s1 is not None:
        parts.append(f"Stage1: {avg_s1:.4f}")
    if avg_s2 is not None:
        parts.append(f"Stage2: {avg_s2:.4f}")
    parts.append(f"Final: {avg_final:.4f}")
    print("平均色差 -> " + " | ".join(parts))
    if skipped:
        print(f"跳过 {len(skipped)} 张图: {skipped}")
    print("=" * 60)

    report = {
        "config": {
            "max_infer_size": args.max_size,
            "model": args.model,
            "target_reading": "ICC (PSOcoated_v3.icc → sRGB)",
            "has_skin": has_skin,
            "has_unet": has_unet,
        },
        "summary": {"avg_stage1": avg_s1, "avg_stage2": avg_s2, "avg_final": avg_final},
        "skipped": skipped,
        "details": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"详细报告已保存: {args.output_json}")

    plt.figure(figsize=(13, 7))
    indices = np.arange(len(results))

    if avg_s1 is not None:
        s1 = np.array([r["stage1"]["mean_delta_e"] if r["stage1"] else np.nan
                       for r in results])
        valid = ~np.isnan(s1)
        plt.scatter(indices[valid], s1[valid], c='blue', s=60, alpha=0.6,
                    label='Stage1: Global Curve')
        plt.axhline(y=avg_s1, color='blue', linestyle='--', linewidth=1.5,
                    label=f'Avg Stage1={avg_s1:.3f}')

    if avg_s2 is not None:
        s2 = np.array([r["stage2"]["mean_delta_e"] if r["stage2"] else np.nan
                       for r in results])
        valid = ~np.isnan(s2)
        plt.scatter(indices[valid], s2[valid], c='green', s=60, alpha=0.6,
                    label='Stage2: + Skin Curve')
        plt.axhline(y=avg_s2, color='green', linestyle='--', linewidth=1.5,
                    label=f'Avg Stage2={avg_s2:.3f}')

    final_des = [r["final"]["mean_delta_e"] for r in results]
    plt.scatter(indices, final_des, c='red', s=60, alpha=0.8,
                label='Final: + Residual U-Net')
    plt.axhline(y=avg_final, color='red', linestyle='-', linewidth=2,
                label=f'Avg Final={avg_final:.3f}')

    plt.xlabel("Image Index", fontsize=12)
    plt.ylabel("Mean ΔE00", fontsize=12)
    plt.title("Delta E Distribution across Processing Stages (ICC-aligned)", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150)
    plt.close()
    print(f"三阶段对比图已保存: {args.output_plot}")


if __name__ == "__main__":
    main()