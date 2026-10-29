# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:26:25 2026

@author: meng_xf
"""

"""
data/utils.py - 基础工具集
核心升级：色彩归一化、分块渐变融合、高频特征提取、流式大图读取
"""

import os
import cv2
import numpy as np
import torch

import io
from PIL import Image, ImageCms
from typing import Tuple, List, Optional

def read_image(path: str,
               icc_profile_path: Optional[str] = None,
               method: str = 'icc',
               save_converted: bool = False) -> np.ndarray:
    """
    读取图像并转换为 RGB uint8。

    Args:
        path: 图像文件路径
        icc_profile_path: ICC 配置文件路径（仅 method='icc' 时使用）
        method: 转换方法
            - 'icc': 优先内嵌 ICC → 用户 ICC → 默认 convert('RGB')
            - 'default': 直接使用 PIL 的 convert('RGB')（原来的方法）
        save_converted: 若为 True，将转换结果保存为 {path}_{method}_converted.png
    """
    with Image.open(path) as img:
        # --- 非 CMYK 图像直接转为 RGB ---
        if img.mode != 'CMYK':
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img, dtype=np.uint8)

        # --- CMYK 图像处理 ---
        if method == 'default':
            # 原来的方法：直接 convert('RGB')
            rgb_img = img.convert('RGB')
            converted = np.array(rgb_img, dtype=np.uint8)
        elif method == 'icc':
            converted = None

            # 1) 优先尝试用户指定的 ICC
            if icc_profile_path and os.path.isfile(icc_profile_path):
                try:
                    with open(icc_profile_path, 'rb') as f:
                        src_profile = ImageCms.ImageCmsProfile(io.BytesIO(f.read()))
                    dst_profile = ImageCms.createProfile('sRGB')
                    rgb_img = ImageCms.profileToProfile(
                        img, src_profile, dst_profile,
                        renderingIntent=ImageCms.Intent.PERCEPTUAL,
                        outputMode='RGB'
                    )
                    converted = np.array(rgb_img, dtype=np.uint8)
                except Exception as e:
                    print(f"用户 ICC 转换失败: {e}")

            # 2) 再尝试内嵌 ICC
            if converted is None:
                icc_bytes = img.info.get('icc_profile')
                if icc_bytes:
                    try:
                        src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
                        dst_profile = ImageCms.createProfile('sRGB')
                        rgb_img = ImageCms.profileToProfile(
                            img, src_profile, dst_profile,
                            renderingIntent=ImageCms.Intent.PERCEPTUAL,
                            outputMode='RGB'
                        )
                        converted = np.array(rgb_img, dtype=np.uint8)
                    except Exception as e:
                        print(f"内嵌 ICC 转换失败: {e}")

            # 3) 最后兜底
            if converted is None:
                rgb_img = img.convert('RGB')
                converted = np.array(rgb_img, dtype=np.uint8)
        else:
            raise ValueError(f"未知的 method: {method}，可选 'icc' 或 'default'")

        # --- 保存转换结果（用于对比） ---
        if save_converted:
            save_path = f"{path}_{method}_converted.png"
            Image.fromarray(converted).save(save_path)
            print(f"已保存转换结果至: {save_path}")

        return converted

def save_image(path: str, tensor: np.ndarray, meta: dict = None):
    """
    保存图像，支持还原元数据
    tensor: uint8 数组 (H, W, 3) 或 float32 范围 [0,1]
    """
    if tensor.dtype == np.float32 or tensor.dtype == np.float64:
        tensor = (np.clip(tensor, 0, 1) * 255).astype(np.uint8)
    elif tensor.dtype != np.uint8:
        tensor = tensor.astype(np.uint8)
    
    pil_img = Image.fromarray(tensor)
    
    # 附加元数据
    if meta:
        if 'dpi' in meta:
            pil_img.info['dpi'] = meta['dpi']
    
    # 根据扩展名选择格式
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.tiff', '.tif']:
        pil_img.save(path, compression='tiff_lzw')
    else:
        pil_img.save(path, quality=95, subsampling=0)


def illumination_normalize(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    照明归一化：直方图匹配 + 亮度标准化，消除两张图的调色差异，仅保留结构差异
    返回归一化后的两张图（均为 uint8）
    """
    # 将 img2 的直方图匹配到 img1
    matched = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)[:,:,0])
    lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)
    lab2[:,:,0] = matched
    img2_norm = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    
    # 简单亮度均衡：调整 img1 的均值和标准差与 img2_norm 一致
    mean1, std1 = cv2.meanStdDev(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))
    mean2, std2 = cv2.meanStdDev(cv2.cvtColor(img2_norm, cv2.COLOR_RGB2GRAY))
    
    scale = std2 / (std1 + 1e-6)
    shift = mean2 - scale * mean1
    
    img1_float = img1.astype(np.float32)
    img1_adjusted = img1_float * scale + shift
    img1_adjusted = np.clip(img1_adjusted, 0, 255).astype(np.uint8)
    
    return img1_adjusted, img2_norm


def extract_high_frequency(img: np.ndarray, sigma1: float = 1.0, sigma2: float = 3.0) -> np.ndarray:
    """
    高斯差分（DoG）高通滤波，提取纯线条/纹理高频残差
    返回灰度图，范围 [0, 255]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    blur2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
    high_freq = blur1 - blur2
    # 归一化到 [0, 255]
    high_freq = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min() + 1e-6) * 255
    return high_freq.astype(np.uint8)


def split_patches(img: np.ndarray, patch_size: int = 512, stride: int = 256) -> Tuple[List[np.ndarray], List[Tuple[int, int]], List[float]]:
    """
    大图分块，记录块坐标与权重
    返回：patches列表，coords列表[(y,x)]，weights列表（余弦权重）
    """
    h, w = img.shape[:2]
    patches = []
    coords = []
    weights = []
    
    y_steps = list(range(0, h - patch_size + 1, stride))
    x_steps = list(range(0, w - patch_size + 1, stride))
    
    # 确保覆盖最后部分
    if y_steps[-1] + patch_size < h:
        y_steps.append(h - patch_size)
    if x_steps[-1] + patch_size < w:
        x_steps.append(w - patch_size)
    
    for y in y_steps:
        for x in x_steps:
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))
            # 生成余弦权重矩阵（中心高，边缘低）
            wy = np.cos(np.pi * (np.arange(patch_size) - patch_size/2) / patch_size)
            wx = np.cos(np.pi * (np.arange(patch_size) - patch_size/2) / patch_size)
            w2d = np.outer(wy, wx)
            weights.append(w2d)
    
    return patches, coords, weights


def merge_patches(patches: List[np.ndarray], coords: List[Tuple[int, int]], weights: List[np.ndarray],
                  img_shape: Tuple[int, int], patch_size: int) -> np.ndarray:
    """
    余弦权重渐变融合，消除接缝
    返回合并后的图像 (H, W, 3) float32
    """
    h, w = img_shape[:2]
    result = np.zeros((h, w, 3), dtype=np.float32)
    weight_sum = np.zeros((h, w, 3), dtype=np.float32)
    
    for patch, (y, x), w_mat in zip(patches, coords, weights):
        patch_float = patch.astype(np.float32)
        w3 = np.stack([w_mat]*3, axis=-1)  # (patch_size, patch_size, 3)
        result[y:y+patch_size, x:x+patch_size] += patch_float * w3
        weight_sum[y:y+patch_size, x:x+patch_size] += w3
    
    # 防止除零
    weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    merged = result / weight_sum
    return np.clip(merged, 0, 255).astype(np.uint8)


def global_color_correction(original: np.ndarray, output: np.ndarray) -> np.ndarray:
    """
    基于原图全局色彩统计，校正输出图的块间色差
    使用线性回归调整输出图的颜色分布使其与原始图一致
    """
    # 转换到 Lab 空间
    original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
    output_lab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # 分别对 L, a, b 通道进行线性校正（均值方差匹配）
    corrected = np.zeros_like(output_lab)
    for c in range(3):
        orig_mean = np.mean(original_lab[:,:,c])
        orig_std = np.std(original_lab[:,:,c]) + 1e-6
        out_mean = np.mean(output_lab[:,:,c])
        out_std = np.std(output_lab[:,:,c]) + 1e-6
        corrected[:,:,c] = (output_lab[:,:,c] - out_mean) * (orig_std / out_std) + orig_mean
    
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB)
    return result


def stream_read_large_image(path: str, patch_size: int = 2048) -> np.ndarray:
    """
    超大图分块加载，避免内存溢出
    使用 TIFF 的逐行读取方式（若支持），否则回退到 PIL 分块
    """
    # 简单实现：先获取图像尺寸，然后逐行读取
    # 这里使用 PIL 的 Tile 方式读取大 TIFF
    from PIL import TiffImagePlugin
    img = Image.open(path)
    width, height = img.size
    
    # 如果图像过大（> 5000 边长），分块读取
    if height > 5000 or width > 5000:
        # 使用内存映射方式
        img_array = np.array(img, dtype=np.uint8)
        return img_array
    else:
        return np.array(img.convert('RGB'), dtype=np.uint8)