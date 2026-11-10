# -*- coding: utf-8 -*-
"""训练 / 推理统一配置（合并 preprint 训练管线字段 + 本地路径约定）"""
import os
from pathlib import Path

_ROOT = os.path.dirname(os.path.abspath(__file__))


class Config:
    # ---- 路径 ----
    DATA_DIR = os.environ.get("PRINT_DATA_DIR", "/workspace/print_color_transfer/data")
    CACHE_DIR = os.path.join(_ROOT, "cache")
    CHECKPOINT_DIR = os.path.join(_ROOT, "checkpoints")
    LOG_DIR = os.path.join(_ROOT, "runs")
    CMYK_ICC_PATH = os.path.join(_ROOT, "utils", "PSOcoated_v3.icc")
    SKIN_MASK_DIR = os.path.join(_ROOT, "mask_cache")
    GLOBAL_CURVE_PATH = os.path.join(CHECKPOINT_DIR, "global_curve.pt")
    GLOBAL_CURVE_TENSOR_PATH = os.path.join(CHECKPOINT_DIR, "global_curve.pt")
    sRGB_ICC_PATH = None

    # ---- 数据 ----
    SEED = 42
    TRAIN_RATIO = 0.9
    IMG_SIZE = 512
    CACHE_TYPE = "float16"

    # ---- 曲线 / CurvePredictor（v3）----
    LUT_DIM = 33
    PRED_BASE = 64
    PRED_RESIDUAL_MAX = 0.1
    PRED_USE_COUPLING = True
    PRED_INPUT_SIZE = 320
    PRED_LR = 1e-4
    PRED_EPOCHS = 150
    PRED_BATCH_SIZE = 8
    PRED_HIST_WEIGHT = 0.1
    PRED_LAB_WEIGHT = 0.3
    PRED_MONO_WEIGHT = 0.05
    PRED_GLOBAL_REG_WEIGHT = 0.0

    # ---- 全局曲线（阶段 1，兼容 train.py）----
    CURVE_BATCH_SIZE = 8
    CURVE_LR = 1e-3
    CURVE_EPOCHS = 80
    GRAD_CLIP = 1.0
    L1_WEIGHT = 1.0
    HIST_WEIGHT = 0.5
    HIST_BINS = 32
    HIST_DARK_WEIGHT = 1.0
    HIST_MID_WEIGHT = 1.0
    HIST_BRIGHT_WEIGHT = 1.0
    MONOTONE_WEIGHT = 0.1
    EARLY_STOP_PATIENCE = 30

    # ---- U-Net（v3 空间门控）----
    UNET_EPOCHS = 100
    UNET_BASE = 64
    UNET_IN_CH = 6
    UNET_OUT_CH = 3
    UNET_GATE_INIT = 0.01
    UNET_GATE_MAX = 0.2
    UNET_LR = 1e-4
    UNET_WEIGHT_DECAY = 1e-6

    # ---- 损失权重（肤色 + 残差）----
    LAB_WEIGHT = 0.3
    L1_SKIN_WEIGHT = 2.0
    LAB_SKIN_WEIGHT = 1.5
    VGG_WEIGHT = 0.0
    RES_TARGET = 0.05
    RES_REG_WEIGHT = 1.0
    GATE_REG_WEIGHT = 0.5

    # ---- ICC ----
    RENDERING_INTENT = "relative"
    BLACK_POINT_COMPENSATION = True

    @classmethod
    def setup_dirs(cls):
        for d in (cls.CACHE_DIR, cls.CHECKPOINT_DIR, cls.LOG_DIR, cls.SKIN_MASK_DIR):
            Path(d).mkdir(parents=True, exist_ok=True)
