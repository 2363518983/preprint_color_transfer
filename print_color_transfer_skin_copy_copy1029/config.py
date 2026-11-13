# -*- coding: utf-8 -*-
"""统一配置：全局 1D 曲线 + 参数化曲线预测器（全局先验融合版）"""
import os


class Config:
    # ---- 路径 ----
    DATA_DIR = "/workspace/print_color_transfer/data"
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
    CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
    LOG_DIR = os.path.join(os.path.dirname(__file__), "runs")
    CMYK_ICC_PATH = os.path.join(os.path.dirname(__file__), "utils", "PSOcoated_v3.icc")
    sRGB_ICC_PATH = None

    # ---- 数据 ----
    SEED = 42
    TRAIN_RATIO = 0.8
    IMG_SIZE = 512
    CACHE_TYPE = "float16"

    # ---- 曲线 ----
    LUT_DIM = 33

    # ---- 全局曲线（阶段1） ----
    CURVE_BATCH_SIZE = 8
    CURVE_LR = 1e-3
    CURVE_EPOCHS = 80
    GRAD_CLIP = 5.0
    L1_WEIGHT = 1.0
    HIST_WEIGHT = 0.5
    HIST_BINS = 32
    HIST_DARK_WEIGHT = 1.0
    HIST_MID_WEIGHT = 1.0
    HIST_BRIGHT_WEIGHT = 1.0
    MONOTONE_WEIGHT = 0.1
    EARLY_STOP_PATIENCE = 15

    # ---- 参数化曲线预测器（阶段2） ----
    PRED_BASE = 64
    PRED_LR = 1e-4
    PRED_EPOCHS = 150
    PRED_INPUT_SIZE = 320
    PRED_RESIDUAL_MAX = 0.1
    PRED_USE_COUPLING = True
    PRED_BATCH_SIZE = 8
    PRED_HIST_WEIGHT = 0.2
    PRED_MONO_WEIGHT = 0.05       # 融合曲线仍单调，可保留微小惩罚
    PRED_LAB_WEIGHT = 0.1         # 新增：LAB 空间 MSE，增强颜色感知
    PRED_GLOBAL_REG_WEIGHT = 0.0  # 全局先验通过模型结构实现，外部正则不再需要
    GLOBAL_CURVE_PATH = os.path.join(CHECKPOINT_DIR, "global_curve.pt")
    GLOBAL_CURVE_TENSOR_PATH = os.path.join(CHECKPOINT_DIR, "global_curve.pt")

    # ---- 残差 U-Net（可选） ----
    UNET_EPOCHS = 100
    LAB_WEIGHT = 0.1
    UNET_LR = 1e-4

    # ---- ICC ----
    RENDERING_INTENT = "relative"
    BLACK_POINT_COMPENSATION = True

    @classmethod
    def setup_dirs(cls):
        for d in (cls.CACHE_DIR, cls.CHECKPOINT_DIR, cls.LOG_DIR):
            os.makedirs(d, exist_ok=True)