import os
from pathlib import Path

class Config:
    SEED = 42
    DATA_DIR = "/workspace/meng_xf/preprint/preprint_data/clean"
    CACHE_DIR = "cache"
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    TRAIN_RATIO = 0.9
    IMG_SIZE = 512

    # CurvePredictor（v3）
    LUT_DIM = 33
    PRED_BASE = 64
    PRED_RESIDUAL_MAX = 0.1
    PRED_USE_COUPLING = True
    PRED_INPUT_SIZE = 320
    PRED_LR = 1e-4
    PRED_EPOCHS = 150
    PRED_BATCH_SIZE = 8           # ← 补齐
    PRED_LAB_WEIGHT = 0.3         # ← 补齐
    PRED_HIST_WEIGHT = 0.1        # ← 补齐
    GRAD_CLIP = 1.0               # ← 补齐
    EARLY_STOP_PATIENCE = 30      # ← 补齐
    GLOBAL_CURVE_PATH = os.path.join(CHECKPOINT_DIR, "global_curve.pt")   # ← 补齐

    # 软直方图损失
    HIST_BINS = 32                # ← 补齐
    HIST_DARK_WEIGHT = 1.0        # ← 补齐
    HIST_MID_WEIGHT = 1.0         # ← 补齐
    HIST_BRIGHT_WEIGHT = 1.0      # ← 补齐

    # U-Net（v5）
    UNET_BASE = 64
    UNET_IN_CH = 6
    UNET_OUT_CH = 3
    UNET_GATE_INIT = 0.01
    UNET_GATE_MAX = 0.2
    UNET_LR = 1e-4
    UNET_WEIGHT_DECAY = 1e-6

    # 损失权重
    L1_WEIGHT = 1.0
    L1_SKIN_WEIGHT = 2.0
    LAB_WEIGHT = 0.3
    LAB_SKIN_WEIGHT = 1.5
    VGG_WEIGHT = 0.0
    RES_TARGET = 0.05            # 统一：有效残差目标 RMS（原为 0.08，与 unet 脚本对齐）
    RES_REG_WEIGHT = 1.0         # 统一：残差正则权重（原为 0.5）
    GATE_REG_WEIGHT = 0.5        # 统一：gate 正则权重（原为 0.1）

    # 路径
    SKIN_MASK_DIR = os.path.join("mask_cache")
    GLOBAL_CURVE_TENSOR_PATH = os.path.join(CHECKPOINT_DIR, "global_curve.pt")

    def setup_dirs(self):
        for d in [self.DATA_DIR, self.CACHE_DIR, self.CHECKPOINT_DIR, self.LOG_DIR, self.SKIN_MASK_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)
