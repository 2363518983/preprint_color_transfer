# -*- coding: utf-8 -*-
"""离线缓存：把原始大图中心裁剪 + resize 到 512，存为 float16 npy。

训练前先跑一次（约 10-20 分钟），之后训练直接读小图，速度提升 ~10x。
"""
import os
import sys
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import center_crop_square, _file_key


def main():
    from config import Config
    cfg = Config()
    data_dir = cfg.DATA_DIR
    cache_dir = cfg.CACHE_DIR
    img_size = cfg.IMG_SIZE
    os.makedirs(cache_dir, exist_ok=True)

    exts = ('jpg', 'jpeg', 'png', 'tif', 'tiff')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(data_dir, f'*_input.{ext}')))
        files.extend(glob.glob(os.path.join(data_dir, f'*_target.{ext}')))
    files = sorted(set(files))
    print(f"共 {len(files)} 个文件，输出尺寸 {img_size}x{img_size}，缓存目录 {cache_dir}")

    for path in tqdm(files):
        key = _file_key(path)
        suffix = 'input' if '_input.' in path else 'target'
        out = os.path.join(cache_dir, f"{key}_{suffix}.npy")
        if os.path.exists(out):
            continue
        img = Image.open(path).convert('RGB')
        img = center_crop_square(img).resize((img_size, img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        np.save(out, arr.astype(np.float16))

    print("✅ 缓存完成")


if __name__ == '__main__':
    main()
