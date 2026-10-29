# -*- coding: utf-8 -*-
"""
批量生成 person/face mask -> mask_cache/
用法:
    python mask_generator.py [--data-dir data/rgb_pairs] [--mask-cache mask_cache] [--vis]

保存文件:
    mask_cache/{stem}_person.npy
    mask_cache/{stem}_face.npy
其中 stem 是图片名去掉 _input 后缀，例如 "concert_001_input.jpg" -> "concert_001"
"""
import os, sys, glob, urllib.request, argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

SELFIE_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite"
SELFIE_PATH = "selfie_segmenter.tflite"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = "face_detection_yunet_2023mar.onnx"


def download(url, path):
    if not os.path.exists(path):
        print(f"Download {path} ...")
        urllib.request.urlretrieve(url, path)
    return path


def create_person_segmenter():
    download(SELFIE_URL, SELFIE_PATH)
    base = mp_python.BaseOptions(model_asset_path=SELFIE_PATH)
    opt = vision.ImageSegmenterOptions(
        base_options=base,
        running_mode=vision.RunningMode.IMAGE,
        output_confidence_masks=True,
        output_category_mask=True,
    )
    return vision.ImageSegmenter.create_from_options(opt)


def segment_person_mediapipe(segmenter, mp_image):
    """返回与输入图同尺寸的 person mask，float32，0/1"""
    res = segmenter.segment(mp_image)
    h, w = mp_image.numpy_view().shape[:2]
    if getattr(res, "confidence_masks", None):
        conf = res.confidence_masks[0].numpy_view().squeeze()
        mask = (conf > 0.5).astype(np.float32)
    else:
        cat = res.category_mask.numpy_view().squeeze()
        uniq, cnts = np.unique(cat, return_counts=True)
        person_label = uniq[np.argsort(cnts)[::-1][1]] if len(uniq) >= 2 else uniq[0]
        mask = (cat == person_label).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = (mask > 0.5).astype(np.float32)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)


def segment_face_yunet(rgb):
    """YuNet(BGR) + 五点关键点椭圆。返回 face_mask (H,W) float32，无脸则全零。"""
    download(YUNET_URL, YUNET_PATH)
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    detector = cv2.FaceDetectorYN_create(YUNET_PATH, "", (320, 320), 0.6, 0.3, 5000)
    detector.setInputSize((w, h))
    _, faces = detector.detect(bgr)
    face_mask = np.zeros((h, w), dtype=np.float32)

    if faces is None or len(faces) == 0:
        print(f"[Face] {w}x{h} 未检测到人脸")
        return face_mask

    best = max(faces, key=lambda f: f[2] * f[3])
    landmarks = best[4:14].reshape(5, 2)

    right_eye, left_eye = landmarks[0], landmarks[1]
    eye_center = (right_eye + left_eye) / 2.0
    eye_dist = np.linalg.norm(right_eye - left_eye)
    face_width = max(eye_dist * 2.6, best[2] * 0.8)

    nose = landmarks[2]
    mouth_center = (landmarks[3] + landmarks[4]) / 2.0
    face_height = np.linalg.norm(mouth_center - eye_center) * 2.4

    center = (int(eye_center[0]), int(eye_center[1] + face_height * 0.08))
    axes = (int(face_width * 0.5), int(face_height * 0.5))

    cv2.ellipse(face_mask, center, axes, 0, 0, 360, 1.0, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    face_mask = cv2.erode(face_mask, kernel, iterations=1)
    return face_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/rgb_pairs")
    parser.add_argument("--mask-cache", default="mask_cache")
    parser.add_argument("--vis", action="store_true", help="保存可视化调试图")
    args = parser.parse_args()

    os.makedirs(args.mask_cache, exist_ok=True)
    if args.vis:
        os.makedirs(os.path.join(args.mask_cache, "vis"), exist_ok=True)

    input_files = sorted(glob.glob(os.path.join(args.data_dir, "*_input.*")))
    if not input_files:
        print(f"未找到 *_input.* 文件，请检查 data_dir: {args.data_dir}")
        sys.exit(1)

    print(f"共 {len(input_files)} 张输入图")
    mp_img_format = mp.ImageFormat.SRGB
    with create_person_segmenter() as segmenter:
        for inp_path in tqdm(input_files, desc="生成 mask"):
            stem = os.path.splitext(os.path.basename(inp_path))[0].replace("_input", "")
            person_out = os.path.join(args.mask_cache, f"{stem}_person.npy")
            face_out = os.path.join(args.mask_cache, f"{stem}_face.npy")
            if os.path.exists(person_out) and os.path.exists(face_out):
                continue

            rgb = np.array(Image.open(inp_path).convert("RGB"), dtype=np.uint8)
            h, w = rgb.shape[:2]

            person = segment_person_mediapipe(segmenter, mp.Image(image_format=mp_img_format, data=rgb))
            face = segment_face_yunet(rgb)
            # 确保 face 在 person 内部
            face = np.logical_and(face > 0.5, person > 0.5).astype(np.float32)

            np.save(person_out, person)
            np.save(face_out, face)

            if args.vis:
                vis = rgb.copy()
                vis[person > 0.5] = (vis[person > 0.5] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
                vis[face > 0.5] = (vis[face > 0.5] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
                Image.fromarray(vis).save(os.path.join(args.mask_cache, "vis", f"{stem}_mask_vis.png"))

    print(f"✅ 全部完成，mask 已存到 {args.mask_cache}")


if __name__ == "__main__":
    main()