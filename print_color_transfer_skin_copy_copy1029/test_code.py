import numpy as np
from processor import process_image_pair, load_model

model = load_model("checkpoints/curve_pred_best.pth")

out, metrics = process_image_pair(
    image_path="/workspace/meng_xf/preprint/preprint_data/clean/5D2A1088_input.jpg",
    model=model,
    output_dir="test_out",
    skin_model_path="checkpoints/skin_curve_best.pth",
    return_metrics=True,
)
print("输出:", out)
print("metrics:", metrics)






import numpy as np
import cv2
from PIL import Image

a = np.array(Image.open("test_out/5D2A1088_input_output.jpg").convert("RGB"), dtype=np.float32) / 255.0
b = np.array(Image.open("test_out/5D2A1088_input_skin_output.jpg").convert("RGB"), dtype=np.float32) / 255.0

bgr = (a[:, :, ::-1] * 255).astype(np.uint8)
ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
Cr = ycrcb[:, :, 1].astype(np.float32)
Cb = ycrcb[:, :, 2].astype(np.float32)
skin = ((Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127))

diff = np.abs(a - b).mean(axis=2)
print(f"全图像素:          {diff.mean()*255:.2f} / 255")
print(f"肤色像素:          {diff[skin].mean()*255:.2f} / 255  (共 {skin.sum()} 像素)")
print(f"肤色区域 Δ>5:      {(diff[skin] > 5/255).mean()*100:.1f}%")
print(f"肤色区域 Δ>10:     {(diff[skin] > 10/255).mean()*100:.1f}%")