import os
from scipy.fftpack import dct, idct
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 输入/输出路径
input_dir = 'adversarial_images'
output_dir = 'denoised_images'
os.makedirs(output_dir, exist_ok=True)

# DCT + IDCT 去噪函数
def dct_idct_process(image_np, keep_ratio=0.1):
    image_np = image_np.astype(np.float32)

    # 二维 DCT
    dct_img = dct(dct(image_np.T, norm='ortho').T, norm='ortho')

    # 保留低频
    h, w = dct_img.shape
    h_keep = int(h * keep_ratio)
    w_keep = int(w * keep_ratio)
    mask = np.zeros_like(dct_img)
    mask[:h_keep, :w_keep] = 1
    dct_img *= mask

    # 二维 IDCT
    idct_img = idct(idct(dct_img.T, norm='ortho').T, norm='ortho')

    return np.clip(idct_img, 0, 255).astype(np.uint8)

# 可视化前几张对比图像
visualize_count = 5
shown = 0

# 遍历图像
for filename in tqdm(os.listdir(input_dir)):
    if not (filename.endswith('.png') or filename.endswith('.jpg')):
        continue

    path = os.path.join(input_dir, filename)
    img = Image.open(path).convert('L')
    img_np = np.array(img)

    # DCT-IDCT 去噪
    denoised_np = dct_idct_process(img_np, keep_ratio=0.1)
    denoised_img = Image.fromarray(denoised_np)

    # 保存去噪图像
    denoised_img.save(os.path.join(output_dir, filename))

    # 前 visualize_count 张图像进行对比展示
    if shown < visualize_count:
        plt.figure(figsize=(8, 4))
        plt.suptitle(f'Comparison: {filename}', fontsize=14)

        plt.subplot(1, 2, 1)
        plt.imshow(img_np, cmap='gray')
        plt.title('Original (Adversarial)')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(denoised_np, cmap='gray')
        plt.title('DCT-IDCT Denoised')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        shown += 1