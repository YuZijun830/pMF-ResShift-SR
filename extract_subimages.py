import os
import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

# 防止线程过量竞争
cv2.setNumThreads(1)

def process_image(args):
    img_name, input_folder, save_folder, crop_size, step = args

    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

    h, w = img.shape[0:2]

        # 如果原图比裁剪尺寸还小，直接跳过
    if h < crop_size or w < crop_size:
        print(f"Image too small, skipped: {img_path} ({h}x{w})")
        return 0
    
    h_space = np.arange(0, h - crop_size + 1, step)
    w_space = np.arange(0, w - crop_size + 1, step)
    
    # 保证边缘也能切到
    if h - (h_space[-1] + crop_size) > 0:
        h_space = np.append(h_space, h - crop_size)
    if w - (w_space[-1] + crop_size) > 0:
        w_space = np.append(w_space, w - crop_size)

    idx = 0
    saved_count = 0

    for y in h_space:
        for x in w_space:
            idx += 1
            y = int(y)
            x = int(x)
            cropped_img = img[y:y + crop_size, x:x + crop_size, ...]
            # 舍弃那些几乎全是纯色（如纯黑/纯白背景）的无效块
            if cropped_img.var() < 100:
                continue
            
            # 保存小图，名字类似 0001_s001.png
            save_name = f"{img_name.split('.')[0]}_s{idx:03d}.png"
            save_path = os.path.join(save_folder, save_name)
            cv2.imwrite(save_path, cropped_img)
            saved_count += 1

    return saved_count

def main():
    # 原始大图路径
    input_folder = './data/DIV2K/original/DIV2K_train_HR'
    # 切割后小图的保存路径
    save_folder = './data/DIV2K/DIV2K_train_HR_sub'
    
    # 裁剪参数：切成 480x480 的小块，步长 240（有一半重叠，增加数据量）
    crop_size = 480
    step = 240
    
    os.makedirs(save_folder, exist_ok=True)
    
    img_list = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg'))]
    print(f"找到 {len(img_list)} 张原始大图，开始切割...")

    tasks = [(img_name, input_folder, save_folder, crop_size, step) for img_name in img_list]

    total_saved = 0
    num_workers = min(16, os.cpu_count() or 1)

    # 开启多进程加速切割
    with Pool(num_workers) as pool:
        for saved in tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks)):
            total_saved += saved

    print(f"切割完成！共保存 {total_saved} 张子图，路径: {save_folder}")


if __name__ == '__main__':
    main()