import os
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torchvision import transforms

# 导入我们自己写的模块
from models.pmf_dit import pmf_DiT
from models.conditioner import LRConditioner
from models.embedder import TimestepEmbedder
from models.flow_matching import ResShiftFlowMatcher
from utils.solver import EulerSolver

# 允许读取部分损坏但仍可解码的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 支持的图片扩展名
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def crop_patch(img, left, top, patch_size=256):
    """
    从原图中裁剪一个 patch。
    """
    w, h = img.size
    if left < 0 or top < 0:
        raise ValueError(f"裁剪坐标不能为负数: left={left}, top={top}")
    if left + patch_size > w or top + patch_size > h:
        raise ValueError(
            f"裁剪区域越界: 图像大小=({w}, {h}), "
            f"请求区域=({left}, {top}, {left+patch_size}, {top+patch_size})"
        )
    return img.crop((left, top, left + patch_size, top + patch_size))

def load_aligned_patches(image_path, crop_x=0, crop_y=0, patch_size=256, scale=4):
    """
    全新对齐逻辑：
    1. 直接从高清原图中切出 256x256 的 Ground Truth (GT)
    2. 将 GT 缩小 4 倍 (模拟真实的 LR 传感器获取)
    3. 将缩小的图放大回 256x256 (模型所需要的模糊输入)
    """
    img = Image.open(image_path).convert("RGB")

    # 1. 获取 256x256 的 Ground Truth
    gt_patch = crop_patch(img, crop_x, crop_y, patch_size=patch_size)

    # 2. 模拟真实世界的低分辨率退化 (256 -> 64)
    lr_w, lr_h = patch_size // scale, patch_size // scale
    lr_small = gt_patch.resize((lr_w, lr_h), resample=Image.Resampling.BICUBIC)

    # 3. 放大回模型所需尺寸 (64 -> 256)
    lr_bicubic = lr_small.resize((patch_size, patch_size), resample=Image.Resampling.BICUBIC)

    # 4. 转 Tensor 并归一化到 [-1, 1]
    tensor = transforms.ToTensor()(lr_bicubic)  
    tensor = tensor * 2.0 - 1.0                  

    return tensor.unsqueeze(0), gt_patch, lr_bicubic

def save_image(tensor, save_path):
    """
    将 [-1, 1] 的 Tensor 转回图像并保存
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img_tensor = (tensor.squeeze(0).detach().cpu().clamp(-1, 1) + 1.0) / 2.0
    img = transforms.ToPILImage()(img_tensor)
    img.save(save_path, format="PNG")

def main():
    # ==========================================
    # 1. 基础配置
    # ==========================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sampling_steps = 1  # 采样步数，后续可加速到 1 步

    input_path = "test/test02/GroundTruth.jpg"
    out_dir = f"test/test02/{sampling_steps}_step"
    os.makedirs(out_dir, exist_ok=True)

    scale = 4
    patch_size = 256  # 我们直接操作 256x256 的物理尺寸

    crop_x = 128
    crop_y = 128

    print(f"初始化完美对齐版推理 | 设备: {device} | 步数: {sampling_steps}")

    # ==========================================
    # 2. 组装模型
    # ==========================================
    hidden_size = 768

    backbone = pmf_DiT(in_channels=3, patch_size=4, hidden_size=hidden_size, depth=12, num_heads=12)
    conditioner = LRConditioner(in_channels=3, hidden_size=hidden_size, num_blocks=4)
    embedder = TimestepEmbedder(hidden_size=hidden_size)

    flow_matcher = ResShiftFlowMatcher(backbone, conditioner, embedder).to(device)

    ckpt_path = "experiments/run_20260418_221940_BF16/checkpoints/pmf_resshift_epoch_100.pth"
    flow_matcher.load_state_dict(torch.load(ckpt_path, map_location=device))
    flow_matcher.eval()
    print("模型权重加载完毕")

    # ==========================================
    # 3. 读取数据与推理
    # ==========================================
    if not os.path.exists(input_path):
        print(f"找不到测试图片: {input_path}")
        return

    # 拿到完美对齐的张量和 PIL 图像
    lr_tensor, gt_patch_img, lr_bicubic_img = load_aligned_patches(
        input_path, crop_x=crop_x, crop_y=crop_y, patch_size=patch_size, scale=scale
    )

    # 把对齐好的 GT 和 LR 先存下来
    gt_patch_img.save(os.path.join(out_dir, "01_GroundTruth_256.png"))
    lr_bicubic_img.save(os.path.join(out_dir, "02_LR_Bicubic_256.png"))
    print(f"已保存 256x256 的 Ground Truth 与 Bicubic 模糊图供对比")

    lr_tensor = lr_tensor.to(device)
    solver = EulerSolver(flow_matcher)

    print("正在沿残差位移路径生成高清细节...")
    with torch.no_grad():
        hr_tensor = solver.sample(lr_tensor, steps=sampling_steps)

    # ==========================================
    # 4. 保存超分结果
    # ==========================================
    output_path = os.path.join(out_dir, "03_HR_ModelPred_256.png")
    save_image(hr_tensor, output_path)
    print(f"成功！三张对齐的 256x256 图像已保存在 {out_dir} 文件夹下。")

if __name__ == "__main__":
    main()