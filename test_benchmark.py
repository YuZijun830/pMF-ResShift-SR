# python test_benchmark.py \
#     --input_dir ./data/benchmark/benchmark/Set5/LR_bicubic/X4 \
#     --output_dir ./results/Set5_SR \
#     --ckpt_path ./experiments/run_20260418_221940_BF16/checkpoints/pmf_resshift_epoch_100.pth \
#     --sampling_steps 10

import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from PIL import Image, ImageFile
from torchvision import transforms

# 导入我们自己写的模块
from models.pmf_dit import pmf_DiT
from models.conditioner import LRConditioner
from models.embedder import TimestepEmbedder
from models.flow_matching import ResShiftFlowMatcher
from utils.solver import EulerSolver

ImageFile.LOAD_TRUNCATED_IMAGES = True
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def parse_args():
    parser = argparse.ArgumentParser(description="pMF-ResShift 批量测试脚本")
    parser.add_argument("--input_dir", type=str, required=True, help="存放低清测试图的文件夹")
    parser.add_argument("--output_dir", type=str, required=True, help="高清结果保存的文件夹")
    parser.add_argument("--ckpt_path", type=str, required=True, help="训练好的模型权重路径")
    parser.add_argument("--sampling_steps", type=int, default=10, help="ODE 采样步数")
    parser.add_argument("--scale", type=int, default=4, help="超分放大倍数")
    return parser.parse_args()

def save_image(tensor, save_path):
    save_path = Path(save_path)
    # 将 [-1, 1] 的 Tensor 转回 [0, 1] 并保存
    img_tensor = (tensor.squeeze(0).detach().cpu().float().clamp(-1, 1) + 1.0) / 2.0
    img = transforms.ToPILImage()(img_tensor)
    img.save(save_path, format="PNG")

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"初始化批量推理流程 | 设备: {device} | 步数: {args.sampling_steps}")

    # ==========================================
    # 1. 组装模型与加载权重
    # ==========================================
    hidden_size = 768
    backbone = pmf_DiT(in_channels=3, patch_size=4, hidden_size=hidden_size, depth=12, num_heads=12)
    conditioner = LRConditioner(in_channels=3, hidden_size=hidden_size, num_blocks=4)
    embedder = TimestepEmbedder(hidden_size=hidden_size)

    flow_matcher = ResShiftFlowMatcher(backbone, conditioner, embedder).to(device)
    
    # 兼容 DDP 保存的权重 (带有 module. 前缀)
    state_dict = torch.load(args.ckpt_path, map_location=device, weights_only=True)
    if "module." in list(state_dict.keys())[0]:
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
    flow_matcher.load_state_dict(state_dict)
    flow_matcher.eval()
    print("模型权重加载完毕")

    solver = EulerSolver(flow_matcher)

    # ==========================================
    # 2. 遍历文件夹并执行超分
    # ==========================================
    img_paths = [p for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    
    if len(img_paths) == 0:
        print(f"在 {args.input_dir} 中没有找到任何支持的图片！")
        return
        
    print(f"找到 {len(img_paths)} 张测试图片，开始极速推理...")

    # 开启无梯度模式，并使用 BF16 混合精度大幅节省显存和提速
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for img_path in tqdm(img_paths, desc="Processing"):
            try:
                # 读取 LR 图片 (比如 128x128)
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                
                # 先用 Bicubic 放大到目标尺寸 (比如 512x512)，作为 Flow Matching 的起点
                sr_w, sr_h = w * args.scale, h * args.scale
                model_input = img.resize((sr_w, sr_h), resample=Image.Resampling.BICUBIC)
                
                # 转为 Tensor 供模型使用 [-1, 1]
                lr_tensor = transforms.ToTensor()(model_input).to(device)
                lr_tensor = lr_tensor * 2.0 - 1.0
                lr_tensor = lr_tensor.unsqueeze(0) # 增加 Batch 维度

                # ODE 求解生成
                hr_tensor = solver.sample(lr_tensor, steps=args.sampling_steps)

                # 保存结果 (保持原文件名)
                save_path = os.path.join(args.output_dir, img_path.name)
                save_image(hr_tensor, save_path)
                
            except Exception as e:
                print(f"处理 {img_path.name} 时出错: {e}")

    print(f"全部处理完成！高清结果已保存在: {args.output_dir}")

if __name__ == "__main__":
    main()