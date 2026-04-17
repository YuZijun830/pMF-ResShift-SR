# 推理脚本 (给定 LR 图像，输出 SR 结果)

import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

# 导入我们自己写的模块
from models.pmf_dit import pmf_DiT
from models.conditioner import LRConditioner
from models.embedder import TimestepEmbedder
from models.flow_matching import ResShiftFlowMatcher
from utils.solver import EulerSolver

def load_image(image_path, scale=4):
    """读取图片并预处理为 [-1, 1] 的 Tensor"""
    # TODO:可以改成自动识别分辨率/图像大小，自动对齐

    img = Image.open(image_path).convert('RGB')
    
    # 模拟 LR 预处理：假设输入的已经是低清小图，我们需要先把它用 Bicubic 放大 scale 倍
    # (如果你的输入图已经是放大过的模糊图，把这两行注释掉即可)
    w, h = img.size
    img = img.resize((w * scale, h * scale), resample=Image.Resampling.BICUBIC)
    
    # 转为 Tensor 并归一化到 [-1, 1]
    tensor = transforms.ToTensor()(img) # [0, 1]
    tensor = tensor * 2.0 - 1.0         # [-1, 1]
    return tensor.unsqueeze(0)          # 加上 Batch 维度 [1, 3, H*scale, W*scale]

def save_image(tensor, save_path):
    """将 [-1, 1] 的 Tensor 转回图像并保存"""
    tensor = (tensor.squeeze(0).clamp(-1, 1) + 1.0) / 2.0 # 映射回 [0, 1]
    img = transforms.ToPILImage()(tensor)
    img.save(save_path)
    print(f"成功保存超分辨率结果至: {save_path}")

def main():
    # ==========================================
    # 1. 基础配置
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_path = "test_lr.png"   # 测试图片路径
    output_path = "output_hr.png" # 保存路径
    sampling_steps = 10          # 推理步数 (ResShift 只需极少步数, pMF只需要 1 步)
    
    print(f"初始化 pMF-ResShift 推理流程 | 设备: {device} | 步数: {sampling_steps}")

    # ==========================================
    # 2. 组装模型大厦
    # ==========================================
    # 参数需与训练时保持绝对一致
    hidden_size = 768
    
    backbone = pmf_DiT(in_channels=3, patch_size=4, hidden_size=hidden_size, depth=12, num_heads=12)
    conditioner = LRConditioner(in_channels=3, hidden_size=hidden_size, num_blocks=4)
    embedder = TimestepEmbedder(hidden_size=hidden_size)
    
    flow_matcher = ResShiftFlowMatcher(backbone, conditioner, embedder).to(device)
    
    # [模拟] 加载预训练权重 (实际使用时取消注释)
    # ckpt_path = "checkpoints/best_model.pth"
    # flow_matcher.load_state_dict(torch.load(ckpt_path, map_location=device))
    # print("模型权重加载完毕")
    
    # ==========================================
    # 3. 读取数据与推理
    # ==========================================
    if not os.path.exists(input_path):
        print(f"找不到测试图片 {input_path}，请准备一张低清图片命名为 {input_path}")
        return

    lr_tensor = load_image(input_path, scale=4).to(device)
    
    # 实例化我们的 Euler 求解器
    solver = EulerSolver(flow_matcher)
    
    # 见证魔法：运行 ODE Solver
    print("正在沿残差位移路径生成高清细节...")
    hr_tensor = solver.sample(lr_tensor, steps=sampling_steps)
    
    # 4. 保存结果
    save_image(hr_tensor, output_path)

if __name__ == "__main__":
    main()