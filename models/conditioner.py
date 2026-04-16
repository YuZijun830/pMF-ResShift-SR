# LR 特征提取器 (将 y 转换为序列特征送入 Cross-Attn)

import torch
import torch.nn as nn
from einops import rearrange

# ==========================================
# 辅助模块：简单的残差卷积块
# ==========================================
class SimpleResBlock(nn.Module):
    """
    用于提取 LR 图像局部空间特征的 CNN 残差块。
    相较于直接使用 Linear 层，卷积能更好地保留图像的边缘和结构信息。
    """
    def __init__(self, channels):
        super().__init__()
        # 使用 3x3 卷积，保持特征图分辨率不变
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU() # 使用 SiLU (Swish) 激活函数，与 DiT 内部保持一致
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 残差连接
        return x + self.conv2(self.act(self.conv1(x)))

# ==========================================
# 核心模块：LR 特征提取器 (Conditioner)
# ==========================================
class LRConditioner(nn.Module):
    def __init__(self, in_channels=3, hidden_size=768, num_blocks=4):
        """
        in_channels: 输入 LR 图像的通道数 (通常为 3)
        hidden_size: 必须与 pmf_DiT 的 hidden_size 保持一致，以便进行 Cross-Attention
        num_blocks: 残差块的数量，控制特征提取的深度
        """
        super().__init__()
        
        # 为了计算效率，我们可以在初始阶段先映射到一个较小的维度 (如 hidden_size // 4)
        cnn_dim = hidden_size // 4
        
        # 1. 初始特征提取
        self.init_conv = nn.Conv2d(in_channels, cnn_dim, kernel_size=3, padding=1)
        
        # 2. 深度特征提取 (堆叠残差块)
        self.blocks = nn.Sequential(*[
            SimpleResBlock(cnn_dim) for _ in range(num_blocks)
        ])
        
        # 3. 维度对齐与激活
        # 将特征通道数提升至 DiT 的 hidden_size (例如 768)
        self.final_conv = nn.Conv2d(cnn_dim, hidden_size, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, lr_img):
        """
        lr_img: 经过预处理的低分辨率图像，通常已经通过插值 (Bicubic) 
                上采样到了与 HR 相同的物理尺寸。
                Shape: [B, C, H, W]
        """
        # 1. 提取 2D 空间特征
        x = self.init_conv(lr_img)
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.act(x)  # Shape: [B, hidden_size, H, W]
        
        # 2. 展平为 1D 序列 (Sequence)
        # 将空间维度 (H, W) 展平为序列长度 (H * W)
        # 这样它就可以作为 Cross-Attention 中的 Key 和 Value 传入 DiT
        # Shape 转换: [B, D, H, W] -> [B, (H*W), D]
        x_seq = rearrange(x, 'b d h w -> b (h w) d')
        
        return x_seq