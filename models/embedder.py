# Patch Embedding 和 Timestep Embedding

import torch
import torch.nn as nn
import numpy as np
import math

# ==========================================
# 组件 1：时间步嵌入器 (Timestep Embedder)
# ==========================================
class TimestepEmbedder(nn.Module):
    """
    将连续标量时间步 t (通常在 [0, 1] 之间) 映射为高维特征。
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        创建一个正弦/余弦的频率嵌入。
        t: [BatchSize] 的 1D Tensor
        dim: 嵌入维度
        """
        # 防止除 0 错误并计算频率
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        # args: [BatchSize, half]
        args = t[:, None].float() * freqs[None, :]
        
        # 拼接 cos 和 sin
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # 1. 将时间标量 t 转为基础的频率向量
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # 2. 通过 MLP 提取更深层的特征，输出形状: [B, hidden_size]
        t_emb = self.mlp(t_freq)
        return t_emb

# ==========================================
# 组件 2：二维正弦位置编码生成器 (2D Positional Embedding)
# ==========================================
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    辅助函数：为 1D 坐标生成正弦/余弦编码。
    pos: 一个包含坐标的一维数组
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= (embed_dim / 2.)
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2) 外积

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    为主干网络生成 2D 绝对位置编码。
    grid_size: (H_patch, W_patch)，即图像在长宽上各有几个 Patch
    返回: [H_patch * W_patch, embed_dim] 形状的 Tensor
    """
    grid_h, grid_w = grid_size
    
    # 建立 2D 坐标网格
    grid_h_idx = np.arange(grid_h, dtype=np.float32)
    grid_w_idx = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w_idx, grid_h_idx)  # (2, H, W)
    
    # 对于 x 轴和 y 轴，各分一半的 embed_dim
    assert embed_dim % 2 == 0
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    # 拼接起来，形成完整的 2D 位置特征
    pos_embed = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return torch.from_numpy(pos_embed).float()