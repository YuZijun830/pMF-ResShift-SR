# DiT Backbone (支持 Cross-Attention)

import torch
import torch.nn as nn
from einops import rearrange
import math

# 引入我们在 embedder.py 中写好的位置编码生成器
from .embedder import get_2d_sincos_pos_embed

# ==========================================
# 辅助函数：时间步条件调制 (adaLN)
# ==========================================
def modulate(x, shift, scale):
    """
    根据时间步 t 调制特征。
    x: Transformer 内部的特征 [B, L, D]
    shift, scale: 从时间步 t 映射过来的偏移和缩放尺度 [B, D]
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# ==========================================
# 核心组件 1：DiT Block (支持 Cross-Attention)
# ==========================================
class DiTCrossBlock(nn.Module):
    """
    包含 Self-Attention, Cross-Attention, MLP 和 AdaLN 的 Transformer 块。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 1. AdaLN 层：用于注入时间步 t 的信息
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Self-Attention
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # 2. Cross-Attention 层：用于注入 LR 图像特征 c_lr
        self.norm2 = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # 3. MLP 层
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        # 4. AdaLN 调制参数生成器：输入时间特征，输出 6 个调制参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, t_emb, context):
        """
        x: 当前图像的 token 序列 [B, SeqLen, D]
        t_emb: 时间步特征 [B, D]
        context: LR 图像特征 [B, Context_SeqLen, D]
        """
        # 生成时间步调制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        
        # 1. Self-Attention 模块 (带有时间调制)
        x_modulated = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_modulated, x_modulated, x_modulated, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # 2. Cross-Attention 模块 (向 LR 图像查询信息)
        # Query: 当前图像; Key, Value: LR 图像特征 (context)
        cross_out, _ = self.cross_attn(self.norm2(x), context, context, need_weights=False)
        x = x + cross_out  # 此处也可以加 gate 参数，为保持简单我们使用残差
        
        # 3. MLP 模块 (带有时间调制)
        x_modulated_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_modulated_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

# ==========================================
# 核心组件 2：图像 Patch 分块与还原
# ==========================================
class PatchEmbed(nn.Module):
    """利用卷积将图像划分为 4x4 的 Patch 并转换为序列"""
    def __init__(self, patch_size=4, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, D, H/P, W/P] -> [B, H/P*W/P, D]
        x = self.proj(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x

# ==========================================
# 核心组件 3：pMF-ResShift 主干网络
# ==========================================
class pmf_DiT(nn.Module):
    def __init__(
        self,
        in_channels=3,
        patch_size=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)

        self.blocks = nn.ModuleList([
            DiTCrossBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, patch_size * patch_size * in_channels, bias=True)
        )
        
        # 用于缓存位置编码，避免每次 forward 都重复计算 NumPy
        self.pos_embed_cache = {}

    def get_dynamic_pos_embed(self, h, w, device):
        """动态获取 2D 位置编码，支持可变分辨率"""
        grid_size = (h // self.patch_size, w // self.patch_size)
        
        # 如果缓存中没有当前分辨率的位置编码，则生成并缓存
        if grid_size not in self.pos_embed_cache:
            pos_embed = get_2d_sincos_pos_embed(self.hidden_size, grid_size)
            # pos_embed shape: [SeqLen, hidden_size]
            self.pos_embed_cache[grid_size] = pos_embed.unsqueeze(0) # 加上 Batch 维度: [1, SeqLen, D]
            
        # 从缓存读取并移动到对应设备
        return self.pos_embed_cache[grid_size].to(device)

    def forward(self, x, t_emb, context):
        b, c, h, w = x.shape
        
        # 1. 图像转序列 (x_seq shape: [B, SeqLen, D])
        x_seq = self.x_embedder(x) 
        
        # 2. 注入动态 2D 绝对位置编码
        pos_embed = self.get_dynamic_pos_embed(h, w, x.device)
        x_seq = x_seq + pos_embed  # Broadcasting over Batch dimension
        
        # 3. Transformer 前向传播
        for block in self.blocks:
            x_seq = block(x_seq, t_emb, context)
            
        # 4. 预测输出 (回归像素)
        x_seq = self.final_layer(x_seq)
        
        # 5. 序列还原为图像
        p = self.patch_size
        hp, wp = h // p, w // p
        out = rearrange(x_seq, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=hp, w=wp, p1=p, p2=p)
        
        return out