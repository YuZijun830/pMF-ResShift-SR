# 流匹配的核心逻辑 (负责构建 z_t 轨迹和计算 Loss)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResShiftFlowMatcher(nn.Module):
    """
    pMF-ResShift 的核心大脑：负责构建残差移动路径并计算 Flow Matching Loss。
    它将作为高层 Wrapper，把我们之前写的网络组件拼装起来。
    """
    def __init__(self, model, conditioner, embedder):
        """
        model: 主干网络 (pmf_DiT)
        conditioner: LR 图像特征提取网络 (LRConditioner)
        embedder: 时间步嵌入网络 (TimestepEmbedder)
        """
        super().__init__()
        self.model = model
        self.conditioner = conditioner
        self.embedder = embedder

    def forward(self, hr_img, lr_img):
        """
        训练时的前向传播函数。
        hr_img: Ground Truth 高分辨率图像 (x0), Shape: [B, C, H, W]
        lr_img: 输入的低分辨率图像 (y) (已插值到HR尺寸), Shape: [B, C, H, W]
        
        返回:
            loss: 当前 batch 的均方误差 (MSE)
        """
        b, c, h, w = hr_img.shape
        device = hr_img.device

        # ==========================================
        # 步骤 1：时间步采样
        # 从均匀分布 U(0, 1) 中随机采样时间步 t
        # ==========================================
        t = torch.rand((b,), device=device)
        
        # 将 t 的形状从 [B] 扩展为 [B, 1, 1, 1]，以便与图像进行广播(Broadcast)计算
        t_expanded = t.view(b, 1, 1, 1)

        # ==========================================
        # 步骤 2：构建 ODE 残差轨迹 (Residual Shifting Path)
        # 物理直觉：当 t=1 时 z_t 完全是 LR；当 t=0 时 z_t 完全是 HR
        # ==========================================
        z_t = (1.0 - t_expanded) * hr_img + t_expanded * lr_img

        # ==========================================
        # 步骤 3：特征条件提取与时间步嵌入
        # ==========================================
        # 将 LR 图像转化为用于 Cross-Attention 的特征序列 (Context)
        # context shape: [B, SeqLen, D]
        with torch.no_grad():
            # 提示：如果你希望 Conditioner 也参与端到端训练，去掉 no_grad 即可。
            # 通常我们是一起端到端训练的，所以这里可以直接调用
            pass
        context = self.conditioner(lr_img)
        
        # 将标量 t 转化为高维的时间步特征向量
        # t_emb shape: [B, D]
        t_emb = self.embedder(t)

        # ==========================================
        # 步骤 4：模型预测
        # pMF 架构的精髓：给定当前路径点 z_t，预测目标 x0
        # ==========================================
        pred_x0 = self.model(x=z_t, t_emb=t_emb, context=context)

        # ==========================================
        # 步骤 5：计算 Loss (目标匹配)
        # Flow Matching 通常使用简单的 L2 Loss (MSE)
        # 也可以考虑改成v-loss、mse前面加t权重系数
        # ==========================================
        loss = F.mse_loss(pred_x0, hr_img)

        return loss

    @torch.no_grad()
    def infer_velocity(self, z_t, t, lr_img):
        """
        这是一个推理辅助函数。
        根据 pMF 的数学框架，虽然模型预测的是 x0，
        但 ODE 求解器需要的是速度场 v。
        根据 z_t = (1-t)x0 + t y 
        可推导出速度 v = dy/dt = y - x0
        """
        # 1. 提取条件
        context = self.conditioner(lr_img)
        t_emb = self.embedder(t)
        
        # 2. 预测 x0
        pred_x0 = self.model(x=z_t, t_emb=t_emb, context=context)
        
        # 3. 计算并返回理论速度场 v
        v = lr_img - pred_x0
        return v, pred_x0