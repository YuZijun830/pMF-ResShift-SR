# 推理采样器 (Euler / RK4 ODE Solver)

import torch
from tqdm import tqdm

class EulerSolver:
    """
    基于欧拉方法的常微分方程 (ODE) 求解器。
    用于在像素空间中，沿着 pMF-ResShift 构建的恒速直线路径，将 LR 图像演化为 HR 图像。
    """
    def __init__(self, model):
        """
        model: 我们在 flow_matching.py 中编写的 ResShiftFlowMatcher 实例
        """
        self.model = model

    @torch.no_grad()
    def sample(self, lr_img, steps=10, return_intermediates=False):
        """
        lr_img: 输入的低分辨率图像 [B, C, H, W]，已通过 Bicubic 插值到目标分辨率。
                取值范围必须是 [-1, 1]。
        steps: 采样步数 (NFE: Number of Function Evaluations)。
               对于 ResShift 路径，通常 4~15 步即可获得极佳效果。
               pMF  加速到 1 步
        """
        self.model.eval()
        device = lr_img.device
        b = lr_img.shape[0]

        # 1. 初始状态：t=1 时的图像就是 LR 本身
        z_t = lr_img.clone()
        
        # 2. 生成时间步序列：从 1.0 均匀地递减到 0.0
        # 例如 steps=10，timesteps 为 [1.0, 0.9, 0.8, ..., 0.0]
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)
        
        intermediates = [z_t.cpu()] if return_intermediates else []

        # 3. 开始 ODE 迭代 (从 t 走向 t-dt)
        for i in tqdm(range(steps), desc="Euler ODE Sampling"):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_curr - t_next  # 步长 (正数，例如 0.1)

            # 构造当前时间的 batch 张量
            t_tensor = t_curr.expand(b)

            # 调用模型推理速度场和预测的高清图 (在 flow_matching.py 中定义的)
            # v = lr_img - pred_x0
            v, pred_x0 = self.model.infer_velocity(z_t, t_tensor, lr_img)

            # 4. 欧拉更新
            # 如果是最后一步 (t_next == 0)，为了消除截断误差，直接返回模型的终点预测 pred_x0
            if i == steps - 1:
                z_t = pred_x0
            else:
                # 核心公式：z_{t-dt} = z_t - v * dt
                z_t = z_t - v * dt
                
            if return_intermediates:
                intermediates.append(z_t.cpu())

        # 确保输出被裁剪到合法的像素值域 [-1, 1] 内
        z_t = torch.clamp(z_t, -1.0, 1.0)

        if return_intermediates:
            return z_t, intermediates
        return z_t