# 主训练脚本 (包含前向传播、反向传播、模型保存)

# ==========================================
# 伪代码：前向传播与 Loss 计算 (Flow Matching 视角)
# ==========================================

def training_step(batch):
    # 1. 获取数据 (HR: 目标高分辨率, LR: 初始低分辨率)
    x0 = batch['HR']  # Shape: [B, C, H, W]
    y = batch['LR']   # Shape: [B, C, H, W] (经过上采样至与 HR 同尺寸)

    # 2. 采样时间步 t ~ Uniform(0, 1)
    # pMF设定：t=1 是起点(LR), t=0 是终点(HR)
    t = torch.rand(batch_size, 1, 1, 1) 

    # 3. 构造残差位移路径 (Residual Shifting Path) -> 当前状态 z_t
    z_t = (1 - t) * x0 + t * y

    # 4. 条件特征提取 
    # 使用一个小网络提取 LR 的深层特征，并展平为 Sequence
    # c_lr shape: [B, Seq_Len, Dim]
    c_lr = conditioner_network(y) 

    # 5. 模型前向预测
    # pmf_dit 接收 z_t (当前图像), t (时间), c_lr (Cross-Attention 条件)
    # 目标是预测 x0 (即 denoised image / HR image)
    pred_x0 = pmf_dit(x_t=z_t, t=t, context=c_lr)

    # 6. 计算 Flow Matching Loss (MSE 形式)
    loss = MSELoss(pred_x0, x0)
    
    return loss

# ==========================================
# 伪代码：推理采样器 (Euler Solver 视角)
# ==========================================

def sample_step(y, steps=10):
    # 推理时，从 t=1 (LR) 一步步走向 t=0 (HR)
    z_t = y  # 初始状态就是 LR 图像
    c_lr = conditioner_network(y) # 只需要提取一次
    
    dt = 1.0 / steps
    
    for t in reversed(linspace(0, 1, steps)): # t: 1.0 -> 0.0
        # 预测 x0
        pred_x0 = pmf_dit(x_t=z_t, t=t, context=c_lr)
        
        # 根据 pMF 论文，推导速度 v = (z_t - pred_x0) / t
        v = (z_t - pred_x0) / t
        
        # Euler 更新：z_{t-dt} = z_t - v * dt
        z_t = z_t - v * dt
        
    return z_t # 最终的 SR 图像