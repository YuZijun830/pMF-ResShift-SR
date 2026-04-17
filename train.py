# 主训练脚本 (包含前向传播、反向传播、模型保存)

import os
from contextlib import nullcontext

import datetime
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.pmf_dit import pmf_DiT
from models.conditioner import LRConditioner
from models.embedder import TimestepEmbedder
from models.flow_matching import ResShiftFlowMatcher
from data.dataset import get_dataloader
from utils.logger import SRLogger  # <-- 新增的 Logger

def main():
    # ==========================================
    # 1. 实验目录与 Logger 初始化
    # ==========================================
    # 生成时间戳 (例如: 20260417_153022)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"run_{current_time}"
    
    # 实验主目录设为 ./experiments/run_xxx/
    exp_dir = os.path.join("./experiments", exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")  # 权重存放在这里
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 初始化 Logger
    logger = SRLogger(exp_dir)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    logger.info(f"启动 pMF-ResShift 训练流程 | 运行设备: {device}")
    logger.info(f"本次实验目录: {exp_dir}")

    # ==========================================
    # 2. 超参数与配置
    # ==========================================
    hr_data_dir = "./data/train_hr"  
    lr_data_dir = None               # 如果没有现成的 LR，设为 None 自动生成
    
    batch_size = 4         # 根据显存随时调整
    patch_size = 256       
    epochs = 500
    learning_rate = 2e-4
    save_every_epochs = 10 

    # 模型架构参数
    hidden_size = 768
    num_heads = 12
    depth = 12

    # ==========================================
    # 3. 组装模型大厦
    # ==========================================
    logger.info("正在初始化网络结构...")
    backbone = pmf_DiT(in_channels=3, patch_size=4, hidden_size=hidden_size, depth=depth, num_heads=num_heads)
    conditioner = LRConditioner(in_channels=3, hidden_size=hidden_size, num_blocks=4)
    embedder = TimestepEmbedder(hidden_size=hidden_size)

    # 将它们封装进 Flow Matcher
    model = ResShiftFlowMatcher(backbone, conditioner, embedder).to(device)

    # ==========================================
    # 4. 数据集、优化器与调度器
    # ==========================================
    logger.info("正在加载数据集...")
    dataloader = get_dataloader(
        hr_dir=hr_data_dir,
        lr_dir=lr_data_dir,
        patch_size=patch_size,
        batch_size=batch_size,
        num_workers=4
    )
    if len(dataloader) == 0:
        raise ValueError(f"DataLoader 为空！请检查 {hr_data_dir} 中是否有足够的图片（当前 batch_size={batch_size}，且启用了 drop_last=True）。")

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # 使用余弦退火学习率，让训练后期更加平滑
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 混合精度 Scaler
    scaler = torch.GradScaler("cuda", enabled=amp_enabled)

    # ==========================================
    # 5. 核心训练循环
    # ==========================================
    logger.info("开始训练...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        # 使用 tqdm 包装 dataloader 显示进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        for step, batch in enumerate(pbar):
            # 获取数据并送入设备
            hr_img = batch["HR"].to(device, non_blocking=True)
            lr_img = batch["LR"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # CUDA 上开启 AMP；CPU 上退化为普通上下文
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )

            with amp_ctx:
                # 这里的 forward 会自动构建 z_t 轨迹并计算 MSE Loss
                loss = model(hr_img=hr_img, lr_img=lr_img)

            # 缩放 Loss 并反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪 (防止梯度爆炸)
            # 先 unscale，然后再裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新权重
            scaler.step(optimizer)
            scaler.update()

            # 记录日志
            loss_value = loss.item()
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_loss += loss_value
            pbar.set_postfix({
                "Loss": f"{loss_value:.4f}",
                "LR": f"{current_lr:.2e}"
            })
        
        scheduler.step()
        
        # 计算并记录当前 Epoch 的平均 Loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch} 结束 | 平均 Loss: {avg_epoch_loss:.6f} | LR: {current_lr:.2e}")
        
        # 记录到 TensorBoard
        logger.log_metrics({
            "Train/Loss": avg_epoch_loss,
            "Train/LR": current_lr
        }, step=epoch)
        
        # ==========================================
        # 6. 模型保存
        # ==========================================
        if epoch % save_every_epochs == 0 or epoch == epochs:
            ckpt_path = os.path.join(ckpt_dir, f"pmf_resshift_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"模型已保存至: {ckpt_path}")

    # 训练结束，关闭 Logger
    logger.info("训练全部完成！")
    logger.close()

if __name__ == "__main__":
    main()