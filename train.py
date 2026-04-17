# 主训练脚本 (包含前向传播、反向传播、模型保存)

import os
from contextlib import nullcontext

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.pmf_dit import pmf_DiT
from models.conditioner import LRConditioner
from models.embedder import TimestepEmbedder
from models.flow_matching import ResShiftFlowMatcher
from data.dataset import get_dataloader


def main():
    # ==========================================
    # 1. 超参数与实验配置 (Configuration)
    # ==========================================
    # 硬件设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    print(f"启动 pMF-ResShift 训练流程 | 运行设备: {device}")

    # 路径设置 (请确保这两个文件夹存在，或者 lr_dir 设为 None 使用在线降质)
    hr_data_dir = "./data/train_hr"  # 替换为你的 GT 高清图路径
    lr_data_dir = None               # 如果没有现成的 LR，设为 None 自动生成
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # 训练超参数
    batch_size = 8
    patch_size = 256
    epochs = 500
    learning_rate = 2e-4
    save_every_epochs = 10  # 每隔几轮保存一次模型

    # 模型架构参数
    hidden_size = 768
    num_heads = 12
    depth = 12

    # ==========================================
    # 2. 组装模型大厦
    # ==========================================
    print("正在初始化网络结构...")
    backbone = pmf_DiT(
        in_channels=3,
        patch_size=4,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads
    )
    conditioner = LRConditioner(in_channels=3, hidden_size=hidden_size, num_blocks=4)
    embedder = TimestepEmbedder(hidden_size=hidden_size)

    # 将它们封装进 Flow Matcher
    model = ResShiftFlowMatcher(backbone, conditioner, embedder).to(device)

    # ==========================================
    # 3. 数据集、优化器与调度器
    # ==========================================
    print("正在加载数据集...")
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
    # 4. 核心训练循环 (Training Loop)
    # ==========================================
    print("开始训练...")
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
            epoch_loss += loss_value
            pbar.set_postfix({
                "Loss": f"{loss_value:.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.2e}"
            })

        # 更新学习率
        scheduler.step()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} 结束 | 平均 Loss: {avg_epoch_loss:.6f}")

        # ==========================================
        # 5. 模型保存
        # ==========================================
        if epoch % save_every_epochs == 0 or epoch == epochs:
            ckpt_path = os.path.join(save_dir, f"pmf_resshift_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"模型已保存至: {ckpt_path}")


if __name__ == "__main__":
    main()