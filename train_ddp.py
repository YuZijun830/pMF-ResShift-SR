# 用法示例：
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_ddp.py
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_ddp.py --batch_size_per_gpu 8 --lr 1e-4 --epochs 200
# CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 train_ddp.py --batch_size_per_gpu 4 --save_every_epochs 20

import os
import argparse
import datetime
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.pmf_dit import pmf_DiT
from models.conditioner import LRConditioner
from models.embedder import TimestepEmbedder
from models.flow_matching import ResShiftFlowMatcher
from data.dataset import SRDataset
from utils.logger import SRLogger


def parse_args():
    parser = argparse.ArgumentParser(description="pMF-ResShift DDP Training")

    # 数据路径
    parser.add_argument("--hr_data_dir", type=str, default="./data/DIV2K/DIV2K_train_HR_sub")
    parser.add_argument("--lr_data_dir", type=str, default=None)

    # 训练参数
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help="每张卡的 batch size")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)

    # 模型参数
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--depth", type=int, default=12)

    return parser.parse_args()


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def setup_ddp():
    if not torch.cuda.is_available():
        raise RuntimeError("DDP 训练需要 CUDA 环境。请使用 GPU 并通过 torchrun 启动。")

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    return local_rank, device


def main():
    args = parse_args()

    # ==========================================
    # 0. 初始化 DDP
    # ==========================================
    local_rank, device = setup_ddp()
    rank = get_rank()
    world_size = get_world_size()
    amp_enabled = device.type == "cuda"

    # ==========================================
    # 1. 实验目录与 Logger 初始化（仅主进程）
    # ==========================================
    logger = None
    exp_dir = None
    ckpt_dir = None

    if is_main_process():
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"run_{current_time}_BF16"
        exp_dir = os.path.join("./experiments", exp_name)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        logger = SRLogger(exp_dir)

        logger.info(f"启动 pMF-ResShift DDP 训练流程 | 运行设备: {device}")
        logger.info(f"本次实验目录: {exp_dir}")
        logger.info(f"World Size: {world_size}")
        logger.info(f"单卡 Batch Size: {args.batch_size_per_gpu}")
        logger.info(f"全局 Batch Size: {args.batch_size_per_gpu * world_size}")
        logger.info(f"学习率: {args.lr}")
        logger.info(f"训练轮数: {args.epochs}")
        logger.info(f"混合精度: 开启 BF16 (BFloat16) 防止 NaN 溢出")

    # ==========================================
    # 2. 组装模型大厦
    # ==========================================
    if is_main_process():
        assert logger is not None
        logger.info("正在初始化网络结构...")

    backbone = pmf_DiT(
        in_channels=3,
        patch_size=4,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads
    )
    conditioner = LRConditioner(
        in_channels=3,
        hidden_size=args.hidden_size,
        num_blocks=4
    )
    embedder = TimestepEmbedder(hidden_size=args.hidden_size)

    model = ResShiftFlowMatcher(backbone, conditioner, embedder).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ==========================================
    # 3. 数据集、优化器与调度器
    # ==========================================
    if is_main_process():
        assert logger is not None
        logger.info("正在加载数据集...")

    dataset = SRDataset(
        hr_dir=args.hr_data_dir,
        lr_dir=args.lr_data_dir,
        patch_size=args.patch_size
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    if len(dataloader) == 0:
        raise ValueError(
            f"DataLoader 为空！请检查 {args.hr_data_dir} 中是否有足够的图片"
            f"（当前单卡 batch_size={args.batch_size_per_gpu}，且启用了 drop_last=True）。"
        )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 核心修复 1：计算整个训练过程的总 Iteration 数量
    total_steps = len(dataloader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    # ==========================================
    # 4. 核心训练循环
    # ==========================================
    if is_main_process():
        assert logger is not None
        logger.info("开始训练...")

    try:
        for epoch in range(1, args.epochs + 1):
            sampler.set_epoch(epoch)

            model.train()
            epoch_loss = 0.0

            if is_main_process():
                pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
                data_iter = pbar
            else:
                pbar = None
                data_iter = dataloader

            for step, batch in enumerate(data_iter):
                hr_img = batch["HR"].to(device, non_blocking=True)
                lr_img = batch["LR"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if amp_enabled
                    else nullcontext()
                )

                with amp_ctx:
                    loss = model(hr_img=hr_img, lr_img=lr_img)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 核心修复 2：把 scheduler.step() 放到 Iteration 循环内部
                scheduler.step()

                loss_value = loss.item()
                current_lr = optimizer.param_groups[0]["lr"]
                epoch_loss += loss_value

                if pbar is not None:
                    pbar.set_postfix({
                        "Loss": f"{loss_value:.4f}",
                        "LR": f"{current_lr:.2e}"
                    })

            # 聚合所有进程的 epoch_loss
            loss_tensor = torch.tensor(epoch_loss, dtype=torch.float32, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_epoch_loss = loss_tensor.item() / (world_size * len(dataloader))

            # ==========================================
            # 5. 日志记录与模型保存（仅主进程）
            # ==========================================
            if is_main_process():
                assert logger is not None
                assert ckpt_dir is not None

                # 取最后一个 batch 的 learning rate 作为该 epoch 的记录
                logger.info(f"Epoch {epoch} 结束 | 平均 Loss: {avg_epoch_loss:.6f} | LR: {current_lr:.2e}")

                logger.log_metrics({
                    "Train/Loss": avg_epoch_loss,
                    "Train/LR": current_lr
                }, step=epoch)

                if epoch % args.save_every_epochs == 0 or epoch == args.epochs:
                    ckpt_path = os.path.join(ckpt_dir, f"pmf_resshift_epoch_{epoch}.pth")
                    torch.save(model.module.state_dict(), ckpt_path)
                    logger.info(f"模型已保存至: {ckpt_path}")

        if is_main_process():
            assert logger is not None
            logger.info("训练全部完成！")
            logger.close()

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()