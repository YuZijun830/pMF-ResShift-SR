# TensorBoard / Wandb 记录import os

import os
import logging
from torch.utils.tensorboard import SummaryWriter

class SRLogger:
    """
    超分辨率项目专属 Logger。
    支持同时输出到终端和 log 文件，并自动记录 TensorBoard 标量。
    """
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 1. 配置 Python 标准 logging
        self.logger = logging.getLogger("pMF_ResShift")
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加 handler 导致重复打印
        if not self.logger.handlers:
            # 文件处理器 (写入到 exp_dir/train.log)
            fh = logging.FileHandler(os.path.join(exp_dir, "train.log"), mode='a', encoding='utf-8')
            fh.setLevel(logging.INFO)
            
            # 控制台处理器 (打印到终端)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # 日志格式
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
            
        # 2. 配置 TensorBoard
        self.tb_writer = SummaryWriter(log_dir=self.exp_dir)

    def info(self, msg):
        """打印并记录常规信息"""
        self.logger.info(msg)

    def log_metrics(self, metrics_dict, step):
        """
        记录数值到 TensorBoard
        metrics_dict: 字典格式，如 {"Train/Loss": 0.01, "Train/LR": 2e-4}
        """
        for key, value in metrics_dict.items():
            self.tb_writer.add_scalar(key, value, step)

    def close(self):
        """关闭 TensorBoard 写入器"""
        self.tb_writer.close()