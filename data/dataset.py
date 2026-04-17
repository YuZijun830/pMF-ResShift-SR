# PyTorch Dataset 实现 (成对的 HR/LR 图像读取)

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

class SRDataset(Dataset):
    """
    pMF-ResShift 定制版超分辨率数据集。
    支持两种模式：
    1. Paired Mode (配对模式): 传入 hr_dir 和 lr_dir，读取现成的成对图片。
    2. Degradation Mode (在线降质): 只传入 hr_dir，在线使用 Bicubic 下采样再上采样生成 LR。
    """
    def __init__(self, hr_dir, lr_dir=None, patch_size=256, scale=4):
        """
        hr_dir: 高清原图 (GT) 文件夹路径
        lr_dir: 低清图文件夹路径 (如果为 None，则开启在线降质模式)
        patch_size: 训练时 HR 图像的裁剪大小 (例如 256x256)
        scale: 超分倍率 (例如 4 倍)
        """
        super().__init__()
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.scale = scale

        # 获取所有 HR 图片的文件名
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        self.image_names = [f for f in os.listdir(hr_dir) if f.lower().endswith(valid_extensions)]
        
        # 基础的张量转换 (ToTensor 会将像素值压缩到 [0, 1])
        self.to_tensor = transforms.ToTensor()

        self.resample = Image.Resampling.BICUBIC


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 1. 读取 HR 图像
        img_name = self.image_names[idx]
        hr_path = os.path.join(self.hr_dir, img_name)
        hr_img = Image.open(hr_path).convert('RGB')

        # 2. 读取或生成 LR 图像
        if self.lr_dir is not None:
            # Paired 模式：假设 LR 图片和 HR 图片名字相同
            lr_path = os.path.join(self.lr_dir, img_name)
            lr_img = Image.open(lr_path).convert('RGB')
            # 注意：在 ResShift 和本框架中，输入到网络的 LR 往往是先插值放到和 HR 一样大
            lr_img = lr_img.resize(hr_img.size, resample=self.resample)
        else:
            # Degradation 模式：在线生成 LR (先缩小，再放大对齐分辨率)
            w, h = hr_img.size
            lr_w, lr_h = w // self.scale, h // self.scale
            lr_img = hr_img.resize((lr_w, lr_h), resample=self.resample)
            lr_img = lr_img.resize((w, h), resample=self.resample)

        # 3. 转换为 Tensor (范围 [0, 1])
        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)

        # 4. 联合数据增强 (Joint Data Augmentation)
        # 必须确保 HR 和 LR 裁剪的是同一个物理位置！
        i, j, h, w = transforms.RandomCrop.get_params(
            hr_tensor,
            output_size=(self.patch_size, self.patch_size)
            )
        hr_img = F.crop(hr_tensor,  i, j, h, w)
        lr_img = F.crop(lr_tensor,  i, j, h, w)

        # 随机水平翻转 (概率 50%)
        if random.random() > 0.5:
            hr_tensor = F.hflip(hr_tensor)
            lr_tensor = F.hflip(lr_tensor)
            
        # 随机垂直翻转 (概率 50%)
        if random.random() > 0.5:
            hr_tensor = F.vflip(hr_tensor)
            lr_tensor = F.vflip(lr_tensor)

        # 5. 像素值归一化到 [-1, 1]
        # pMF 等扩散/流匹配模型通常在 [-1, 1] 空间训练表现更好
        hr_tensor = hr_tensor * 2.0 - 1.0
        lr_tensor = lr_tensor * 2.0 - 1.0

        # 将最新的 torchvision 特殊子类强行转换为 numpy
        # 斩断所有底层锁死的 Storage，再转回 torch.Tensor
        # ==========================================
        hr_pure = torch.from_numpy(hr_tensor.numpy().copy())
        lr_pure = torch.from_numpy(lr_tensor.numpy().copy())

        # 返回一个字典，完美对接我们在 flow_matching.py 里的逻辑
        return {
            'HR': hr_pure,
            'LR': lr_pure
        }


# ==========================================
# 简单的 DataLoader 包装函数 (供 train.py 调用)
# ==========================================
def get_dataloader(hr_dir, lr_dir=None, patch_size=256, batch_size=16, num_workers=4, scale=4):
    dataset = SRDataset(hr_dir, lr_dir, patch_size=patch_size, scale=scale)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True, # 加速数据转移到 GPU
        drop_last=True
    )
    return dataloader