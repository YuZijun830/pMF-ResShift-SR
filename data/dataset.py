import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SRDataset(Dataset):
    """
    pMF-ResShift 定制版超分辨率数据集。
    完全使用 PIL 原生方法进行增强，防弹级代码，免疫一切 PyTorch 版本冲突。
    """
    def __init__(self, hr_dir, lr_dir=None, patch_size=256, scale=4):
        super().__init__()
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.scale = scale

        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        self.image_names = [f for f in os.listdir(hr_dir) if f.lower().endswith(valid_extensions)]
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 1. 读取原生 PIL 图像
        img_name = self.image_names[idx]
        hr_path = os.path.join(self.hr_dir, img_name)
        hr_img = Image.open(hr_path).convert('RGB')

        # 2. 读取或生成 LR 图像
        if self.lr_dir is not None:
            lr_path = os.path.join(self.lr_dir, img_name)
            lr_img = Image.open(lr_path).convert('RGB')
            lr_img = lr_img.resize(hr_img.size, resample=Image.Resampling.BICUBIC)
        else:
            w, h = hr_img.size
            lr_w, lr_h = w // self.scale, h // self.scale
            lr_img = hr_img.resize((lr_w, lr_h), resample=Image.Resampling.BICUBIC)
            lr_img = lr_img.resize((w, h), resample=Image.Resampling.BICUBIC)

        # ==========================================
        # 👑 无敌裁剪法：纯原生 PIL 数学裁剪
        # ==========================================
        w, h = hr_img.size
        
        # 安全机制：如果原图比 256x256 还小，先把它放大，防止裁剪报错
        if w < self.patch_size or h < self.patch_size:
            target_w = max(w, self.patch_size)
            target_h = max(h, self.patch_size)
            hr_img = hr_img.resize((target_w, target_h), Image.Resampling.BICUBIC)
            lr_img = lr_img.resize((target_w, target_h), Image.Resampling.BICUBIC)
            w, h = target_w, target_h

        # 随机生成裁剪框的左上角坐标
        left = random.randint(0, w - self.patch_size)
        top = random.randint(0, h - self.patch_size)
        right = left + self.patch_size
        bottom = top + self.patch_size

        # 执行绝对安全的 PIL 裁剪
        hr_img = hr_img.crop((left, top, right, bottom))
        lr_img = lr_img.crop((left, top, right, bottom))

        # ==========================================
        # 原生 PIL 翻转
        # ==========================================
        if random.random() > 0.5:
            hr_img = hr_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            lr_img = lr_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            
        if random.random() > 0.5:
            hr_img = hr_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            lr_img = lr_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # ==========================================
        # 3. 最终转为 Tensor 供网络使用
        # 此时的图 100% 都是严格的 256x256
        # ==========================================
        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)

        # 归一化到 [-1, 1]
        hr_tensor = hr_tensor * 2.0 - 1.0
        lr_tensor = lr_tensor * 2.0 - 1.0

        return {
            'HR': hr_tensor,
            'LR': lr_tensor
        }

# ==========================================
# DataLoader 包装函数
# ==========================================
def get_dataloader(hr_dir, lr_dir=None, patch_size=256, batch_size=16, num_workers=4):
    dataset = SRDataset(hr_dir, lr_dir, patch_size=patch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True, 
        drop_last=True
    )
    return dataloader