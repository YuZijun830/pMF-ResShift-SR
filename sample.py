# python sample.py \
#     --input_path test/test02/GroundTruth.jpg \
#     --output_dir test/test02/1_step_full \
#     --ckpt_path experiments/run_20260418_221940_BF16/checkpoints/pmf_resshift_epoch_100.pth \
#     --sampling_steps 1 \
#     --scale 4

# python sample.py \
#     --input_path test/test02/GroundTruth.jpg \
#     --output_dir test/test02/1_step_crop \
#     --ckpt_path experiments/run_20260418_221940_BF16/checkpoints/pmf_resshift_epoch_100.pth \
#     --sampling_steps 1 \
#     --scale 4 \
#     --crop_x 128 \
#     --crop_y 128 \
#     --crop_size 256

import os
import math
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
from PIL import Image, ImageFile
from torchvision import transforms

# 导入我们自己写的模块
from models.pmf_dit import pmf_DiT
from models.conditioner import LRConditioner
from models.embedder import TimestepEmbedder
from models.flow_matching import ResShiftFlowMatcher
from utils.solver import EulerSolver

# 允许读取部分损坏但仍可解码的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description="单张图像推理脚本（支持任意分辨率）")

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        # default="test/test03/GroundTruth.jpg",
        help="输入高清原图路径（用于构造 GT / LR / Bicubic 对比）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        # default="test/test03",
        help="输出结果保存目录"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        # default="experiments/run_20260418_221940_BF16/checkpoints/pmf_resshift_epoch_100.pth",
        help="训练好的模型权重路径"
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=1,
        help="ODE 采样步数"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="超分倍率"
    )

    # 可选裁剪参数；如果不传 crop_size，就直接使用整张图
    parser.add_argument(
        "--crop_x",
        type=int,
        default=0,
        help="裁剪左上角 x 坐标（仅当指定 crop_size 时生效）"
    )
    parser.add_argument(
        "--crop_y",
        type=int,
        default=0,
        help="裁剪左上角 y 坐标（仅当指定 crop_size 时生效）"
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=None,
        help="可选裁剪边长；不指定则使用整张图"
    )

    return parser.parse_args()


def crop_patch(img, left, top, crop_w, crop_h):
    """
    从图像中裁剪任意大小区域。
    """
    w, h = img.size

    if left < 0 or top < 0:
        raise ValueError(f"裁剪坐标不能为负数: left={left}, top={top}")

    if left + crop_w > w or top + crop_h > h:
        raise ValueError(
            f"裁剪区域越界: 原图大小=({w}, {h}), "
            f"请求区域=({left}, {top}, {left + crop_w}, {top + crop_h})"
        )

    return img.crop((left, top, left + crop_w, top + crop_h))


def center_crop_to_multiple(img, multiple):
    """
    将图像中心裁剪到 multiple 的整数倍大小。
    这样既能保证:
    1. 可被 scale 整除
    2. 也满足模型 patch_size 的要求
    """
    w, h = img.size
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple

    if new_w <= 0 or new_h <= 0:
        raise ValueError(
            f"图像尺寸过小，无法裁到 {multiple} 的整数倍。当前尺寸: ({w}, {h})"
        )

    if new_w == w and new_h == h:
        return img, False

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped, True


def load_aligned_inputs(image_path, crop_x=0, crop_y=0, crop_size=None, scale=4, model_patch_size=4):
    """
    支持任意分辨率输入的对齐逻辑：

    1. 读取高清原图
    2. 如果指定 crop_size，则先裁剪指定区域；否则直接使用整张图
    3. 为了兼容 scale 和模型 patch_size，将图像中心裁剪到公倍数尺寸
    4. 将 GT 缩小 scale 倍，模拟 LR
    5. 再把 LR bicubic 放大回 GT 尺寸，作为模型输入
    """
    img = Image.open(image_path).convert("RGB")

    # 先裁剪用户指定区域（如果有）
    if crop_size is not None:
        gt_img = crop_patch(img, crop_x, crop_y, crop_size, crop_size)
    else:
        gt_img = img

    # 同时满足 scale 和 patch_size 的整数倍
    align_unit = math.lcm(scale, model_patch_size)
    gt_img, was_center_cropped = center_crop_to_multiple(gt_img, align_unit)

    gt_w, gt_h = gt_img.size
    lr_w, lr_h = gt_w // scale, gt_h // scale

    # 退化：GT -> LR
    lr_small = gt_img.resize((lr_w, lr_h), resample=Image.Resampling.BICUBIC)

    # 放大：LR -> Bicubic
    lr_bicubic = lr_small.resize((gt_w, gt_h), resample=Image.Resampling.BICUBIC)

    # 转 Tensor，并归一化到 [-1, 1]
    tensor = transforms.ToTensor()(lr_bicubic)
    tensor = tensor * 2.0 - 1.0

    return tensor.unsqueeze(0), gt_img, lr_bicubic, was_center_cropped


def save_image(tensor, save_path):
    """
    将 [-1, 1] 的 Tensor 转回图像并保存
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    img_tensor = (tensor.squeeze(0).detach().cpu().float().clamp(-1, 1) + 1.0) / 2.0
    img = transforms.ToPILImage()(img_tensor)
    img.save(save_path, format="PNG")


def load_model_state(ckpt_path, device):
    """
    更稳健地加载权重，兼容:
    1. 纯 state_dict
    2. DDP 保存（带 module. 前缀）
    3. 可能包在 state_dict / model_state_dict / model 里的情况
    """
    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_path, map_location=device)

    if not isinstance(state_dict, dict):
        raise TypeError("加载得到的 checkpoint 不是 dict，无法解析。")

    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    elif "model_state_dict" in state_dict and isinstance(state_dict["model_state_dict"], dict):
        state_dict = state_dict["model_state_dict"]
    elif "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    return state_dict


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"初始化单图推理 | 设备: {device} | 步数: {args.sampling_steps}")

    if not os.path.exists(args.input_path):
        print(f"找不到输入图片: {args.input_path}")
        return

    # ==========================================
    # 1. 组装模型
    # ==========================================
    hidden_size = 768
    model_patch_size = 4

    backbone = pmf_DiT(
        in_channels=3,
        patch_size=model_patch_size,
        hidden_size=hidden_size,
        depth=12,
        num_heads=12
    )
    conditioner = LRConditioner(in_channels=3, hidden_size=hidden_size, num_blocks=4)
    embedder = TimestepEmbedder(hidden_size=hidden_size)

    flow_matcher = ResShiftFlowMatcher(backbone, conditioner, embedder).to(device)

    state_dict = load_model_state(args.ckpt_path, device)
    flow_matcher.load_state_dict(state_dict)
    flow_matcher.eval()
    print("模型权重加载完毕")

    solver = EulerSolver(flow_matcher)

    # ==========================================
    # 2. 读取数据（支持整图 / 任意 crop）
    # ==========================================
    lr_tensor, gt_img, lr_bicubic_img, was_center_cropped = load_aligned_inputs(
        image_path=args.input_path,
        crop_x=args.crop_x,
        crop_y=args.crop_y,
        crop_size=args.crop_size,
        scale=args.scale,
        model_patch_size=model_patch_size,
    )

    gt_w, gt_h = gt_img.size
    print(f"当前参与推理的对齐后尺寸: {gt_w} x {gt_h}")

    if was_center_cropped:
        print(
            f"注意：为满足 scale={args.scale} 和 patch_size={model_patch_size} 的整除要求，"
            f"图像已自动做中心裁剪。"
        )

    # 先保存 GT 和 Bicubic
    gt_save_path = os.path.join(args.output_dir, "01_GroundTruth.png")
    bicubic_save_path = os.path.join(args.output_dir, "02_LR_Bicubic.png")
    gt_img.save(gt_save_path)
    lr_bicubic_img.save(bicubic_save_path)
    print("已保存 Ground Truth 与 Bicubic 图像")

    lr_tensor = lr_tensor.to(device)

    # ==========================================
    # 3. 推理
    # ==========================================
    print("正在沿残差位移路径生成高清细节...")

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.no_grad(), amp_ctx:
        hr_tensor = solver.sample(lr_tensor, steps=args.sampling_steps)

    # ==========================================
    # 4. 保存模型输出
    # ==========================================
    output_path = os.path.join(args.output_dir, "03_HR_ModelPred.png")
    save_image(hr_tensor, output_path)

    print(f"成功！结果已保存在: {args.output_dir}")
    print(f"  - GT       : {gt_save_path}")
    print(f"  - Bicubic  : {bicubic_save_path}")
    print(f"  - pMF-ResShift-SR : {output_path}")


if __name__ == "__main__":
    main()