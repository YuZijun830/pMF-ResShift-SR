import os
from pathlib import Path

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

# 支持的图片扩展名
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_image_path(image_path):
    """
    解析输入图片路径：
    1. 如果路径存在，直接返回
    2. 如果用户没写扩展名，则自动查找同名图片
    """
    path = Path(image_path)

    if path.exists() and path.is_file():
        return path

    # 如果没写扩展名，尝试自动匹配常见图片格式
    if path.suffix == "":
        for ext in SUPPORTED_IMAGE_EXTS:
            candidate = path.with_suffix(ext)
            if candidate.exists() and candidate.is_file():
                return candidate

            # 再尝试大写扩展名
            candidate_upper = path.with_suffix(ext.upper())
            if candidate_upper.exists() and candidate_upper.is_file():
                return candidate_upper

    raise FileNotFoundError(f"找不到输入图片: {image_path}")


def get_output_format(save_path):
    """
    根据输出路径扩展名推断保存格式
    """
    ext = Path(save_path).suffix.lower()

    format_map = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".bmp": "BMP",
        ".webp": "WEBP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
    }

    if ext not in format_map:
        raise ValueError(
            f"不支持的输出格式: {ext}。"
            f"当前支持: {sorted(SUPPORTED_IMAGE_EXTS)}"
        )

    return format_map[ext]


def crop_patch(img, left, top, patch_size=64):
    """
    从原图中裁剪一个 patch。
    left, top 为左上角像素坐标。
    若越界则自动报错。
    """
    w, h = img.size

    if left < 0 or top < 0:
        raise ValueError(f"裁剪坐标不能为负数: left={left}, top={top}")

    if left + patch_size > w or top + patch_size > h:
        raise ValueError(
            f"裁剪区域越界: 图像大小=({w}, {h}), "
            f"请求区域=({left}, {top}, {left+patch_size}, {top+patch_size})"
        )

    return img.crop((left, top, left + patch_size, top + patch_size))


def load_image_patch(image_path, crop_x=0, crop_y=0, lr_patch_size=64, scale=4):
    """
    从输入图中裁一个 64x64 的 LR patch，
    再 Bicubic 放大到 256x256，
    最后转成 [-1, 1] Tensor。
    """
    img = Image.open(image_path).convert("RGB")

    # 1. 从原图裁 LR 小块
    lr_patch = crop_patch(img, crop_x, crop_y, patch_size=lr_patch_size)

    # 2. 放大到模型训练时的输入尺寸
    sr_input_size = lr_patch_size * scale   # 64 * 4 = 256
    model_input = lr_patch.resize(
        (sr_input_size, sr_input_size),
        resample=Image.Resampling.BICUBIC
    )

    # 3. 转 Tensor 并归一化到 [-1, 1]
    tensor = transforms.ToTensor()(model_input)  # [0, 1]
    tensor = tensor * 2.0 - 1.0                  # [-1, 1]

    return tensor.unsqueeze(0), lr_patch, model_input


def save_image(tensor, save_path):
    """
    将 [-1, 1] 的 Tensor 转回图像并保存
    输出格式由 save_path 的扩展名决定
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    img_tensor = (tensor.squeeze(0).detach().cpu().clamp(-1, 1) + 1.0) / 2.0
    img = transforms.ToPILImage()(img_tensor)

    output_format = get_output_format(save_path)

    # JPEG 不支持透明通道，且一般推荐设置质量参数
    save_kwargs = {}
    if output_format == "JPEG":
        img = img.convert("RGB")
        save_kwargs["quality"] = 95
        save_kwargs["subsampling"] = 0

    img.save(save_path, format=output_format, **save_kwargs)
    print(f"成功保存超分辨率结果至: {save_path}")


def main():
    # ==========================================
    # 1. 基础配置
    # ==========================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 输入可以写成：
    # "test/test_lr.png"
    # "test/test_lr.jpg"
    # "test/test_lr"   <-- 不写扩展名也行，会自动找
    input_path = "test/test_lr.png"

    # 输出格式由扩展名决定
    output_path = "test/output_hr.png"

    sampling_steps = 5
    scale = 4
    lr_patch_size = 64

    # 这里控制裁剪区域左上角坐标
    crop_x = 0
    crop_y = 0

    print(f"初始化 pMF-ResShift 推理流程 | 设备: {device} | 步数: {sampling_steps}")
    print(f"裁剪 LR patch: 左上角=({crop_x}, {crop_y}), 大小={lr_patch_size}x{lr_patch_size}")

    # ==========================================
    # 2. 组装模型
    # ==========================================
    hidden_size = 768

    backbone = pmf_DiT(
        in_channels=3,
        patch_size=4,
        hidden_size=hidden_size,
        depth=12,
        num_heads=12
    )
    conditioner = LRConditioner(
        in_channels=3,
        hidden_size=hidden_size,
        num_blocks=4
    )
    embedder = TimestepEmbedder(hidden_size=hidden_size)

    flow_matcher = ResShiftFlowMatcher(backbone, conditioner, embedder).to(device)

    ckpt_path = "experiments/run_20260417_203802/checkpoints/pmf_resshift_epoch_500.pth"
    flow_matcher.load_state_dict(torch.load(ckpt_path, map_location=device))
    flow_matcher.eval()
    print("模型权重加载完毕")

    # ==========================================
    # 3. 读取数据与推理
    # ==========================================
    if not os.path.exists(input_path):
        print(f"找不到测试图片: {input_path}")
        return

    try:
        lr_tensor, lr_patch_img, bicubic_img = load_image_patch(
            input_path,
            crop_x=crop_x,
            crop_y=crop_y,
            lr_patch_size=lr_patch_size,
            scale=scale
        )
    except Exception as e:
        print(f"读取/裁剪图片失败: {e}")
        return

    lr_tensor = lr_tensor.to(device)

    # 可选：把裁出来的 64x64 patch 和 Bicubic 输入也保存下来，便于对比
    os.makedirs("test", exist_ok=True)
    lr_patch_img.save("test/cropped_lr_patch.png")
    bicubic_img.save("test/cropped_lr_patch_bicubic_x4.png")

    solver = EulerSolver(flow_matcher)

    print("正在沿残差位移路径生成高清细节...")
    with torch.no_grad():
        hr_tensor = solver.sample(lr_tensor, steps=sampling_steps)

    # ==========================================
    # 4. 保存结果
    # ==========================================
    save_image(hr_tensor, output_path)


if __name__ == "__main__":
    main()