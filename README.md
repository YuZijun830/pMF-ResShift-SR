
-----

# pMF-ResShift-SR: 像素级流匹配残差位移超分辨率研究

  

本项目是一个创新的图像超分辨率（ISR）研究框架，核心思想是将 **ResShift** 的高效残差位移路径引入到 **pMF (Pixel Mean Flows)** 的像素级流匹配（Flow Matching）框架中。通过使用 **DiT (Diffusion Transformer)** 作为主干网络，本项目实现了在像素空间内直接进行高质量图像恢复，支持从单步（1-Step）到多步的极速 ODE 采样。

## 🌟 核心特性

  - **数学创新**：定义了从 $t=1$（LR 图像）到 $t=0$（HR 图像）的线性演化轨迹 $z_t = (1-t)x_0 + ty$，将超分任务建模为确定性概率流。
  - **先进架构**：
      - **DiT-B/4 Backbone**：基于 Transformer 的生成主干，原生支持 Flash Attention 以优化大分辨率计算。
      - **Cross-Attention 条件注入**：通过轻量级 CNN 编码器下采样 4 倍提取 LR 特征，作为 Key/Value 注入 Transformer 块。
      - **动态位置编码**：支持可变分辨率输入的 2D Sinusoidal Positional Embedding 缓存机制。
  - **高性能工程实现**：
      - **内存优化**：激活 Flash Attention（SDPA），显存占用较朴素实现降低约 16 倍。
      - **鲁棒数据流**：基于原生 PIL 实现的裁剪与增强，彻底杜绝了 PyTorch 多线程加载下的张量视图冲突（Storage Resizing Bug）。

## 📂 目录结构

```text
pMF-ResShift-SR/
├── models/
│   ├── pmf_dit.py        # DiT 主干网络 (支持 Cross-Attention 与动态位置编码)
│   ├── conditioner.py    # LR 特征提取器 (Stride=4 下采样对齐)
│   ├── embedder.py       # 时间步与 2D 空间位置编码生成器
│   └── flow_matching.py  # 流匹配核心逻辑 (Loss 计算与速度场推理)
├── data/
│   └── dataset.py        # 鲁棒性 Dataset 实现 (PIL 增强与在线降质)
├── utils/
│   ├── solver.py         # Euler ODE 求解器 (支持 NFE 步数调节)
│   └── logger.py         # 综合日志器 (支持 TensorBoard 与文本记录)
├── experiments/          # 自动生成的实验结果、权重与日志
├── train.py              # 主训练脚本 (支持 AMP 混合精度与梯度裁剪)
├── sample.py             # 推理脚本 (单图测试入口)
├── environment.yaml      # 配置环境
└── README.md
```

## 🛠️ 环境安装
conda install -r environment.yaml

## 🚀 快速开始

### 1\. 准备数据

将你的高清原图（Ground Truth）放入 `data/train_hr` 目录下。代码会自动进行双三次插值降质生成对应的低清图像。

### 2\. 模型训练

```bash
python train.py
```

训练过程中的 Loss 曲线和学习率变化将自动保存至 `./experiments/run_TIMESTAMP/` 目录下。你可以通过 TensorBoard 实时查看：

```bash
tensorboard --logdir=./experiments
```

### 3\. 执行推理

准备一张低分辨率测试图 `test_lr.png`，运行：

```bash
python sample.py
```

推理支持通过修改 `sampling_steps` 参数来平衡生成质量与速度。


## 📜 致谢

本项目参考了以下学术成果：

  - [Pixel Mean Flows (pMF)](https://arxiv.org/abs/2601.22158)
  - [ResShift: Efficient Diffusion Model for Image Super-resolution](https://arxiv.org/abs/2307.12348)

-----
