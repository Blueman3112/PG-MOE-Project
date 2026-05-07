# PG-MoE：基于混合专家机制的深度伪造图像检测模型

## 项目简介

PG-MoE（Patch-Guided Mixture of Experts）是一种用于检测 AI 生成（深度伪造）图像的深度学习模型。该项目将冻结的 CLIP 视觉编码器（ViT-L/14）作为特征提取骨干网络，结合空间域与频率域的双路专家模块（Mixture of Experts, MoE），并通过动态门控路由器自适应地融合两路专家的判别特征，从而实现对真实图像与生成图像的高精度二分类检测。

在两个独立测试集上的实验结果表明，PG-MoE 分别达到了 **AUC = 0.9976**（数据集 A）和 **AUC = 0.9994**（数据集 B）的检测性能，显著优于 LGrad 和 DCT 等基线方法。

---

## 目录结构

```
PG-MOE-Project/
├── code/                        # 核心模型实现
│   ├── model.py                 # PG-MoE 模型架构
│   ├── dataset.py               # 数据集加载工具
│   ├── loss.py                  # 自定义损失函数
│   └── train.py                 # 训练流程主脚本
├── baseline/                    # 对比基线方法
│   ├── DCT-own/                 # 基于 DCT 的基线
│   │   ├── dataset_dct.py
│   │   ├── train_dct.py
│   │   └── run_dct.sh
│   ├── LGrad-own/               # 基于 LGrad 的基线
│   │   ├── train_lgrad.py
│   │   └── run_lgrad.sh
│   ├── data4DCT/                # DCT 数据预处理
│   ├── data4LGrad/              # LGrad 数据预处理（含 GAN 模型）
│   └── models/                  # 生成器/判别器网络结构
├── datasets/                    # 训练数据（需单独下载）
├── pretrained_models/           # CLIP 预训练权重（约 1.2 GB，需单独下载）
├── results/                     # 训练结果与评估指标
└── run.sh                       # 训练启动脚本
```

---

## 模型架构

### 整体设计

```
输入图像
   │
   ▼
CLIP ViT-L/14（冻结）──────┐
   │  Patch Tokens            │  Class Token
   ▼                          ▼
SpatialAdapter          FrequencyAdapter
（空间专家）               （频率专家）
   │                          │
   └──────────┬───────────────┘
              ▼
         GatingRouter
      （动态门控路由器）
              │
              ▼
     加权融合：F = w_s·F_s + w_f·F_f
              │
              ▼
         二分类头（全连接层）
              │
              ▼
        真实 / 伪造
```

### 核心模块说明

| 模块 | 类名 | 功能描述 |
|------|------|----------|
| 空间专家 | `SpatialAdapter` | 基于 CNN 的模块，对 CLIP Patch Tokens 进行空间特征提取；输入形状 `[B, 256, 1024]`，输出 512 维空间特征向量 |
| 频率专家 | `FrequencyAdapter` | 基于 Transformer + FFT 的模块，提取图像频率域特征；输出 512 维频率特征向量 |
| 门控路由器 | `GatingRouter` | 线性层 + Softmax，将 Class Token 映射为两路专家的混合权重 `(w_s, w_f)` |
| 完整模型 | `PGMoE` | 集成以上三个模块，冻结 CLIP 骨干，通过 Forward Hook 捕获中间层特征，输出二分类概率 |

### 损失函数

- **Focal Loss**：缓解真实/伪造样本类别不均衡问题，降低易分样本权重（默认 α=0.25，γ=2.0）。
- **正交损失（Orthogonal Loss）**：显式惩罚两路专家特征向量之间的余弦相似度，鼓励空间专家与频率专家学习互补、多样的表征：
  ```
  L_orth = E[(F_s_norm · F_f_norm)²]
  ```
- **总损失**：
  ```
  L_total = L_focal + λ_orth × L_orth
  ```

---

## 环境依赖

| 依赖 | 版本要求 |
|------|---------|
| Python | ≥ 3.8 |
| PyTorch | ≥ 1.9.0 |
| torchvision | 与 PyTorch 对应版本 |
| open_clip_torch | 最新版 |
| scikit-learn | 最新版 |
| numpy | 最新版 |
| tqdm | 最新版 |

**硬件建议：** 推荐使用支持 CUDA 的 NVIDIA GPU（ViT-L/14 + batch=32 约需 8~10 GB 显存）。

---

## 数据集准备

数据集需按以下目录结构组织，放置在 `datasets/` 目录下：

```
datasets/
└── dataset-{A 或 B}/
    ├── train/
    │   ├── 0/          # 真实图像
    │   └── 1/          # AI 生成（伪造）图像
    ├── val/
    │   ├── 0/
    │   └── 1/
    └── test/
        ├── 0/
        └── 1/
```

图像统一预处理为 **224×224** 分辨率，使用 CLIP 官方的均值与标准差进行归一化。

---

## 预训练权重

CLIP ViT-L/14 权重文件（约 1.2 GB）需手动下载并放置于：

```
pretrained_models/open_clip_pytorch_model.bin
```

---

## 快速开始

### 方式一：使用启动脚本

```bash
bash run.sh --dataset dataset-A --epochs 20 --lr 1e-4 --batch_size 32 --lambda_orth 0.05
```

### 方式二：直接运行 Python 脚本

```bash
python code/train.py \
  --dataset dataset-B \
  --epochs 20 \
  --lr 1e-4 \
  --batch_size 32 \
  --lambda_orth 0.05 \
  --num_workers 4 \
  --results_dir ./results
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `dataset-A` | 数据集名称（与 `datasets/` 下目录名一致） |
| `--epochs` | `20` | 训练总轮数 |
| `--lr` | `1e-4` | 学习率 |
| `--batch_size` | `32` | 批大小 |
| `--lambda_orth` | `0.05` | 正交损失权重系数 |
| `--num_workers` | `4` | DataLoader 工作进程数 |
| `--results_dir` | `./results` | 结果输出目录 |

---

## 训练策略

1. **两阶段训练**：
   - 第 1~9 轮：同时训练空间专家、频率专家、门控路由器及分类头；
   - 第 10 轮起：冻结两路专家模块，仅微调门控路由器与分类头，提升后期训练稳定性。

2. **学习率调度**：余弦退火策略，学习率从初始值线性衰减至 1e-6。

3. **早停机制**：若连续 20 个 epoch 验证集 AUC 无显著提升，则提前终止训练。

4. **优化器**：AdamW，仅优化可训练参数（专家模块 + 路由器 + 分类头），CLIP 骨干保持冻结。

---

## 输出结果

每次训练会在 `results/` 下生成以时间戳命名的子目录，包含：

| 文件 | 内容 |
|------|------|
| `train.log` | 完整训练日志 |
| `training_metrics.csv` | 每轮训练/验证指标（AUC、Acc、F1 等） |
| `train_info.txt` | 超参数汇总及最终测试分数 |
| 模型检查点 | 验证集 AUC 最高时对应的模型权重 |

评估指标包括：**AUC、准确率（Accuracy）、F1 分数、精确率（Precision）、召回率（Recall）、推理 FPS**。

---

## 实验结果

### PG-MoE 与基线方法对比

| 方法 | 数据集 | AUC | 准确率 |
|------|--------|-----|--------|
| **PG-MoE** | A | **0.9976** | **0.9719** |
| **PG-MoE** | B | **0.9994** | **0.9889** |
| LGrad 基线 | B | 0.9877 | 0.9414 |
| DCT 基线 | B | 0.9145 | 0.8359 |

PG-MoE 在两个数据集上均超越基线方法，在数据集 B 上 AUC 达到 **99.94%**。

---

## 基线方法

`baseline/` 目录提供了两种对比方法的完整实现：

- **LGrad**（`baseline/LGrad-own/`）：基于梯度统计特征的深度伪造检测方法。
- **DCT**（`baseline/DCT-own/`）：基于离散余弦变换（DCT）频谱特征的检测方法。

两种基线均使用与 PG-MoE 相同的数据集和评估协议进行对比实验。

---

## 技术亮点

1. **冻结 CLIP 骨干 + Forward Hook**：无需修改预训练模型前向传播，通过钩子机制高效捕获中间层 Patch Token 特征。
2. **正交正则化**：显式约束两路专家学习互补特征，避免特征冗余，提升模型泛化能力。
3. **两阶段训练**：先充分训练专家模块，再微调融合层，兼顾收敛速度与训练稳定性。
4. **Focal Loss**：有效应对真实/伪造样本分布不均的问题。
5. **推理效率评估**：训练结束后自动统计推理 FPS，便于部署评估。
