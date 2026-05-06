# PG-MOE-Project 项目结构与备份清单

由于服务器资源定期清理，本文档用于记录截至 `2026年05月06日` 的项目中各个脚本、数据结构的分布与用途，以便于未来在其他环境中快速恢复实验上下文。

## 📁 目录结构概览

```text
/hy-tmp/PG-MOE-Project
├── baseline/                  # 【对比基线实验专区】用于生成或训练传统基准对比模型的环境
│   ├── DCT-own/               # DCT（离散余弦变换）频域基线方法源码
│   ├── LGrad-own/             # LGrad基线方法源码
│   ├── data4DCT/              # 用于将图片处理为DCT特征的脚本
│   └── data4LGrad/            # 内部包含各种预训练骨干 (StyleGAN, PGGAN等架构文件)
├── code/                      # 【核心代码区】包括原始框架与极速运行框架
│   ├── dataset.py             # 原始数据集加载(原图解码+CLIP前处理)
│   ├── dataset_fast.py        # 极速数据集加载(直接利用mmap映射预解算的.pt特征文件)
│   ├── extract_clip_features.py # 核心预处理：将图片提转为.pt高速缓存张量的脚本
│   ├── loss.py                # 损失函数定义 (含非常核心的 OrthogonalLoss 正交损失)
│   ├── model.py               # 原始版PG-MoE模型定义(包含CLIP挂载)
│   ├── model_fast.py          # 极速版模型定义(卸载了CLIP，直接接手特征)
│   ├── model_fastnew.py       # 极速版拓展模型(支持精确控制 spatial_only, frequency_only, add 等消融结构)
│   ├── train.py               # 原始耗时训练循环
│   ├── train_fast.py          # 配合特征缓存的极速版训练循环
│   ├── train_fastnew.py       # 搭配 model_fastnew.py 支持更细粒度的消融框架的训练循环
│   ├── generate_report.py     # 报告导出脚本(V1: 输出特征级相似度与路由权重)
│   └── generate_report_v2.py  # 报告导出脚本(V2: 自动解析图像所属细分子类如 SDV5/proGAN)
├── pretrained_models/         # 【预训练权重目录】
│   ├── open_clip_pytorch_model.bin # CLIP 骨干网本地权重
│   └── 单独下载CLIP模型.txt        # 备忘录
├── LastResults/               # 【历史跑批实验图表与作图脚本仓库】(直接挂钩论文各个实验小节)
│   ├── Expr1 ~ Expr3/         # 各个阶段实验的分类存档 (原始/无正交/各路变体)
│   ├── Expr9--orthLoss/       # 正交损失曲线对比
│   └── 包含大量的 .py 作图脚本 (如 plot_fig5_8_*.py) 与生成的论文配图 .png
├── results/                   # 【当前模型输出区】(极速模式的输出结果)
│   ├── PG-MoE_bestModel.pth   # （软链接/复制的）最终极选模型权重
│   ├── dataset-B_test_report_*.csv # 生成的各种V2测试报告结果表格
│   └── fast-dataset-*/        # 各项极速实验训练结束后保存的 epoch 权重、混淆矩阵、log 报告
├── *.sh (Bash Scripts)        # 【快速启动脚本】
│   ├── run.sh                 # 原始启动脚本
│   ├── run_fast.sh            # 极速模式启动
│   └── run_fastnew.sh         # 支持细粒度结构控制的极速启动脚本
└── *.md (Markdown Docs)       # 【实验规划与复盘备忘】
    ├── 规划0420.md              # 早期论文规划路线
    ├── 规划0505.md              # 修正后的 Dataset-B 及正交消融架构规划
    ├── 比对.md                  # 两种规划差异及演进论述
    └── plan1156.md              # 阶段性复盘记录
```

## 📝 核心模块用途说明

### 1. `code/` (核心业务逻辑)
这是整个PG-MoE网络真正的灵魂所在。本项目的特色是在训练过程中因为爆显存/慢，而衍生出了**独立特征提取机制（Fast框架）**。
- **慢速原图框架**：`dataset.py`, `model.py`, `train.py`。从加载 JPEG 图片开始，使用 CPU 前处理并扔给 CLIP Backbone。
- **特征工程预计算**：`extract_clip_features.py`。将原图通过 CLIP 提前剥离为 `patch_tokens` 和 `class_token` 存入 `datasets_features` 目录下巨大的 `.pt` 缓存文件。
- **极速训练框架**：`dataset_fast.py`, `model_fastnew.py`, `train_fastnew.py`。这是目前最好用的实验流。利用 `mmap=True` 零拷贝映射硬盘数据，完美突破 30+GB 特征的内存溢出瓶颈，1 分钟跑完 20 个 Epoch。

### 2. 模型核心设计验证 (`loss.py` & `model_fastnew.py`)
- 图表已经证明：**Orthogonal Loss** (`loss.py`) 能有效压制由于网络惰性导致的频率、空间特征同质化（特征塌缩）。
- 在 `model_fastnew.py` 中，通过精确剔除单路网络实现了 `spatial_only`、`frequency_only` 和直接相加 `add` 的消融比对。从而严谨验证了利用交叉路由（Router）调配两种解耦特征（PG-MoE）的效果最大化。

### 3. 数据分析汇报 (`generate_report_v2.py`)
用于评估最后 `.pth` 模型对测试集的准确度。它会自动剥离出 512 维特征，并只输出 `cosine_similarity`（验证特征是否同质）、`w_s` 和 `w_f`（路由器怎么发配权重）。V2 版额外加入了根据路径结构判定子类别（如 `SDV5`、`proGAN`），极大方便用以作细分类统计。

### 4. 论文作图文件 (`LastResults/`)
这里留存着各种将 `training_metrics.csv` 或者 `test_report.csv` 转为图谱的 Python 画图代码。例如 AUC柱状图、特征余弦衰减对比图、ROC 曲线等，都分存在 Expr1 至 Expr9 目录下。

## 💡 服务器数据防丢指南
如果在未来实例被销毁，需在新系统中复现工作：
1. **源码克隆**：拉取上述代码结构。
2. **下载原版数据提取特征**：放入 `datasets/`，运行 `python code/extract_clip_features.py` 重新生成高速缓存。
3. **极速训练**：运行 `run_fastnew.sh` 对各结构（moe, add, spatial_only, noOrth等等）进行重训。
4. **复原曲线**：调用 `generate_report_v2.py` 重提最终指标或将输出复制到电脑本地画图。
