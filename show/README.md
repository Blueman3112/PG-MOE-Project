
# PG-MoE Web Demo 独立运行指南

这是一个独立的演示模块，允许在不依赖完整训练环境的情况下运行 PG-MoE 图像检测演示。

## 1. 快速开始

### 步骤 A: 安装依赖
进入 `show` 目录并安装所需 Python 包：
```bash
cd show
pip install -r requirements.txt
```

### 步骤 B: 准备模型权重
请确保您已经训练好了模型。
脚本会自动扫描以下路径寻找 `.pth` 模型文件：
1. `show/results/` (推荐，将模型文件复制到这里)
2. `../results/` (如果是从完整项目运行，会自动读取上级目录的训练结果)

### 步骤 C: 运行演示
```bash
python web_demo.py
```
启动后，访问终端显示的链接 (通常是 `http://127.0.0.1:7860`)。

## 2. 独立部署说明
如果您在全新的服务器上克隆了本项目：
1. 确保已安装 Python 环境。
2. 将训练好的 `.pth` 模型文件放入 `show/results/` 文件夹中（如果没有该文件夹请新建）。
3. 运行 `python web_demo.py` 即可，首次运行会自动下载 CLIP 预训练模型。
