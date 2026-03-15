#!/bin/bash

# 确保脚本在项目根目录下运行
if [ ! -d "code" ]; then
    echo "错误: 请在项目根目录下运行此脚本 (例如: bash run.sh)"
    exit 1
fi

echo "准备启动后台训练..."

# 创建 results 目录（如果不存在）
mkdir -p results

# 使用 nohup 运行
# 注意：主要的训练日志(train.log)会由 Python 脚本在 results/ 目录下的对应实验文件夹中生成
# 这个 nohup_launch.log 主要用于捕捉 Python 解释器级别的错误或启动阶段的输出
nohup python code/train.py > results/nohup_launch.log 2>&1 &

PID=$!
echo "------------------------------------------------"
echo "训练已在后台启动！"
echo "进程 PID: $PID"
echo "------------------------------------------------"
echo "1. 启动日志: tail -f results/nohup_launch.log"
echo "   (如果训练没有开始，请检查此文件)"
echo ""
echo "2. 实时训练日志 & 结果:"
echo "   请查看 results/ 目录下最新生成的文件夹 (以 Temp_ 开头)"
echo "   在该文件夹中会包含 train.log 和 training_metrics.csv"
echo "------------------------------------------------"
