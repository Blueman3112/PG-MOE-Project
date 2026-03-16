#!/bin/bash

# 确保脚本在项目根目录下运行
if [ ! -d "code" ]; then
    echo "错误: 请在项目根目录下运行此脚本 (例如: bash run.sh)"
    exit 1
fi

echo "准备启动后台训练..."

# 创建 results 目录（如果不存在）
mkdir -p results

# 默认参数
DATASET="dataset-A"
EPOCHS=20
LR=1e-4
BATCH_SIZE=32
NUM_WORKERS=4
LAMBDA_ORTH=0.05

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --num_workers) NUM_WORKERS="$2"; shift ;;
        --lambda_orth) LAMBDA_ORTH="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

echo "配置信息:"
echo "  数据集: $DATASET"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Num Workers: $NUM_WORKERS"
echo "  Lambda Orth: $LAMBDA_ORTH"

# 使用 nohup 运行
# 注意：主要的训练日志(train.log)会由 Python 脚本在 results/ 目录下的对应实验文件夹中生成
# 这个 nohup_launch.log 主要用于捕捉 Python 解释器级别的错误或启动阶段的输出
nohup python code/train.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --lambda_orth "$LAMBDA_ORTH" \
    > results/nohup_launch.log 2>&1 &

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
