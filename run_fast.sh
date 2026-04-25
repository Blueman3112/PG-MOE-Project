#!/bin/bash

# 确保脚本在项目根目录下运行
if [ ! -d "code" ]; then
    echo "错误: 请在项目根目录下运行此脚本 (例如: bash run_fast.sh)"
    exit 1
fi

echo "准备启动极速后台训练 (Fast Mode)..."

# 创建 results 目录（如果不存在）
mkdir -p results

# 默认参数（极速模式下 Batch Size 可以设得很大）
DATASET="dataset-A"
FUSION_TYPE="moe"  # 可选: moe, concat
EPOCHS=20
LR=1e-4
BATCH_SIZE=256
NUM_WORKERS=4
LAMBDA_ORTH=0.05

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --fusion_type) FUSION_TYPE="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --num_workers) NUM_WORKERS="$2"; shift ;;
        --lambda_orth) LAMBDA_ORTH="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

echo "极速模式配置信息:"
echo "  数据集: $DATASET"
echo "  融合机制: $FUSION_TYPE"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Num Workers: $NUM_WORKERS"
echo "  Lambda Orth: $LAMBDA_ORTH"

# 使用 nohup 运行极速训练脚本
nohup python code/train_fast.py \
    --dataset "$DATASET" \
    --fusion_type "$FUSION_TYPE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --lambda_orth "$LAMBDA_ORTH" \
    > results/nohup_fast_launch.log 2>&1 &

PID=$!
echo "------------------------------------------------"
echo "极速训练已在后台启动！"
echo "进程 PID: $PID"
echo "------------------------------------------------"
echo "1. 启动日志: tail -f results/nohup_fast_launch.log"
echo "   (随时查看运行是否正常)"
echo ""
echo "2. 实时极速训练日志 & 结果:"
echo "   请查看 results/ 目录下最新生成的文件夹 (以 Temp_Fast_ 开头)"
echo "------------------------------------------------"
