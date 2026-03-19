#!/bin/bash

# Ensure we are in the correct directory context or handle paths relative to project root
# This script is intended to be run from the project root or the script's directory.
# We'll assume it's run from project root for consistency with PG-MoE run.sh, but we'll handle paths carefully.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="/hy-tmp/PG-MOE-Project"
MODEL_URL="https://lid-1302259812.cos.ap-nanjing.myqcloud.com/tmp/karras2019stylegan-bedrooms-256x256_discriminator.pth"
MODEL_FILE="$SCRIPT_DIR/karras2019stylegan-bedrooms-256x256_discriminator.pth"

# Default dataset
DATASET="dataset-A"
BATCH_SIZE=64 # Increased batch size for GPU acceleration

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Batch Size: $BATCH_SIZE"
echo "  Script Directory: $SCRIPT_DIR"

# 1. Check/Download Model
if [ ! -f "$MODEL_FILE" ]; then
    echo "Model file not found. Downloading..."
    wget "$MODEL_URL" -O "$MODEL_FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download model."
        exit 1
    fi
    echo "Model downloaded successfully."
else
    echo "Model file exists."
fi

# 2. Define Paths
INPUT_ROOT="$PROJECT_ROOT/datasets/$DATASET"
# INPUT_ROOT="$PROJECT_ROOT/$DATASET"
OUTPUT_ROOT="$PROJECT_ROOT/${DATASET}-LGrad"

if [ ! -d "$INPUT_ROOT" ]; then
    echo "Error: Input dataset directory $INPUT_ROOT does not exist."
    exit 1
fi

echo "  Input Root: $INPUT_ROOT"
echo "  Output Root: $OUTPUT_ROOT"

# 3. Create Logs Directory
LOG_DIR="$PROJECT_ROOT/results/LGrad_conversion_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
LOG_FILE="$LOG_DIR/${DATASET}_conversion_${TIMESTAMP}.log"

echo "Starting conversion in background..."
echo "Logs will be saved to: $LOG_FILE"

# 4. Run Process with nohup
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
nohup python "$SCRIPT_DIR/process_dataset.py" \
    --dataset_root "$INPUT_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --model_path "$MODEL_FILE" \
    --batch_size "$BATCH_SIZE" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "------------------------------------------------"
echo "Conversion process started!"
echo "PID: $PID"
echo "Log File: $LOG_FILE"
echo "------------------------------------------------"
echo "You can monitor the progress with:"
echo "tail -f $LOG_FILE"
