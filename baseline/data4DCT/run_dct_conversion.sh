#!/bin/bash

# Ensure we are in the correct directory context
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="/hy-tmp/PG-MOE-Project"

# Default parameters
DATASET="dataset-A"
NUM_WORKERS=16 # Default to a reasonable number of CPU cores

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --num_workers) NUM_WORKERS="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Num Workers: $NUM_WORKERS"

# Define Paths
INPUT_ROOT="$PROJECT_ROOT/datasets/$DATASET"
OUTPUT_ROOT="$PROJECT_ROOT/${DATASET}-DCT"

if [ ! -d "$INPUT_ROOT" ]; then
    echo "Error: Input dataset directory $INPUT_ROOT does not exist."
    exit 1
fi

echo "  Input Root: $INPUT_ROOT"
echo "  Output Root: $OUTPUT_ROOT"

# Create Logs Directory
LOG_DIR="$PROJECT_ROOT/results/DCT_conversion_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
LOG_FILE="$LOG_DIR/${DATASET}_conversion_${TIMESTAMP}.log"

echo "Starting DCT conversion in background..."
echo "Logs will be saved to: $LOG_FILE"

# Run Process with nohup
nohup python "$SCRIPT_DIR/process_dct.py" \
    --dataset_root "$INPUT_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --num_workers "$NUM_WORKERS" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "------------------------------------------------"
echo "DCT Conversion process started!"
echo "PID: $PID"
echo "Log File: $LOG_FILE"
echo "------------------------------------------------"
echo "You can monitor the progress with:"
echo "tail -f $LOG_FILE"