#!/bin/bash

# Evaluation script for GC-VLN
# Usage: bash run_eval.sh [r2r|rxr]

# Default arguments
DATASET="${1:-rxr}"
SPLIT_NUM=1
SPLIT_INDEX=0
GSAM2_SERVER_PORT=7000
DEVICE=0
export NLTK_DATA=./data/nltk_data

# Set config and experiment ID based on dataset
if [ "$DATASET" = "rxr" ]; then
    EXP_CONFIG="config/rxr_vlnce.yaml"
    EXPERIMENT_ID="rxr_eval_$(date +%Y%m%d-%H%M%S)"
elif [ "$DATASET" = "r2r" ]; then
    EXP_CONFIG="config/r2r_vlnce.yaml"
    EXPERIMENT_ID="r2r_eval_$(date +%Y%m%d-%H%M%S)"
else
    echo "Error: dataset must be 'r2r' or 'rxr'"
    exit 1
fi

echo "========================================="
echo "GC-VLN Evaluation"
echo "========================================="
echo "Dataset: $DATASET"
echo "Config: $EXP_CONFIG"
echo "Experiment ID: $EXPERIMENT_ID"
echo "Device: GPU $DEVICE"
echo "GSAM2 Server Port: $GSAM2_SERVER_PORT"
echo "========================================="

# Run evaluation
CUDA_VISIBLE_DEVICES=$DEVICE python main.py \
    --exp-config $EXP_CONFIG \
    --split_num $SPLIT_NUM \
    --split_index $SPLIT_INDEX \
    --GSAM2_server_port $GSAM2_SERVER_PORT \
    --dataset $DATASET \
    --experiment_id $EXPERIMENT_ID \
    SIMULATOR_GPU_IDS [0] \
    TORCH_GPU_IDS [0] \
    GPU_NUMBERS 1 \
    NUM_ENVIRONMENTS 1 \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True

echo "Evaluation completed!"