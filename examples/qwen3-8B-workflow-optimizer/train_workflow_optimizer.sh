#!/bin/bash

# Train Workflow Optimizer Script
#
# This script trains qwen3-8b to optimize agent workflows using ROLL + GRPO

set -e  # Exit on error

echo "======================================"
echo "Training Qwen3-8B Workflow Optimizer"
echo "======================================"

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Use 8 GPUs
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set (needed for workflow execution reward)"
    echo "Please set: export OPENAI_API_KEY='your-key'"
    exit 1
fi

# Check for training data
if [ ! -f "data/rl_training_data/train_data.jsonl" ]; then
    echo "Error: Training data not found at data/rl_training_data/train_data.jsonl"
    echo "Please run data collection and conversion first:"
    echo "  1. python scripts/collect_aflow_experience.py"
    echo "  2. python scripts/convert_trajectories.py"
    exit 1
fi

# Training parameters (can override via command line)
MAX_STEPS=${1:-1000}
BATCH_SIZE=${2:-16}
LEARNING_RATE=${3:-1.0e-6}

echo "Training Configuration:"
echo "  Max steps: $MAX_STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo ""

# Run training
python examples/start_rlvr_pipeline.py \
  --config_name examples/qwen3-8B-workflow-optimizer/workflow_optimizer_config.yaml \
  max_steps=$MAX_STEPS \
  rollout_batch_size=$BATCH_SIZE \
  actor_train.training_args.learning_rate=$LEARNING_RATE \
  2>&1 | tee training_log.txt

echo ""
echo "======================================"
echo "Training Complete!"
echo "======================================"
echo "Checkpoints saved to: data/checkpoints/qwen3-8b-workflow-optimizer"
echo "Logs saved to: training_log.txt"
