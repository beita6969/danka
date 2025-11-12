#!/bin/bash

# Train Workflow Optimizer - Single A100 80G Configuration
#
# Optimized for single GPU training with memory-efficient settings

set -e

echo "=========================================="
echo "Qwen3-8B Workflow Optimizer - Single GPU"
echo "=========================================="
echo ""

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LD_LIBRARY_PATH="/usr/lib64-nvidia:${LD_LIBRARY_PATH}"

# Check GPU
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    echo "Please set: export OPENAI_API_KEY='your-key'"
    exit 1
fi
echo "✓ OpenAI API key found"

# Check training data
if [ ! -f "data/rl_training_data_full/train_data.jsonl" ]; then
    echo "❌ Error: Training data not found"
    echo ""
    echo "Please run data conversion:"
    echo "  cd /home/claude-user"
    echo "  python scripts/convert_all_evaluations.py --output data/rl_training_data_full"
    exit 1
fi

# Count training samples
TRAIN_SAMPLES=$(wc -l < data/rl_training_data_full/train_data.jsonl)
echo "✓ Training data found: $TRAIN_SAMPLES samples"
echo ""

# Training parameters
MAX_STEPS=${1:-500}  # Default 500 for testing
BATCH_SIZE=${2:-8}
LEARNING_RATE=${3:-1.0e-6}

echo "Training Configuration:"
echo "  GPU: Single A100 80G"
echo "  Max steps: $MAX_STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Gradient accumulation: 32"
echo "  Effective batch: $(($BATCH_SIZE * 32)) samples"
echo ""

# Memory monitoring
echo "Pre-training GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo ""

# Create checkpoint directory
mkdir -p data/checkpoints/qwen3-8b-workflow-optimizer-single-gpu
mkdir -p data/checkpoints/qwen3-8b-workflow-optimizer-single-gpu/logs

echo "Starting training..."
echo "Log file: training_single_gpu.log"
echo ""

# Run training
python examples/start_rlvr_pipeline.py \
  --config_name examples/qwen3-8B-workflow-optimizer/workflow_optimizer_single_gpu.yaml \
  max_steps=$MAX_STEPS \
  rollout_batch_size=$BATCH_SIZE \
  actor_train.training_args.learning_rate=$LEARNING_RATE \
  2>&1 | tee training_single_gpu.log

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Checkpoints: data/checkpoints/qwen3-8b-workflow-optimizer-single-gpu/checkpoints"
echo "Logs: training_single_gpu.log"
echo "TensorBoard: data/checkpoints/qwen3-8b-workflow-optimizer-single-gpu/tensorboard"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir data/checkpoints/qwen3-8b-workflow-optimizer-single-gpu/tensorboard"
echo ""
echo "To check final GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
