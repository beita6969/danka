#!/bin/bash
################################################################################
# ROLL Qwen2.5-7B Workflow Optimizer - 训练启动脚本
# 一键启动完整训练流程
################################################################################

set -e

echo "=========================================="
echo "启动 ROLL 工作流优化器训练"
echo "=========================================="

# 设置环境变量
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# 配置文件
CONFIG_PATH="qwen3-8B-workflow-optimizer"
CONFIG_NAME="workflow_optimizer_full_training"
LOG_FILE="training_1000steps.log"

echo "配置路径: $CONFIG_PATH"
echo "配置名称: $CONFIG_NAME"
echo "日志文件: $LOG_FILE"
echo ""

# 检查是否已有训练在运行
if pgrep -f "start_rlvr_pipeline.py" > /dev/null; then
    echo "检测到已有训练进程在运行！"
    echo "是否停止现有训练并重新开始? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "停止现有训练..."
        pkill -f "start_rlvr_pipeline.py"
        ray stop --force
        sleep 3
        echo "已停止现有训练"
    else
        echo "退出..."
        exit 0
    fi
fi

# 启动训练
echo "启动训练进程..."
nohup python examples/start_rlvr_pipeline.py \
    --config_path "$CONFIG_PATH" \
    --config_name "$CONFIG_NAME" \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "训练已在后台启动 (PID: $TRAIN_PID)"
echo ""

# 等待几秒让训练初始化
sleep 5

# 检查进程是否还在运行
if ps -p $TRAIN_PID > /dev/null; then
    echo "✓ 训练进程运行正常"
    echo ""
    echo "监控命令:"
    echo "  实时日志: tail -f $LOG_FILE"
    echo "  查看进程: ps aux | grep start_rlvr_pipeline"
    echo "  停止训练: pkill -f start_rlvr_pipeline.py && ray stop"
    echo ""
    echo "初始日志输出:"
    echo "----------------------------------------"
    tail -20 "$LOG_FILE"
else
    echo "✗ 训练进程启动失败"
    echo "请检查日志: cat $LOG_FILE"
    exit 1
fi
