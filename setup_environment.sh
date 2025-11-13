#!/bin/bash
################################################################################
# ROLL Qwen2.5-7B Workflow Optimizer - 环境配置脚本
# 一键配置完整训练环境
################################################################################

set -e  # 遇到错误立即退出

echo "=========================================="
echo "ROLL 工作流优化器环境配置"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# 检查是否为root用户
if [ "$EUID" -eq 0 ]; then
    print_error "请不要使用root用户运行此脚本"
    exit 1
fi

# 1. 检查系统环境
echo ""
echo "步骤 1/8: 检查系统环境..."
if command -v nvidia-smi &> /dev/null; then
    print_success "检测到 NVIDIA GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_error "未检测到 NVIDIA GPU 或驱动未安装"
    exit 1
fi

# 2. 检查CUDA
echo ""
echo "步骤 2/8: 检查 CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    print_success "CUDA 版本: $CUDA_VERSION"
else
    print_warning "nvcc 未找到，将尝试使用系统 CUDA"
fi

# 设置CUDA环境变量
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
print_success "CUDA 环境变量已设置"

# 3. 检查Python环境
echo ""
echo "步骤 3/8: 检查 Python 环境..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "$PYTHON_VERSION"
else
    print_error "Python3 未安装"
    exit 1
fi

# 4. 安装Python依赖
echo ""
echo "步骤 4/8: 安装 Python 依赖包..."
if [ -f "requirements.txt" ]; then
    print_warning "正在安装依赖包，这可能需要几分钟..."
    pip3 install -r requirements.txt -q || {
        print_error "依赖包安装失败"
        exit 1
    }
    print_success "依赖包安装完成"
else
    print_warning "requirements.txt 未找到，跳过依赖安装"
fi

# 5. 安装ROLL框架
echo ""
echo "步骤 5/8: 安装 ROLL 框架..."
if [ -f "setup.py" ]; then
    pip3 install -e . -q || {
        print_error "ROLL 安装失败"
        exit 1
    }
    print_success "ROLL 框架安装完成"
else
    print_warning "setup.py 未找到，假设 ROLL 已安装"
fi

# 6. 检查数据集
echo ""
echo "步骤 6/8: 检查训练数据集..."
if [ -f "data/rl_training_data_full/train_data.jsonl" ]; then
    TRAIN_SAMPLES=$(wc -l < data/rl_training_data_full/train_data.jsonl)
    VAL_SAMPLES=$(wc -l < data/rl_training_data_full/val_data.jsonl)
    print_success "训练集: $TRAIN_SAMPLES 样本"
    print_success "验证集: $VAL_SAMPLES 样本"
else
    print_error "训练数据集未找到！"
    print_error "请确保 data/rl_training_data_full/ 目录包含训练数据"
    exit 1
fi

# 7. 检查配置文件
echo ""
echo "步骤 7/8: 检查训练配置..."
CONFIG_FILE="examples/qwen3-8B-workflow-optimizer/workflow_optimizer_full_training.yaml"
if [ -f "$CONFIG_FILE" ]; then
    print_success "配置文件存在: $CONFIG_FILE"

    # 显示关键配置
    echo ""
    echo "关键配置参数:"
    grep "max_steps:" "$CONFIG_FILE" | head -1
    grep "rollout_batch_size:" "$CONFIG_FILE" | head -1
    grep "pretrain:" "$CONFIG_FILE" | head -1
else
    print_error "配置文件未找到: $CONFIG_FILE"
    exit 1
fi

# 8. 创建输出目录
echo ""
echo "步骤 8/8: 创建输出目录..."
mkdir -p output/logs
mkdir -p output/tensorboard
mkdir -p data/checkpoints
print_success "输出目录已创建"

# 完成
echo ""
echo "=========================================="
print_success "环境配置完成！"
echo "=========================================="
echo ""
echo "下一步操作:"
echo "1. 启动训练:"
echo "   ./run_training.sh"
echo ""
echo "2. 监控训练进度:"
echo "   tail -f training_1000steps.log"
echo ""
echo "3. 查看 TensorBoard:"
echo "   tensorboard --logdir=./output/tensorboard"
echo ""
