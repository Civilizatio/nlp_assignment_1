#!/bin/bash
# evaluate_models.sh

source /home/KeLi/anaconda3/bin/activate
conda activate nlp

export MODEL_PATH="/home/KeLi/models"
export DATASET_DIR="/home/KeLi/datasets"

# cuda
CUDA_DEVICE=5

# model type list
declare -a model_configs=(
    # 预训练模型
    # "openbmb/MiniCPM-1B-sft-bf16 0"
    
    # 标准 CBOW 模型
    "cbow_1000_100/cbow_100.pth 1"
    "cbow_1000_300/cbow_300.pth 1"
    # "cbow_1000_500/cbow_500.pth 1"
    
    # 负采样 CBOW 模型
    "cbow_neg_1000_100/cbow_100_neg.pth 1"
    "cbow_neg_1000_300/cbow_300_neg.pth 1"
    "cbow_neg_1000_500/cbow_500_neg.pth 1"
    
    # 知识增强 CBOW 模型
    "cbow_enhanced_1000_100/cbow_100_neg.pth 1"
    # "cbow_enhanced_1000_300 1"
    # "cbow_enhanced_1000_500 1"
)

# 遍历所有配置
for config in "${model_configs[@]}"; do
    # 解析配置
    model_path=$(echo $config | cut -d' ' -f1)
    model_type=$(echo $config | cut -d' ' -f2)
    
    # 运行评估命令
    echo "======================================================================="
    echo "Evaluating model: $model_path (type $model_type)"
    echo "======================================================================="
    
    python cbow/evaluate_sentences.py \
        --data_path "msr_paraphrase_test.txt" \
        --model_path "$model_path" \
        --model_type "$model_type" \
        --cuda "$CUDA_DEVICE"
    
    echo -e "\n\n"
done

