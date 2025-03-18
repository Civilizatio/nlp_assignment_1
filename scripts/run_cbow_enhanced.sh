#!/bin/bash


export DATASET_DIR="/home/KeLi/datasets"
export NLTK_DATA="/home/KeLi/datasets/nltk_data"

source /home/KeLi/anaconda3/bin/activate
conda activate nlp


embedding_dims=(100 300 500)

# 基础命令部分，不含 embedding_dim 和 exp_dir
base_command="python cbow/train_enhanced_cbow.py \
    --batch_size 12800 \
    --context_size 2 \
    --num_neg_samples 5 \
    --power 0.75 \
    --epochs 10 \
    --lr 0.002 \
    --syn_weight 0.3 \
    --hyper_weight 0.2 \
    --dataset_path wiki_1000.txt \
    --eval_freq 1000 \
    --cuda 5"

# 循环执行不同 embedding_dim 的实验
for dim in "${embedding_dims[@]}"
do
    # 构建 exp_dir 名称
    exp_dir="exps/cbow_enhanced_1000_${dim}"
    # 构建完整命令
    full_command="${base_command} --embedding_dim ${dim} --exp_dir ${exp_dir}"

    echo "Running command: ${full_command}"
    # 执行命令
    eval $full_command

    if [ $? -eq 0 ]; then
        echo "Experiment with embedding_dim=${dim} completed successfully."
    else
        echo "Error occurred while running experiment with embedding_dim=${dim}."
    fi
done