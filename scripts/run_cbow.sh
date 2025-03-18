#!/bash/bin

export DATASET_DIR="/home/KeLi/datasets"
export NLTK_DATA="/home/KeLi/datasets/nltk_data"

source /home/KeLi/anaconda3/bin/activate
conda activate nlp

#!/bin/bash

# 定义不同的 embedding_dim 值
embedding_dims=(100 300 500)

# 基础命令部分，不含 embedding_dim 和 exp_dir
base_command="python cbow/train_cbow.py \
    --batch_size 2560 \
    --context_size 2 \
    --epochs 10 \
    --lr 0.002 \
    --dataset_path \"wiki_1000.txt\" \
    --cuda \"4\""

# 循环执行不同 embedding_dim 的实验
for dim in "${embedding_dims[@]}"
do
    # 构建 exp_dir 名称
    exp_dir="exps/cbow_1000_${dim}"
    # 构建完整命令
    full_command="${base_command} --embedding_dim ${dim} --exp_dir \"${exp_dir}\""

    echo "Running command: ${full_command}"
    # 执行命令
    eval $full_command

    if [ $? -eq 0 ]; then
        echo "Experiment with embedding_dim=${dim} completed successfully."
    else
        echo "Error occurred while running experiment with embedding_dim=${dim}."
    fi
done

