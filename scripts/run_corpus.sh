#!/bin/bash

export DATASET_DIR="/home/KeLi/datasets"
export NLTK_DATA="/home/KeLi/datasets/nltk_data"

source /home/KeLi/anaconda3/bin/activate
conda activate nlp


python cbow/create_corpus.py \
    --num_articles 5000 \
    --target_path "wiki_5000.txt" \
    # --all_data
