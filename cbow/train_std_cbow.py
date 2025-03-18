from gensim.models import Word2Vec
import os
import torch
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


DATASETS_DIR = os.getenv("DATASETS_DIR", "/home/KeLi/datasets")

def load_data(file_path):
    """ Load data from file"""
    with open(file_path, 'r') as f:
        corpus = f.readlines()
        
    # For example: [['I', 'love', 'you'], ['I', 'hate', 'you']]
    corpus = [sentence.strip().split() for sentence in corpus] 
    
    return corpus


file_path = "wiki.txt"
sentences = load_data(os.path.join(DATASETS_DIR,file_path))
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    negative=5,
    workers=4,
    sg=0, # CBOW
)

# Save model
model.save("word2vec_cbow.bin")

def evaluate_on_wordsim353(model,df_wordsim353):
    
    # Extract words and human scores
    words1 = df_wordsim353["Word 1"].values
    words2 = df_wordsim353["Word 2"].values
    human_scores = df_wordsim353["Human (mean)"].values
    
    # Compute model scores
    model_scores = []
    for word1, word2 in zip(words1, words2):
        
        try:
            v1 = model.wv[word1]
            v2 = model.wv[word2]
            score = cosine_similarity([v1],[v2])[0][0]
            model_scores.append(score)
        except KeyError:
            model_scores.append(0.5)
        
            
    # Compute Spearman correlation
    correlation, _ = spearmanr(human_scores, model_scores)
    return correlation

wordsim353_path = os.path.join(DATASETS_DIR,"wordsim353/combined.csv")
df_wordsim353 = pd.read_csv(wordsim353_path)

spearman_correlation = evaluate_on_wordsim353(model, df_wordsim353)
print(f"Spearman correlation on WordSim353: {spearman_correlation}")