import torch
import pandas as pd
from scipy.stats import spearmanr
import torch.nn.functional as F

from cbow_net import Cbow, CbowNegSampling
from config_parser import get_test_parser


import os
import logging

DATASETS_DIR = os.getenv("DATASETS_DIR", "/home/KeLi/datasets")

def load_wordsim353(file_path):
    df = pd.read_csv(file_path, sep=",")
    return df

def cosine_similarity(v1, v2):
    return F.cosine_similarity(v1, v2, dim=0).item()

def evaluate_on_wordsim353(model, word2idx, df_wordsim353, device):
    
    # Extract words and human scores
    words1 = df_wordsim353["Word 1"].values
    words2 = df_wordsim353["Word 2"].values
    human_scores = df_wordsim353["Human (mean)"].values
    
    # Compute model scores
    model_scores = []
    for word1, word2 in zip(words1, words2):
        if word1 in word2idx and word2 in word2idx:
            idx1 = word2idx[word1]
            idx2 = word2idx[word2]
            v1 = model.embeddings(torch.tensor(idx1, dtype=torch.long).to(device))
            v2 = model.embeddings(torch.tensor(idx2, dtype=torch.long).to(device))
            model_scores.append(cosine_similarity(v1, v2))
        else:
            model_scores.append(0.5) # If word is not in vocabulary, assign 0.5
            
    # Compute Spearman correlation
    correlation, _ = spearmanr(human_scores, model_scores)
    return correlation

def visualize_embeddings(model,dataset,device,file_path,words=['bank', 'money', 'river', 'finance']):
    """ Visualize embeddings"""
    
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
   
    indices = [dataset.word2idx[word] for word in words]
    embeddings = model.embeddings(torch.tensor(indices).to(device))
    embeddings = embeddings.detach().cpu().numpy()
    
    # Reduce dimensionality
    tsne = TSNE(n_components=2, random_state=0, perplexity=3)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(6,6))
    for i, label in enumerate(words):
        x, y = embeddings_2d[i,:]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.title("Embeddings visualization")
    plt.savefig(file_path)
    plt.close()
    
 
def main(args):
    
    # Logging configuration
    logger = logging.getLogger(f"Test CBOW")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh= logging.FileHandler(os.path.join(args.exp_dir, 'cbow_test.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    
    # Load dictionary
    configs = torch.load(os.path.join(args.exp_dir, args.model_path))
    word2idx = configs["word2idx"]
    vocab_size = configs["vocab_size"]
    embedding_dim = configs["embedding_dim"]
    
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Embedding dimension: {embedding_dim}")
     
    # Set device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load model
    model = CbowNegSampling(vocab_size, embedding_dim) if args.neg_sampling else Cbow(vocab_size, embedding_dim)
    model.load_state_dict(configs["model_state_dict"])
    model.to(device)
    
    
    # Evaluate model
    wordsim353_path = os.path.join(DATASETS_DIR, args.wordsim_path)
    correlation = evaluate_on_wordsim353(model, word2idx, load_wordsim353(wordsim353_path), device)
    logger.info(f"Spearman correlation on wordsim353: {correlation}")
    

if __name__ == "__main__":
    
    parser = get_test_parser()
    args = parser.parse_args()
    main(args)
    