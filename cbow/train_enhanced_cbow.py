# Leveraging external word knowledge sources to
# improve word embeddings
from cbow_net import CbowNegSampling
from create_dataset import CBOWDatasetNegSampling, CBOWDatasetNegSamplingOptimized
from config_parser import get_train_enhanced_cbow_parser
from test_similarity import evaluate_on_wordsim353, visualize_embeddings

import logging
import sys
import os
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import shutil

DATASETS_DIR = os.getenv("DATASETS_DIR", "/home/KeLi/datasets")
import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from scipy.stats import spearmanr

def build_knowledge_cache(vocab):
    """ Build a knowledge cache for each word in the vocabulary"""
    
    kg_cache =defaultdict(dict)
    for word in tqdm(vocab,desc="Building knowledge cache"):
        synsets = wn.synsets(word)
        relations = {
            'synonyms':set(), # siblings
            'hypernyms':set(), # parent
        }
        
        for synset in synsets:
            
            # Synonyms
            for lemma in synset.lemmas():
                relations['synonyms'].add(lemma.name().lower())
            
            # Hypernyms
            for hypernym in synset.hypernyms():
                relations['hypernyms'].update([lemma.name().lower() for lemma in hypernym.lemmas()])
                
        kg_cache[word] = relations
    return kg_cache

def calculate_knowledge_loss(model,target,dataset,syn_weight,hyper_weight,kg_cache,device):
    """ Calculate knowledge loss based on external knowledge cache"""
    
    loss = torch.tensor(0.0).to(device)
    
    target_words = [
        dataset.idx2word[idx.item()] for idx in target
    ]
    
    for word in target_words:
        relations = kg_cache.get(word,{})
        if not relations:
            continue
        
        word_idx = dataset.word2idx[word]
        valid_synonyms = [dataset.word2idx[synonym] for synonym in relations.get("synonyms",[]) if synonym in dataset.word2idx]
        valid_hypernyms = [dataset.word2idx[hypernym] for hypernym in relations.get("hypernyms",[]) if hypernym in dataset.word2idx]
        
        # Synonyms
        if valid_synonyms:
            syn_indices = torch.tensor(valid_synonyms).to(device)
            syn_embeds = model.embeddings(syn_indices)
            word_embeds = model.embeddings(torch.tensor(word_idx).to(device))
            loss += syn_weight * F.mse_loss(word_embeds, syn_embeds.mean(dim=0))
            
        # Hypernyms
        if valid_hypernyms:
            hyper_indices = torch.tensor(valid_hypernyms).to(device)
            hyper_embeds = model.embeddings(hyper_indices)
            word_embeds = model.embeddings(torch.tensor(word_idx).to(device))
            cos_sim = F.cosine_similarity(word_embeds, hyper_embeds)
            loss += hyper_weight * (1-cos_sim).mean()
            
    return loss/len(target_words)





def main(args):

    
    exp_dir = os.path.join(args.exp_dir)
    try:
        os.makedirs(exp_dir)
    except FileExistsError:
        shutil.rmtree(exp_dir)
        os.makedirs(exp_dir)
    
    # Logging configuration
    logger = logging.getLogger(f"Train CBOW Neg Sampling")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(exp_dir, 'cbow_neg_sampling.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Load dataset
    dataset_path = os.path.join(DATASETS_DIR, args.dataset_path)
    logger.info(f"Dataset: {args.dataset_path}")
    
    dataset = CBOWDatasetNegSamplingOptimized(dataset_path, context_size=args.context_size, num_neg_samples=args.num_neg_samples,power=args.power)
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Save word2idx
    dataset.save_word2idx(os.path.join(exp_dir, "word2idx.txt"))
    logger.info(f"word2idx saved to {os.path.join(exp_dir, 'word2idx.txt')}")
    
    # Set device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load model
    model = CbowNegSampling(len(dataset.vocab), args.embedding_dim)
    model.to(device)
    
    kg_cache = build_knowledge_cache(dataset.vocab)
    
    # Load wordsim353 dataset
    wordsim353_path = os.path.join(DATASETS_DIR, "wordsim353/combined.csv")
    df = pd.read_csv(wordsim353_path)
    

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    scaler = torch.amp.GradScaler("cuda")
    
    # Train model
    loss_list = []
    for epoch in range(args.epochs):
        
        total_loss = 0
        pbar = tqdm(enumerate(dataloader),total=len(dataloader),desc=f"Epoch {epoch}")
        for i, (context, target, negative_samples) in pbar:
           
            context = context.to(device)
            target = target.to(device)
            negative_samples = negative_samples.to(device)
            
            with torch.amp.autocast("cuda"):
                loss = model(context, target, negative_samples)
            
                # Add knowledge loss
                knowledge_loss = calculate_knowledge_loss(model,target,dataset,args.syn_weight,args.hyper_weight,kg_cache,device)
                
                # Add knowledge loss to total loss
                loss += knowledge_loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            optimizer.zero_grad()
            
            loss_list.append(loss.item())
            total_loss += loss.item()
            if i % args.eval_freq == 0:
                logger.info(f"Epoch: {epoch}, Iter: {i}, Loss: {loss.item()}")
        else: 
            corr = evaluate_on_wordsim353(model,dataset.word2idx,df,device)
            logger.info(f"Epoch: {epoch}, Wordsim353 correlation: {corr}")
            visualize_embeddings(model,dataset,device,os.path.join(exp_dir,f"epoch_{epoch}.png"))
                
            logger.info(f"Epoch: {epoch}, Total loss: {total_loss}")
    
    # Training finished
    logger.info("Training finished.")
    
    # Plot loss
    fig = plt.figure()
    plt.plot(loss_list)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.savefig(os.path.join(exp_dir, "loss.png"))
    logger.info(f"Loss plot saved to {os.path.join(exp_dir, 'loss.png')}")
    
    
    # Save model
    model_path = os.path.join(exp_dir, f"cbow_{args.embedding_dim}_neg.pth")
    save_dict = {
        "model_state_dict": model.state_dict(),
        "word2idx": dataset.word2idx,
        "embedding_dim": args.embedding_dim,
        "vocab_size": len(dataset.vocab)
    }
    torch.save(save_dict, model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = get_train_enhanced_cbow_parser()
    args = parser.parse_args()
    main(args)
    
