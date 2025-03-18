from cbow_net import CbowNegSampling
from create_dataset import CBOWDatasetNegSampling, CBOWDatasetNegSamplingOptimized
from config_parser import get_train_cbow_neg_parser

import logging
import sys
import os
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import shutil
import pandas as pd
from test_similarity import evaluate_on_wordsim353, visualize_embeddings


DATASETS_DIR = os.getenv("DATASETS_DIR", "/home/KeLi/datasets")




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
    
    # Load wordsim353 dataset
    wordsim353_path = os.path.join(DATASETS_DIR, "wordsim353/combined.csv")
    df = pd.read_csv(wordsim353_path)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    loss_list = []
    for epoch in range(args.epochs):
        
        total_loss = 0
        for i, (context, target, negative_samples) in enumerate(dataloader):
           
            context = context.to(device)
            target = target.to(device)
            negative_samples = negative_samples.to(device)
            
            loss = model(context, target, negative_samples)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())
            total_loss += loss.item()
            if i % 100 == 0:
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
    parser = get_train_cbow_neg_parser()
    args = parser.parse_args()
    main(args)
    