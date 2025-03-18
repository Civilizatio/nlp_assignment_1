import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from nltk.corpus import wordnet as wn


class Cbow(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Cbow,self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
        # Initialize weights using uniform distribution for weights and zero for bias
        
        initrange = 0.5
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        
        
    def forward(self, inputs):
        # inputs: [batch_size, context_size]
        embeds = self.embeddings(inputs) # [batch_size, context_size, embedding_dim]
        out = torch.sum(embeds, dim=1) # [batch_size, embedding_dim]
        out = self.linear(out) # [batch_size, vocab_size]
        
        return out
    

class CbowNegSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CbowNegSampling,self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings_out = nn.Embedding(vocab_size, embedding_dim)
        
        self.init_weights()
        
    def init_weights(self):
        # Initialize weights using uniform distribution for weights and zero for bias
        
        initrange = 0.5
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.embeddings_out.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, context, target, negative_samples):
        # context: [batch_size, context_size]
        # target: [batch_size]
        # negative_samples: [batch_size, num_neg_samples]
        
        embeds = self.embeddings(context) # [batch_size, context_size, embedding_dim]
        embeds = torch.sum(embeds, dim=1) # [batch_size, embedding_dim]
        
        target_embeds = self.embeddings_out(target) # [batch_size, embedding_dim]
        target_score = torch.mul(embeds, target_embeds).sum(dim=1) # [batch_size]
        
        negative_embeds = self.embeddings_out(negative_samples) # [batch_size, num_neg_samples, embedding_dim]
        negative_score = torch.bmm(negative_embeds, embeds.unsqueeze(2)).squeeze() # [batch_size, num_neg_samples]
        
        return -torch.mean(torch.log(torch.sigmoid(target_score))) - torch.mean(torch.log(torch.sigmoid(-negative_score))) 
    
    
class KnowledgeEnhancedCBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, syn_weight=0.3,hyper_weight=0.2):
        super(KnowledgeEnhancedCBOW,self).__init__()
        
        self.embedding_dim = embedding_dim
        self.syn_weight = syn_weight
        self.hyper_weight = hyper_weight
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings_out = nn.Embedding(vocab_size, embedding_dim)
        self.init_weights()
        
        
    def init_weights(self):
        # Initialize weights using uniform distribution for weights and zero for bias
        
        initrange = 0.5
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.embeddings_out.weight.data.uniform_(-initrange, initrange)
        
    
    
    def forward(self, context, target, negative_samples):
        # context: [batch_size, context_size]
        # target: [batch_size]
        # negative_samples: [batch_size, num_neg_samples]
        
        context_embeds = self.embeddings(context)
        context_embeds = torch.sum(context_embeds, dim=1)
        
        target_embeds = self.embeddings_out(target)
        target_score = torch.mul(context_embeds, target_embeds).sum(dim=1)
        
        negative_embeds = self.embeddings_out(negative_samples)
        negative_score = torch.bmm(negative_embeds, context_embeds.unsqueeze(2)).squeeze()
        
        # NCE loss
        nce_loss = -torch.mean(torch.log(torch.sigmoid(target_score))) - torch.mean(torch.log(torch.sigmoid(-negative_score)))
        
        
        return nce_loss 
    
