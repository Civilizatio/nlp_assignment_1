import torch
from torch.utils.data import Dataset
from collections import Counter
import numpy as np

class CBOWDataset(Dataset):
    def __init__(self, file_path, context_size=2):
        """ """
        self.context_size = context_size
        self.corpus = self._load_data(file_path)
        self.vocab = self._build_vocab(self.corpus)
        self.pairs = self.generate_pairs(self.corpus)
        
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
    def _load_data(self, file_path):
        with open(file_path, 'r') as f:
            corpus = f.readlines()
            
        # For example: [['I', 'love', 'you'], ['I', 'hate', 'you']]
        corpus = [sentence.strip().split() for sentence in corpus] 
        
        return corpus
    
    def _build_vocab(self, corpus):
        vocab = set()
        for sentence in corpus:
            for word in sentence:
                vocab.add(word)
        vocab.add('<UNK>') # Unknown token
        return list(vocab)
        
    def generate_pairs(self, corpus):
        pairs = []
        for sentence in corpus:
            for i, target in enumerate(sentence):
                context = [
                    sentence[j] if j >= 0 and j < len(sentence) else '<UNK>'
                    for j in range(i - self.context_size, i + self.context_size + 1)
                    if j != i
                ]
                pairs.append((context, target))
        return pairs
    
    def save_word2idx(self, file_path):
        # Save word2idx to file for later testing
        with open(file_path, 'w') as f:
            for word, idx in self.word2idx.items():
                f.write(f"{word}\t{idx}\n")
                
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        context, target = self.pairs[idx]
        context = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in context]
        target = self.word2idx.get(target, self.word2idx['<UNK>'])
        return torch.tensor(context,dtype=torch.long), torch.tensor(target,dtype=torch.long)

class CBOWDatasetNegSampling(Dataset):
    def __init__(self, file_path, context_size=2,num_neg_samples=5,power=0.75):
        """ CBOW datasets with negative sampling
        
        Args:
            file_path (str): The path to the dataset file.
            context_size (int): The context size.
            num_neg_samples (int): The number of negative samples.
            power (float): The power for negative sampling.
            
        
        """
        
        self.context_size = context_size
        self.num_neg_samples = num_neg_samples
        self.power = power
        self.corpus = self._load_data(file_path)
        self.vocab, self.word_freqs = self._build_vocab(self.corpus)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # Compute sample weights for negative sampling
        self.sample_weights = self._compute_sample_weights(self.word_freqs)
        
        # Generate pairs:
        # (context, target, negative_samples)
        self.pairs = self.generate_pairs(self.corpus)
        
        
    def _load_data(self, file_path):
        with open(file_path, 'r') as f:
            corpus = f.readlines()
            
        # For example: [['I', 'love', 'you'], ['I', 'hate', 'you']]
        corpus = [sentence.strip().split() for sentence in corpus] 
        
        return corpus
    
    def _build_vocab(self, corpus):
        
        word_counts = Counter()
        for sentence in corpus:
            word_counts.update(sentence)
        
        # Sort words by frequency and add <UNK> token
        # by descending order
        vocab = ['<UNK>'] + sorted(word_counts, key=word_counts.get, reverse=True)
        
        word_freqs = np.array([word_counts.get(word, 0) for word in vocab], dtype=np.float32)
        
        return vocab, word_freqs
    
    def _compute_sample_weights(self, word_freqs):
        """ Compute sample weights for negative sampling.
        
        Args:
            word_freqs (np.array): The word frequencies.
        
        Returns:
            np.array: The sample weights.
        
        """
        # Compute the probability of each word
        weights = word_freqs ** self.power
        weights /= weights.sum()
        
        return weights
    
    def generate_negative_samples(self, target):
        """ Generate negative samples.
        
        Args:
            target (int): The target word index.
        
        Returns:
            np.array: The negative samples.
        
        """
        # Sample negative samples based on the computed weights
        # Adjust the weights to avoid sampling the target word
        # sample_weights = self.sample_weights.copy()
        # sample_weights[target] = 0
        
        negative_samples = np.random.choice(
            len(self.vocab),
            size=self.num_neg_samples,
            p=self.sample_weights,
            replace=False
        )
        
        # Ensure that the target word is not in the negative samples
        negative_samples = [sample if sample != target else 0 for sample in negative_samples]
        
        return negative_samples
    
    def generate_pairs(self, corpus):
        pairs = []
        for sentence in corpus:
            for i, target in enumerate(sentence):
                context = [
                    sentence[j] if j >= 0 and j < len(sentence) else '<UNK>'
                    for j in range(i - self.context_size, i + self.context_size + 1)
                    if j != i
                ]
                target_idx = self.word2idx.get(target, 0)
                context_idx = [self.word2idx.get(word, 0) for word in context]
                
                negative_samples =self.generate_negative_samples(target_idx)
                
                pairs.append((context_idx, target_idx, negative_samples))
        return pairs
    
    def save_word2idx(self, file_path):
        # Save word2idx to file for later testing
        with open(file_path, 'w') as f:
            for word, idx in self.word2idx.items():
                f.write(f"{word}\t{idx}\n")
                
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        
        context, target, negative_samples = self.pairs[idx]
        
        return (
            torch.tensor(context, dtype=torch.long), # size: [context_size]
            torch.tensor(target, dtype=torch.long), # size: [1]
            torch.tensor(negative_samples, dtype=torch.long) # size: [num_neg_samples]
        )
        
class CBOWDatasetNegSamplingOptimized(Dataset):
    def __init__(self, file_path, context_size=2, num_neg_samples=5, power=0.75):
        self.context_size = context_size
        self.num_neg_samples = num_neg_samples
        self.power = power
        self.corpus = self._load_data(file_path)
        self.vocab, self.word_freqs = self._build_vocab(self.corpus)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.sample_weights = self._compute_sample_weights(self.word_freqs)
        self.pairs = self.generate_pairs(self.corpus)

    def _load_data(self, file_path):
        with open(file_path, 'r') as f:
            corpus = f.readlines()
        corpus = [sentence.strip().split() for sentence in corpus]
        return corpus

    def _build_vocab(self, corpus):
        word_counts = Counter()
        for sentence in corpus:
            word_counts.update(sentence)
        vocab = ['<UNK>'] + sorted(word_counts, key=word_counts.get, reverse=True)
        word_freqs = np.array([word_counts.get(word, 0) for word in vocab], dtype=np.float32)
        return vocab, word_freqs

    def _compute_sample_weights(self, word_freqs):
        weights = word_freqs ** self.power
        weights /= weights.sum()
        return weights

    def generate_pairs(self, corpus):
        pairs = []
        for sentence in corpus:
            for i, target in enumerate(sentence):
                context = [
                    sentence[j] if 0 <= j < len(sentence) else '<UNK>'
                    for j in range(i - self.context_size, i + self.context_size + 1)
                    if j != i
                ]
                target_idx = self.word2idx.get(target, 0)
                context_idx = [self.word2idx.get(word, 0) for word in context]
                pairs.append((context_idx, target_idx))
        return pairs

    def save_word2idx(self, file_path):
        with open(file_path, 'w') as f:
            for word, idx in self.word2idx.items():
                f.write(f"{word}\t{idx}\n")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        context, target = self.pairs[idx]
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(target, dtype=torch.long)
        )

    def collate_fn(self, batch):
        contexts, targets = zip(*batch)
        contexts = torch.stack(contexts)  # (batch_size, context_size)
        targets = torch.stack(targets)    # (batch_size,)
        batch_size = targets.size(0)
        
        # Convert sample_weights to tensor and move to the same device as targets
        sample_weights = torch.from_numpy(self.sample_weights).float().to(targets.device) # (vocab_size,)
        
        # Expand weights to batch_size x vocab_size and exclude target indices
        sample_weights_expanded = sample_weights.unsqueeze(0).expand(batch_size, -1).clone() # (batch_size, vocab_size)
        rows = torch.arange(batch_size, device=targets.device)
        sample_weights_expanded[rows, targets] = 0 # Exclude target indices
        
        # Handle rows where sum becomes zero after exclusion
        row_sums = sample_weights_expanded.sum(dim=1, keepdim=True)
        zero_mask = (row_sums.squeeze(1) == 0)
        if zero_mask.any():
            # Fallback to original weights for zero-sum rows
            sample_weights_expanded[zero_mask] = sample_weights.unsqueeze(0).expand(zero_mask.sum().item(), -1)
            row_sums[zero_mask] = sample_weights_expanded[zero_mask].sum(dim=1, keepdim=True)
        
        # Normalize and sample negatives
        sample_weights_expanded_normalized = sample_weights_expanded / row_sums
        neg_samples = torch.multinomial(sample_weights_expanded_normalized, self.num_neg_samples, replacement=True)
        
        return contexts, targets, neg_samples