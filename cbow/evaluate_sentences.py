import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import argparse
import logging

from cbow_net import KnowledgeEnhancedCBOW, CbowNegSampling, Cbow
from create_corpus import custom_tokenizer_with_stopwords


logger = logging.getLogger(f"Evaluate Sentences")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh = logging.FileHandler("evaluate_sentences.log")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)


MODEL_PATH = os.getenv("MODEL_PATH", "/home/KeLi/models")
DATASET_DIR = os.getenv("DATASET_DIR", "/home/KeLi/datasets")

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="msr_paraphrase_test.txt")
parser.add_argument("--model_path", type=str, default="openbmb/MiniCPM-1B-sft-bf16")
parser.add_argument(
    "--model_type",
    type=int,
    default=0,
    help="whether pretrained (0) or implement by self (1)",
)
parser.add_argument("--cuda", type=int, default=5)
args = parser.parse_args()


def load_msrpc_data(path):

    df = pd.read_csv(
        path,
        sep="\t",
        quoting=3,
        on_bad_lines="skip",
        header=None,
        names=["Quality", "#1 ID", "#2 ID", "#1 String", "#2 String"],
    )

    # 强制转换 Quality 列为数值并过滤非法值
    df["Quality"] = pd.to_numeric(df["Quality"], errors="coerce")
    df = df.dropna(subset=["Quality"])
    df["Quality"] = df["Quality"].astype(int)
    df = df[df["Quality"].isin([0, 1])]  # 仅保留 0/1
    return df[["#1 String", "#2 String", "Quality"]].dropna()


def load_model(model_path):
    pre_trained_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(MODEL_PATH, model_path), trust_remote_code=True
    )
    pre_trained_model = AutoModelForCausalLM.from_pretrained(
        os.path.join(MODEL_PATH, model_path), trust_remote_code=True
    )
    return pre_trained_model, pre_trained_tokenizer


def load_model_(model_path):
    configs = torch.load(model_path)

    word2idx = configs["word2idx"]
    vocab_size = configs["vocab_size"]
    embedding_dim = configs["embedding_dim"]

    if "enhanced" in model_path:
        model = KnowledgeEnhancedCBOW(vocab_size, embedding_dim)
    elif "neg" in model_path:
        model = CbowNegSampling(vocab_size, embedding_dim)
    else:
        model = Cbow(vocab_size, embedding_dim)

    model.load_state_dict(configs["model_state_dict"])

    tokenizer = custom_tokenizer_with_stopwords
    return model, tokenizer, word2idx


def compute_bert_style_similarity(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    sentence1: str,
    sentence2: str,
    device: str = "cuda",
) -> float:

    def get_token_embeddings(sentence):
        inputs = tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1][0]  # [1, seq_len, hidden_size]
        return embeddings.cpu().numpy()

    embedding_1 = get_token_embeddings(sentence1)
    embedding_2 = get_token_embeddings(sentence2)

    sim_matrix = cosine_similarity(embedding_1, embedding_2)  # [seq_len1, seq_len2]

    P = np.mean(np.max(sim_matrix, axis=1))
    R = np.mean(np.max(sim_matrix, axis=0))

    if P + R == 0:
        return 0
    return 2 * P * R / (P + R)


def compute_bert_style_similarity_(
    model: Cbow,
    tokenizer,
    word2idx,
    sentence1: str,
    sentence2: str,
    device: str = "cuda",
) -> float:

    def get_token_embeddings(sentence):
        tokens = tokenizer(sentence)
        token_ids = [word2idx.get(token, 0) for token in tokens]
        token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

        with torch.no_grad():
            embeddings = model.embeddings(token_ids)

        return embeddings.cpu().squeeze().numpy()

    embedding_1 = get_token_embeddings(sentence1)
    embedding_2 = get_token_embeddings(sentence2)

    sim_matrix = cosine_similarity(embedding_1, embedding_2)

    P = np.mean(np.max(sim_matrix, axis=1))
    R = np.mean(np.max(sim_matrix, axis=0))

    if P + R == 0:
        return 0
    return 2 * P * R / (P + R)


def get_sentence_embedding(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sentence: str,
    device: str = "cuda",
    pooling: str = "mean",
) -> np.ndarray:
    """Get sentence embedding from a pre-trained model.

    Args:
        model: Pre-trained model.
        tokenizer: Tokenizer.
        sentence: Input sentence.
        device: Device to run the model.
        pooling: Pooling strategy.

    Returns:
        Sentence embedding.

    """
    # No sentence
    if not sentence.strip():
        return np.zeros(model.config.hidden_size)

    # input encoding
    inputs = tokenizer(
        sentence, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    # forward
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # last hidden state
    last_hidden = outputs.hidden_states[-1]

    # pooling
    if pooling == "cls":
        embedding = last_hidden[:, 0, :]  # [CLS]
    elif pooling == "max":
        embedding = last_hidden.max(dim=1).values
    else:  # mean pooling
        # mask
        mask = inputs["attention_mask"].unsqueeze(-1)
        embedding = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)

    # 归一化处理
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.cpu().numpy().squeeze()  # [1, hidden_size] -> [hidden_size]


def get_sentence_embedding_(
    model: Cbow,
    tokenizer,
    word2idx,
    sentence: str,
    device: str = "cuda",
    pooling: str = "mean",
) -> np.ndarray:

    # No sentence
    if not sentence.strip():
        return np.zeros(model.embeddings.weight.size(1))

    tokens = tokenizer(sentence)
    token_ids = [word2idx.get(token, 0) for token in tokens]
    token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.embeddings(token_ids)

    if pooling == "mean":
        embedding = torch.mean(embedding, dim=1)
    elif pooling == "max":
        embedding = torch.max(embedding, dim=1)
    else:
        embedding = torch.sum(embedding, dim=1)

    return embedding.cpu().numpy().squeeze()


def compute_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings.

    Args:
        embedding1: Sentence embedding 1.
        embedding2: Sentence embedding 2.

    Returns:
        Cosine similarity.

    """
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]


def find_optimal_threshold(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]


if __name__ == "__main__":

    # Load data
    data = load_msrpc_data(os.path.join(DATASET_DIR, args.data_path))

    # Load model
    if args.model_type == 0:
        model, tokenizer = load_model(os.path.join(MODEL_PATH, args.model_path))
    else:
        model, tokenizer, word2idx = load_model_(
            os.path.join("exps", args.model_path)
        )

    model.to(args.cuda)
    
    logger.info("="*50)
    logger.info(f"Evaluating Model loaded: {args.model_path}")
    logger.info("="*50)
    

    # Get sentence embeddings
    data["#1 Embedding"] = data["#1 String"].apply(
        lambda x: (
            get_sentence_embedding(model, tokenizer, x, device=args.cuda)
            if args.model_type == 0
            else get_sentence_embedding_(
                model, tokenizer, word2idx, x, device=args.cuda
            )
        )
    )
    data["#2 Embedding"] = data["#2 String"].apply(
        lambda x: (
            get_sentence_embedding(model, tokenizer, x, device=args.cuda)
            if args.model_type == 0
            else get_sentence_embedding_(
                model, tokenizer, word2idx, x, device=args.cuda
            )
        )
    )

    # Compute similarity
    data["Similarity"] = data.apply(
        lambda x: compute_similarity(x["#1 Embedding"], x["#2 Embedding"]), axis=1
    )

    # Find optimal threshold
    threshold = find_optimal_threshold(data["Quality"], data["Similarity"])
    logger.info(f"Optimal threshold: {threshold}")

    # Compute accuracy
    y_pred = data["Similarity"] > threshold
    acc = accuracy_score(data["Quality"], y_pred)
    f1 = f1_score(data["Quality"], y_pred)
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1: {f1:.4f}")

    # Compute BERT-style similarity
    data["BERT Similarity"] = data.apply(
        lambda row: (
            compute_bert_style_similarity(
                model,
                tokenizer,
                row["#1 String"],
                row["#2 String"],
                device=f"cuda:{args.cuda}",
            )
            if args.model_type == 0
            else compute_bert_style_similarity_(
                model,
                tokenizer,
                word2idx,
                row["#1 String"],
                row["#2 String"],
                device=f"cuda:{args.cuda}",
            )
        ),
        axis=1,
    )

    bert_threshold = find_optimal_threshold(data["Quality"], data["BERT Similarity"])
    logger.info(f"Optimal BERT-style Threshold: {bert_threshold:.4f}")

    # Compute accuracy
    y_pred = data["BERT Similarity"] > bert_threshold
    acc = accuracy_score(data["Quality"], y_pred)
    f1 = f1_score(data["Quality"], y_pred)
    logger.info(f"BERT-style Accuracy: {acc:.4f}")
    logger.info(f"BERT-style F1: {f1:.4f}")

