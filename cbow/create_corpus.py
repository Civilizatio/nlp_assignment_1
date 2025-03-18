from gensim.corpora import WikiCorpus
from gensim.corpora.wikicorpus import tokenize
import os
from functools import partial
from nltk.corpus import stopwords
import sys
import logging
import re
import argparse


def custom_tokenizer(text: str, token_min_len: int = 3, token_max_len: int = 15, 
                    lower: bool = True, stopwords: set = None, alphas_only: bool = True):
    """ Tokenize a text string. """
    stopwords = set(stopwords) if stopwords is not None else set()
    tokens = tokenize(text, token_min_len, token_max_len, lower)
    
    if alphas_only:
        tokens = [token for token in tokens if re.match(r'^[\w\'"]+$', token)]
    
    return [token for token in tokens if token not in stopwords]

nltk_stopwords = set(stopwords.words('english'))
custom_tokenizer_with_stopwords = partial(
    custom_tokenizer,
    stopwords=nltk_stopwords,
    alphas_only=True,
)

def main():
    # parser
    parser = argparse.ArgumentParser(description="Create a corpus from a Wikipedia dump.")
    parser.add_argument("--all_data", action='store_true', help="Use all data.")
    parser.add_argument("--num_articles", type=int, default=100, help="The number of articles to use.")
    parser.add_argument("--target_path", type=str, default="wiki.txt", help="The target path.")
    args = parser.parse_args()

    # logger config
    logger = logging.getLogger("Create Wiki Corpus")
    logger.setLevel(logging.ESTOINFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # data process
    DATASETS_DIR = os.getenv("DATASETS_DIR", "/home/KeLi/datasets")
    dataset_path = "enwiki-20241201-pages-articles-multistream1.xml-p1p41242.bz2"
    target_path = args.target_path

    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"NLTK stopwords: {len(nltk_stopwords)}")
    logger.info(f"NLTK stopwords sample: {list(nltk_stopwords)[:10]}")

    # read data
    wiki = WikiCorpus(
        os.path.join(DATASETS_DIR, dataset_path),
        dictionary={},
        tokenizer_func=custom_tokenizer_with_stopwords,
        lower=True,
        token_min_len=2,
        token_max_len=15,
    )

    # write data
    with open(os.path.join(DATASETS_DIR, target_path), 'w') as f:
        for i, text in enumerate(wiki.get_texts()):
            f.write(' '.join(text) + '\n')
            if not args.all_data and i >= args.num_articles:
                break
    logger.info(f"Finished processing articles")

    # close logger
    logger.removeHandler(sh)
    sh.close()

if __name__ == "__main__":
    main() 