import argparse

def get_base_parser():
    
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size.",
    )
    
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=100,
        help="The embedding dimension.",
    )
    
    parser.add_argument(
        "--context_size",
        type=int,
        default=2,
        help="The context size.",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The number of epochs.",
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="The learning rate.",
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="wiki.txt",
        help="The dataset path.",
    )
    
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="exp",
        help="The experiment directory.",
    )
    
    parser.add_argument(
        "--cuda",
        type=str,
        default="0",
        help="The device to use.",
    )
    return parser

def get_train_cbow_parser():
    
    base_parser = get_base_parser()
    parser = argparse.ArgumentParser(description="Train a word2vec model using cbow.", parents=[base_parser],add_help=True)    
    return parser

def get_train_cbow_neg_parser():
    
    base_parser = get_base_parser()
    parser = argparse.ArgumentParser(description="Train a word2vec model using cbow with negative sampling.", parents=[base_parser],add_help=True)
    parser.add_argument(
        "--num_neg_samples",
        type=int,
        default=5,
        help="The number of negative samples.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.75,
        help="The power for negative sampling.",
    )
    return parser

def get_train_enhanced_cbow_parser():
    
    base_parser = get_base_parser()
    parser = argparse.ArgumentParser(description="Train a word2vec model using enhanced cbow.", parents=[base_parser],add_help=True)
    parser.add_argument(
        "--num_neg_samples",
        type=int,
        default=5,
        help="The number of negative samples.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.75,
        help="The power for negative sampling.",
    )
    parser.add_argument(
        "--syn_weight",
        type=float,
        default=0.5,
        help="The weight for synonym loss.",
    )
    parser.add_argument(
        "--hyper_weight",
        type=float,
        default=0.5,
        help="The weight for hypernym loss.",
    )
    
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1,
        help="The evaluation frequency.",
    )
    return parser

def get_test_parser():
    parser = argparse.ArgumentParser(description="Test a word2vec model.")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path to the trained model file.",
    )
    
    parser.add_argument(
        "--neg_sampling",
        action="store_true",
        help="Whether to use negative sampling.",
    )
    
    parser.add_argument(
        "--wordsim_path",
        type=str,
        required=True,
        help="The path to the wordsim353 dataset file.",
    )
    
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="exp",
        help="The experiment directory.",
    )
    
    parser.add_argument(
        "--cuda",
        type=str,
        default="0",
        help="The device to use.",
    )
    
    return parser