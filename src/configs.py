import argparse
from ast import parse
from email.policy import default
from tkinter.tix import Tree
from numpy import double
import torch
from transformers import BertConfig, BertTokenizer, BertForMaskedLM, \
                         RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, \
                         AlbertConfig, AlbertTokenizer, AlbertModel

_GLOBAL_ARGS = None

_MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model': BertForMaskedLM,
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model': RobertaForMaskedLM,   #
    },
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model': AlbertModel
    }
}

def get_args_parser():
    """get_args_parser"""
    parser = argparse.ArgumentParser(description="CMD for args parser.")
    
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--template_id", type=int, required=True)
    parser.add_argument("--num_train_samples_per_label", type=int, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, required=True)
    parser.add_argument("--num_label_types", type=int, required=True)
    parser.add_argument("--criterion_type", type=str, required=True)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_original_label_types", type=int, required=True)
    parser.add_argument("--dic_dataset", type=str, required=True)
    parser.add_argument("--label_words_kb", type=str, required=True)
    parser.add_argument("--round_num", type=int, required=True) # 0为第一轮，1为第二轮
    parser.add_argument("--file_train_dataset", type=str, required=True)
    parser.add_argument("--file_test_dataset", type=str, required=True)
    parser.add_argument("--do_valid", type=int, required=True) # 0为false
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--best_k_ratio", type=double, required=False)
    parser.add_argument("--warmup_steps", type=int, required=True, default=0)
    parser.add_argument("--learning_rate", type=double, required=True, default=2e-5)
    parser.add_argument("--file_valid_dataset", type=str, required=True)
    parser.add_argument("--template_file", type=str, required=True)
    parser.add_argument("--file_train_labels_dataset", type=str, required=False)
    parser.add_argument("--file_test_labels_dataset", type=str, required=False)

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args

    return args

def get_model_classes():
    return _MODEL_CLASSES

