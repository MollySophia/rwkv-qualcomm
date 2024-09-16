from rwkv_src.rwkv_tokenizer import RWKV_TOKENIZER, ABCTokenizer
from rwkv_src.rwkv_model import RWKV_RNN, sample_logits, make_chunks, run_prompt
import types
import os, sys
import torch
import numpy as np

model_args = types.SimpleNamespace()
model_args.USE_CUDA = True
model_args.USE_XPU = False
model_args.fp16 = True
model_args.USE_EMBEDDING = True
model_args.RESCALE_LAYER = 0
model_args.wkv_customop = False
model_dir = '/home/molly/workspace/models/'
model_args.MODEL_NAME = model_dir + 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096'

tokenizer = RWKV_TOKENIZER("./rwkv_vocab_v20230424.txt")

model = make_chunks(2, model_args)

from torchvision import datasets
from datasets import load_dataset

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
print("dataset len:", len(dataset['text']))
run_prompt(model, ''.join(dataset['text'][:20]), tokenizer=tokenizer, length=0, generate_samples=True)
