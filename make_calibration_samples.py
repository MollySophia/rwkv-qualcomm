from rwkv_src.rwkv_tokenizer import RWKV_TOKENIZER, ABCTokenizer
from rwkv_src.rwkv_model import RWKV_RNN, sample_logits, make_chunks, run_prompt
import types
import os, sys
import torch
import numpy as np

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False
model_args.USE_XPU = True
model_args.fp16 = True
model_args.USE_EMBEDDING = True
model_args.RESCALE_LAYER = 0
model_dir = '/home/molly/workspace/models/'
# model_args.MODEL_NAME = model_dir + 'RWKV-6-ABC-85M-v1-20240217-ctx1024'
# model_args.MODEL_NAME = model_dir + 'RWKV-6-MIDI-120M-20240220-ctx4096'
# model_args.MODEL_NAME = model_dir + 'RWKV-x060-World-3B-v2.1-20240417-ctx4096'
model_args.MODEL_NAME = model_dir + 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096'
# model_args.MODEL_NAME = model_dir + 'RWKV-x060-World-7B-v2.1-20240507-ctx4096'
# model_args.MODEL_NAME = model_dir + 'RWKV-5-ABC-82M-v1-20230901-ctx1024'
# model_args.MODEL_NAME = model_dir + 'RWKV-5-MIDI-120M-v1-20230728-ctx4096'
# model_args.MODEL_NAME = model_dir + 'RWKV-5-World-0.4B-v2-20231113-ctx4096'
# model_args.MODEL_NAME = model_dir + 'RWKV-5-World-1B5-v2-20231025-ctx4096'
# model_args.MODEL_NAME = model_dir + 'RWKV-5-World-3B-v2-20231118-ctx16k'

tokenizer = RWKV_TOKENIZER("./rwkv_vocab_v20230424.txt")

model = make_chunks(2, model_args)

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import models, datasets, transforms
from datasets import load_dataset

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
run_prompt(model, ''.join(dataset['text'][:40]), tokenizer=tokenizer, length=0, generate_samples=True)
