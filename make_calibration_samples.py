from rwkv_src.rwkv_tokenizer import RWKV_TOKENIZER
from rwkv_src.rwkv_model import RWKV_RNN, make_chunks, run_prompt
import types
import os, sys
import torch
import argparse
from pathlib import Path

from torchvision import datasets
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description='Make calibration sample files')
    parser.add_argument('model', type=Path, help='Path to RWKV pth file')
    parser.add_argument('output', type=Path, help='Path to output folder')
    parser.add_argument('chunks', type=int, help='Number of chunks')
    parser.add_argument('--ext_embedding', action='store_true', default=False, help='Use external embedding')
    args = parser.parse_args()

    model_args = types.SimpleNamespace()
    model_args.USE_CUDA = torch.cuda.is_available()
    model_args.fp16 = False
    model_args.USE_EMBEDDING = False if args.ext_embedding else True
    model_args.RESCALE_LAYER = 0
    model_args.wkv_customop = False

    model_args.MODEL_NAME = str(args.model)

    tokenizer = RWKV_TOKENIZER("./assets/rwkv_vocab_v20230424.txt")

    model = make_chunks(args.chunks, model_args) if args.chunks > 1 else RWKV_RNN(model_args)

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    print("dataset len:", len(dataset['text']))
    run_prompt(model, ''.join(dataset['text'][:20]), tokenizer=tokenizer, length=0, generate_samples=True, samples_output=str(args.output))

if __name__ == '__main__':
    main()