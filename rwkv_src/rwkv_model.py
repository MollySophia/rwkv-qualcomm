########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List,Set,Dict
import os
from tqdm import tqdm
import torch.utils.cpp_extension
from rwkv_src.rwkv_v6_modules import Rwkv6SelfAttention, Rwkv6FeedForward
from rwkv_src.rwkv_v5_modules import Rwkv5SelfAttention, Rwkv5FeedForward

def sample_logits(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out, dim=-1).squeeze().cpu().numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = torch.tensor(probs).pow(1.0 / temperature).numpy()
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out

def check_rwkv_info(state_dict):
    n_layer = 0
    version = 5
    n_head = 0
    for k in state_dict.keys():
        layer_id = int(k.split('.')[1]) if ('blocks.' in k) else 0
        n_layer = max(n_layer, layer_id + 1)
        if 'ln_x' in k:
            version = max(5, version)
        if 'gate.weight' in k:
            version = max(5.1, version)
        if int(version) == 5 and 'att.time_decay' in k:
            n_head = state_dict[k].shape[0]
            if len(state_dict[k].shape) > 1:
                if state_dict[k].shape[1] > 1:
                    version = max(5.2, version)
        if 'time_maa' in k:
            version = max(6, version)
        if int(version) == 6 and 'time_faaaa' in k:
            n_head = state_dict[k].shape[0]
    return version, n_layer, n_head

class RWKV_RNN(torch.nn.Module):
    def __init__(self, args, chunks=1, chunk_idx=0):
        super().__init__()
        self.args = args
        self.eval() # set torch to inference mode

        if '.pth' in args.MODEL_NAME:
            args.MODEL_NAME = args.MODEL_NAME.replace('.pth', '')
        w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        self.args.n_embd = w['emb.weight'].shape[1]
        self.args.vocab_size = w['emb.weight'].shape[0]
        self.args.n_att = w['blocks.0.att.key.weight'].shape[0]
        self.args.n_ffn = w['blocks.0.ffn.key.weight'].shape[0]
        self.args.version, self.args.n_layer, self.args.n_head = check_rwkv_info(w)
        self.args.head_size = self.args.n_embd // self.args.n_head
        
        if chunk_idx == 0:
            print("Model version:", self.args.version)
            print("n_layer:", self.args.n_layer)
            print("n_embd:", self.args.n_embd)
            print("vocab_size:", self.args.vocab_size)
            print("n_att:", self.args.n_att)
            print("n_ffn:", self.args.n_ffn)

        layers_per_chunk = self.args.n_layer // chunks
        self.layer_begin = chunk_idx * layers_per_chunk
        self.layer_end = min(self.args.n_layer, (chunk_idx + 1) * layers_per_chunk)
        self.chunk_idx = chunk_idx
        self.chunks = chunks
        print(f"Chunk {chunk_idx}: layers {self.layer_begin} to {self.layer_end}")

        self.device = torch.device('cuda') if self.args.USE_CUDA and torch.cuda.is_available() else torch.device('cpu')
        self.gpu = True if self.device is not torch.device('cpu') else False

        for k in w.keys():
            w[k] = w[k].float()

        if self.args.USE_EMBEDDING:
            emb_weight = w['emb.weight']
            emb_weight = F.layer_norm(emb_weight, emb_weight.size()[-1:], weight=w['blocks.0.ln0.weight'].flatten(), bias=w['blocks.0.ln0.bias'].flatten())
            self.embedding = torch.nn.Embedding.from_pretrained(emb_weight)

        if self.args.version == 6:
            self.att = nn.ModuleList([Rwkv6SelfAttention(w, self.args.n_embd, self.args.head_size, i, rescale_layer=self.args.RESCALE_LAYER) for i in range(self.layer_begin, self.layer_end)])
            self.ffn = nn.ModuleList([Rwkv6FeedForward(w, self.args.n_embd, self.args.n_ffn, i, rescale_layer=self.args.RESCALE_LAYER) for i in range(self.layer_begin, self.layer_end)])
        else:
            self.att = nn.ModuleList([Rwkv5SelfAttention(w, self.args.n_embd, self.args.head_size, version=self.args.version, layer_id=i, rescale_layer=self.args.RESCALE_LAYER) for i in range(self.layer_begin, self.layer_end)])
            self.ffn = nn.ModuleList([Rwkv5FeedForward(w, self.args.n_embd, self.args.n_ffn, layer_id=i, rescale_layer=self.args.RESCALE_LAYER) for i in range(self.layer_begin, self.layer_end)])
        
        self.ln_out = nn.LayerNorm(self.args.n_embd, eps=1e-5)
        self.ln_out.weight = nn.Parameter(w['ln_out.weight'])
        self.ln_out.bias = nn.Parameter(w['ln_out.bias'])
        self.head = nn.Linear(self.args.n_embd, self.args.vocab_size)
        self.head.weight = nn.Parameter(w['head.weight'])

        if self.gpu:
            self.to(self.device)
        
        if self.args.fp16:
            self.half()
        else:
            self.float()

    def forward(self, in0, *states):
        with torch.no_grad():
            for i in range(self.layer_begin, self.layer_end):
                self.__dict__[f"state{3*i}"] = states[3*(i-self.layer_begin)]
                self.__dict__[f"state{3*i+1}"] = states[3*(i-self.layer_begin)+1]
                self.__dict__[f"state{3*i+2}"] = states[3*(i-self.layer_begin)+2]
            if self.args.USE_EMBEDDING and self.chunk_idx == 0:
                x = self.embedding(in0)
            else:
                x = in0

            for i in range(self.layer_begin, self.layer_end):
                x, self.__dict__[f"state{3*i}"], self.__dict__[f"state{3*i+1}"] = self.att[i-self.layer_begin](x, self.__dict__[f"state{3*i}"], self.__dict__[f"state{3*i+1}"])
                x, self.__dict__[f"state{3*i+2}"] = self.ffn[i-self.layer_begin](x, self.__dict__[f"state{3*i+2}"])
                if self.args.RESCALE_LAYER > 0:
                    if (i+1) % self.args.RESCALE_LAYER == 0:
                        x = x / 2

            if self.chunk_idx == self.chunks - 1:
                x = self.ln_out(x)
                x = self.head(x)
            else:
                x = x.view(1, 1, self.args.n_embd)
            return_list = [x]
            for i in range(self.layer_begin, self.layer_end):
                return_list.append(self.__dict__[f"state{3*i}"])
                return_list.append(self.__dict__[f"state{3*i+1}"])
                return_list.append(self.__dict__[f"state{3*i+2}"])
            return return_list

iteration_count = 0
def run_prompt(model, context, length=150, generate_samples=False, samples_output=None, tokenizer=None, TEMPERATURE=1.0, TOP_P=0.8):
    global iteration_count
    if iteration_count == 0 and generate_samples:
        not os.path.exists(os.path.join(samples_output, "input_list.txt")) or os.remove(os.path.join(samples_output, "input_list.txt"))
        not os.path.exists(os.path.join(samples_output, "input_list_chunk0.txt")) or os.remove(os.path.join(samples_output, "input_list_chunk0.txt"))
        not os.path.exists(os.path.join(samples_output, "input_list_chunk1.txt")) or os.remove(os.path.join(samples_output, "input_list_chunk1.txt"))
        not os.path.exists(os.path.join(samples_output, "input_list_chunk2.txt")) or os.remove(os.path.join(samples_output, "input_list_chunk2.txt"))
        not os.path.exists(os.path.join(samples_output, "input_list_chunk3.txt")) or os.remove(os.path.join(samples_output, "input_list_chunk3.txt"))
    assert tokenizer != None
    input_list_lines = []
    if length != 0:
        print(context, end="")
    if type(model) == list:
        args = model[0].args
        chunks = len(model)
    else:
        args = model.args
        chunks = -1
    fp16 = args.fp16
    device = model[0].device if chunks > 0 else model.device

    states = []
    for i in range(args.n_layer):
        if fp16:
            states.append(torch.zeros(1, args.n_embd, dtype=torch.float16))
            states.append(torch.zeros(args.n_head, args.head_size, args.head_size, dtype=torch.float16))
            states.append(torch.zeros(1, args.n_embd, dtype=torch.float16))
        else:
            states.append(torch.zeros(1, args.n_embd))
            states.append(torch.zeros(args.n_head, args.head_size, args.head_size))
            states.append(torch.zeros(1, args.n_embd))
        if device is not torch.device('cpu'):
            states = [tensor.to(device) for tensor in states]

    def encode(tokenizer, x):
        if 'Tokenizer' in str(type(tokenizer)) and not 'ABCTokenizer' in str(type(tokenizer)):
            return tokenizer.encode(x).ids
        else:
            return tokenizer.encode(x)
    
    if chunks > 0:
        if generate_samples:
            input_list_lines = [[] for i in range(chunks)]

    iterator = encode(tokenizer, context) if length != 0 else tqdm(encode(tokenizer, context))
    for token in iterator:
        if chunks > 0:
            for i in range(chunks):
                if args.USE_EMBEDDING:
                    in0 = torch.LongTensor([token]) if i == 0 else inputs[0]
                else:
                    in0 = model[0].w.emb.weight[token].view(1, 1, -1) if i == 0 else inputs[0]
                if device is not torch.device('cpu'):
                    in0 = in0.to(device)
                inputs = [in0] + [states[j] for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
                if generate_samples:
                    os.path.exists(f"{samples_output}/samples_chunk{i}") or os.mkdir(f"{samples_output}/samples_chunk{i}")
                    os.path.exists(f"{samples_output}/samples_chunk{i}/{iteration_count}") or os.mkdir(f"{samples_output}/samples_chunk{i}/{iteration_count}")
                    for j in range(len(inputs)):
                        inputs[j].cpu().numpy().astype(np.float32).tofile(f"{samples_output}/samples_chunk{i}/{iteration_count}/input_{j}.bin")
                    input_list_lines[i].append(" ".join([f"samples_chunk{i}/{iteration_count}/input_{j}.bin" for j in range(len(inputs))]) + "\n")
                inputs = model[i].forward(*inputs)
                for j in range(3*model[i].layer_begin, 3*model[i].layer_end):
                    states[j] = inputs[j - 3*model[i].layer_begin + 1]
            if generate_samples:
                iteration_count += 1
        else:
            if args.USE_EMBEDDING:
                in0 = torch.LongTensor([token])
            else:
                in0 = model.w.emb.weight[token].view(1, 1, -1)
            if device is not torch.device('cpu'):
                in0 = in0.to(device)
            inputs = [in0] + states
            if generate_samples:
                os.path.exists(f"{samples_output}/samples") or os.mkdir(f"{samples_output}/samples")
                os.path.exists(f"{samples_output}/samples/{iteration_count}") or os.mkdir(f"{samples_output}/samples/{iteration_count}")
                for j in range(len(inputs)):
                    inputs[j].cpu().numpy().astype(np.float32).tofile(f"{samples_output}/samples/{iteration_count}/input_{j}.bin")
                input_list_lines.append(" ".join([f"samples/{iteration_count}/input_{j}.bin" for j in range(len(inputs))]) + "\n")
                iteration_count += 1
            inputs = model.forward(*inputs)
            states = inputs[1:]           

    all_tokens = []
    occurrence = {}
    out_last = 0
    for i in range(length):
        if 'MIDI' in args.MODEL_NAME:
            for n in occurrence:
                inputs[0][n] -= (0 + occurrence[n] * 0.5)
            inputs[0][0] += (i - 2000) / 500
            inputs[0][127] -= 1
        token = sample_logits(inputs[0], TEMPERATURE, TOP_P)
        if 'MIDI' in args.MODEL_NAME:
            for n in occurrence: occurrence[n] *= 0.997 #### decay repetition penalty
            if token >= 128 or token == 127:
                occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
            else:
                occurrence[token] = 0.3 + (occurrence[token] if token in occurrence else 0)
        all_tokens += [token]
        try:
            tmp = tokenizer.decode(all_tokens[out_last:])
            if 'MIDI' in args.MODEL_NAME:
                tmp = ' ' + tmp
            print(tmp, end="", flush=True)
            out_last = i + 1
        except:
            pass

        if chunks > 0:
            for i in range(chunks):
                if args.USE_EMBEDDING:
                    in0 = torch.LongTensor([token]) if i == 0 else inputs[0]
                else:
                    in0 = model[0].w.emb.weight[token].view(1, 1, -1) if i == 0 else inputs[0]
                if device is not torch.device('cpu'):
                    in0 = in0.to(device)
                inputs = [in0] + [states[j] for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
                if generate_samples:
                    os.path.exists(f"{samples_output}/samples_chunk{i}/{iteration_count}") or os.mkdir(f"{samples_output}/samples_chunk{i}/{iteration_count}")
                    for j in range(len(inputs)):
                        inputs[j].cpu().numpy().astype(np.float32).tofile(f"{samples_output}/samples_chunk{i}/{iteration_count}/input_{j}.bin")
                    input_list_lines[i].append(" ".join([f"samples_chunk{i}/{iteration_count}/input_{j}.bin" for j in range(len(inputs))]) + "\n")
                inputs = model[i].forward(*inputs)
                for j in range(3*model[i].layer_begin, 3*model[i].layer_end):
                    states[j] = inputs[j - 3*model[i].layer_begin + 1]
            if generate_samples:
                iteration_count += 1
        else:
            if args.USE_EMBEDDING:
                in0 = torch.LongTensor([token])
            else:
                in0 = model.w.emb.weight[token].view(1, 1, -1)
            if device is not torch.device('cpu'):
                in0 = in0.to(device)
            inputs = [in0] + states
            if generate_samples:
                os.path.exists(f"{samples_output}/samples/{iteration_count}") or os.mkdir(f"{samples_output}/samples/{iteration_count}")
                for j in range(len(inputs)):
                    inputs[j].cpu().numpy().astype(np.float32).tofile(f"{samples_output}/samples/{iteration_count}/input_{j}.bin")
                input_list_lines.append(" ".join([f"samples/{iteration_count}/input_{j}.bin" for j in range(len(inputs))]) + "\n")
                iteration_count += 1
            inputs = model.forward(*inputs)
            states = inputs[1:]

    print('\n')
    if generate_samples:
        if chunks > 0:
            for i in range(chunks):
                with open(f"{samples_output}/input_list_chunk{i}.txt", "w") as f:
                    f.writelines(input_list_lines[i])
        else:
            with open(f"{samples_output}/input_list.txt", "w") as f:
                f.writelines(input_list_lines)

def make_chunks(chunks, args):
    return [RWKV_RNN(args, chunks=chunks, chunk_idx=i) for i in range(chunks)]
