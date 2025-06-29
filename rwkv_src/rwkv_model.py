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
# from rwkv_src.rwkv_v5_modules import Rwkv5SelfAttention, Rwkv5FeedForward
from rwkv_src.rwkv_v7_modules_conv import Rwkv7SelfAttention, Rwkv7FeedForward
from aimet_torch.v2.nn.modules.custom import Permute, Concat, Reshape

def sample_logits(out, temperature=1.0, top_p=0.8, top_k=128):
    probs = F.softmax(out, dim=-1).squeeze().cpu().numpy()
    if top_k == 0:
        return np.argmax(probs)
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    cutoff = sorted_probs[top_k]
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
        if 'r_k' in k:
            version = max(7, version)
            n_head, _ = state_dict[k].shape
        if int(version) == 6 and 'time_faaaa' in k:
            n_head = state_dict[k].shape[0]
    return version, n_layer, n_head

class RWKV_Block(nn.Module):
    def __init__(self, state_dict, n_embd, head_size, n_ffn, layer_id, layer_begin, rescale_layer=0, version=6.0, custom_wkv=False, layer_total=0, output_last=False):
        super().__init__()
        self.version = version
        self.layer_id = layer_id
        self.layer_offset = layer_id - layer_begin
        self.layer_total = layer_total
        if self.version == 7:
            self.att = Rwkv7SelfAttention(state_dict, n_embd, head_size, layer_id=layer_id, custom_wkv=custom_wkv)
            self.ffn = Rwkv7FeedForward(state_dict, n_embd, n_ffn, layer_id=layer_id, layer_total=layer_total, output_last=output_last)
        elif self.version == 6:
            self.att = Rwkv6SelfAttention(state_dict, n_embd, head_size, layer_id=layer_id, rescale_layer=rescale_layer, custom_wkv=custom_wkv)
            self.ffn = Rwkv6FeedForward(state_dict, n_embd, n_ffn, layer_id=layer_id, rescale_layer=rescale_layer)
        # else:
        #     self.att = Rwkv5SelfAttention(state_dict, n_embd, head_size, version=version, layer_id=layer_id, rescale_layer=rescale_layer)
        #     self.ffn = Rwkv5FeedForward(state_dict, n_embd, n_ffn, layer_id=layer_id, rescale_layer=rescale_layer)
        else:
            assert False, "Not implemented with new workflow yet"

    def forward(self, x, state, v_first=None):
        if self.version == 7:
            # if self.layer_id == 0:
            x, state[3*self.layer_offset], state[3*self.layer_offset+1], v_first = self.att(x, state[3*self.layer_offset], state[3*self.layer_offset+1], v_first)
            x, state[3*self.layer_offset+2] = self.ffn(x, state[3*self.layer_offset+2])
            return x, state, v_first
            # else:
            #     x, state[3*self.layer_offset], state[3*self.layer_offset+1] = self.att(x, state[3*self.layer_offset], state[3*self.layer_offset+1], v_first)
            #     x, state[3*self.layer_offset+2] = self.ffn(x, state[3*self.layer_offset+2])
            #     return x, state
            
        else:
            x, state[3*self.layer_offset], state[3*self.layer_offset+1] = self.att(x, state[3*self.layer_offset], state[3*self.layer_offset+1])
            x, state[3*self.layer_offset+2] = self.ffn(x, state[3*self.layer_offset+2])
            return x, state

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

        if self.args.version == 7:
            self.args.RESCALE_LAYER = 0

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

        emb_weight = w['emb.weight']
        if self.args.USE_EMBEDDING:
            emb_weight = F.layer_norm(emb_weight, emb_weight.size()[-1:], weight=w['blocks.0.ln0.weight'].flatten(), bias=w['blocks.0.ln0.bias'].flatten())
            self.embedding = torch.nn.Embedding.from_pretrained(emb_weight)
        else:
            self.pre_ln = nn.LayerNorm(self.args.n_embd, eps=1e-5)
            self.pre_ln.weight = nn.Parameter(w['blocks.0.ln0.weight'])
            self.pre_ln.bias = nn.Parameter(w['blocks.0.ln0.bias'])

            if self.args.fp16:
                self.emb_weight = emb_weight.half()
            else:
                self.emb_weight = emb_weight

        self.blocks = nn.ModuleList([RWKV_Block(w, self.args.n_embd, self.args.head_size, self.args.n_ffn, \
            layer_id=i,layer_begin=self.layer_begin, rescale_layer=self.args.RESCALE_LAYER, version=self.args.version, \
            custom_wkv=self.args.wkv_customop, layer_total=self.args.n_layer, \
            output_last=self.args.output_last) for i in range(self.layer_begin, self.layer_end)])
        self.ln_out = nn.LayerNorm(self.args.n_embd, eps=1e-5)
        self.ln_out.weight = nn.Parameter(w['ln_out.weight'])
        self.ln_out.bias = nn.Parameter(w['ln_out.bias'])
        # self.head = nn.Linear(self.args.n_embd, self.args.vocab_size, bias=False)
        # self.head.weight = nn.Parameter(w['head.weight'])
        if 'head.bias' in w.keys():
            self.head = nn.Conv2d(self.args.n_embd, self.args.vocab_size, 1, bias=False)
            self.head.weight = nn.Parameter(w['head.weight'].view(self.args.vocab_size, self.args.n_embd, 1, 1))
            self.head.bias = nn.Parameter(w['head.bias'].reshape(-1))
        else:
            self.head = nn.Conv2d(self.args.n_embd, self.args.vocab_size, 1, bias=False)
            self.head.weight = nn.Parameter(w['head.weight'].view(self.args.vocab_size, self.args.n_embd, 1, 1))

        self.head_pre_reshape = Reshape()
        self.head_post_reshape = Reshape()
        self.head_pre_permute = Permute()
        self.head_post_permute = Permute()

        if self.args.fp16:
            self.half()
        else:
            self.float()

        if self.gpu:
            self.to(self.device)

    def forward(self, in0, state: List[torch.Tensor], v_first: torch.Tensor|None=None):
        with torch.no_grad():
            if self.args.USE_EMBEDDING and self.chunk_idx == 0:
                x = self.embedding(in0)
            else:
                x = in0
                x = self.pre_ln(x)

            try:
                batch_size, seq_length, _ = x.size()
            except:
                batch_size, seq_length = 1, 1

            for i in range(self.layer_begin, self.layer_end):
                if self.args.version == 7:
                    # if i == 0:
                    x, state, v_first = self.blocks[i-self.layer_begin](x, state, v_first)
                    # else:
                    #     x, state = self.blocks[i-self.layer_begin](x, state, v_first)
                else:
                    x, state = self.blocks[i-self.layer_begin](x, state)
                if self.args.RESCALE_LAYER > 0:
                    if (i+1) % self.args.RESCALE_LAYER == 0:
                        x = x / 2

            if self.chunk_idx == self.chunks - 1:
                x = self.ln_out(x)
                x = self.head_pre_reshape(x, [batch_size, -1, 1, self.args.n_embd])
                x = self.head_pre_permute(x, [0, 3, 2, 1])
                x = self.head(x)
                x = self.head_post_permute(x, [0, 3, 2, 1])
                x = self.head_post_reshape(x, [batch_size, -1, self.args.vocab_size])
            else:
                x = x.view(batch_size, seq_length, self.args.n_embd)
            if self.args.version == 7 and self.chunk_idx == 0 and self.layer_end < self.args.n_layer:
                return x, state, v_first
            else:
                return x, state

def make_chunks(chunks, args):
    return [RWKV_RNN(args, chunks=chunks, chunk_idx=i) for i in range(chunks)]
