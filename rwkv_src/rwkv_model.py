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
import scipy
from tqdm import tqdm
import torch.utils.cpp_extension
wkv_c_impl_src = """
#include <torch/script.h>
#include <tuple>

static std::tuple<torch::Tensor, torch::Tensor> custom_wkv(
    torch::Tensor k, torch::Tensor v, torch::Tensor r,
    torch::Tensor state2, torch::Tensor time_first,
    torch::Tensor time_decay) {
    auto kv = torch::matmul(k, v);
    auto wkv = torch::matmul(r, (time_first * kv + state2));
    auto new_state2 = time_decay * state2 + kv;
    return std::make_tuple(wkv, new_state2);
}

TORCH_LIBRARY(rwkv, m) {
  m.def("custom_wkv", &custom_wkv);
}
"""

def sample_logits(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out, dim=-1).cpu().numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = torch.tensor(probs).pow(1.0 / temperature).numpy()
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out

class RWKV_RNN(torch.nn.Module):
    def __init__(self, args, chunks=1, chunk_idx=0):
        super().__init__()
        self.args = args
        self.eval() # set torch to inference mode
        
        w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        self.args.n_embd = w['emb.weight'].shape[1]
        self.args.vocab_size = w['emb.weight'].shape[0]
        self.args.n_att = w['blocks.0.att.key.weight'].shape[0]
        self.args.n_ffn = w['blocks.0.ffn.key.weight'].shape[0]
        self.args.n_layer = 0
        self.args.version = 5
        REAL_TIME_FIRST = False

        if self.args.wkv_customop:
            try:
                module = torch.utils.cpp_extension.load_inline(
                    name='custom_wkv', cpp_sources=[wkv_c_impl_src])
            except:
                pass
            self.wkv_func = torch.ops.rwkv.custom_wkv
        else:
            self.wkv_func = self.wkv

        for k in w.keys():
            layer_id = int(k.split('.')[1]) if ('blocks.' in k) else 0
            self.args.n_layer = max(self.args.n_layer, layer_id + 1)
            if 'ln_x' in k:
                self.args.version = max(5, self.args.version)
            if 'gate.weight' in k:
                self.args.version = max(5.1, self.args.version)
            if int(self.args.version) == 5 and 'att.time_decay' in k:
                self.args.n_head = w[k].shape[0]
                if len(w[k].shape) > 1:
                    if w[k].shape[1] > 1:
                        self.args.version = max(5.2, self.args.version)
            if 'time_maa' in k:
                self.args.version = max(6, self.args.version)
            if int(self.args.version) == 6 and 'time_faaaa' in k:
                self.args.n_head = w[k].shape[0]
        
        if chunk_idx == 0:
            print("Model version:", self.args.version)
            print("n_layer:", self.args.n_layer)
            print("n_embd:", self.args.n_embd)
            print("vocab_size:", self.args.vocab_size)
            print("n_att:", self.args.n_att)
            print("n_ffn:", self.args.n_ffn)
        self.args.head_size = w['blocks.0.ln1.weight'].shape[0] // self.args.n_head

        REAL_TIME_FIRST = False
        for x in list(w.keys()):
            if '.time_faaaa' in x: REAL_TIME_FIRST = True
        if REAL_TIME_FIRST:
            w = {k.replace('.time_faaaa','.time_first') if '.time_faaaa' in k else k: v for k, v in w.items()}

        layers_per_chunk = self.args.n_layer // chunks
        self.layer_begin = chunk_idx * layers_per_chunk
        self.layer_end = min(self.args.n_layer, (chunk_idx + 1) * layers_per_chunk)
        self.chunk_idx = chunk_idx
        self.chunks = chunks
        print(f"Chunk {chunk_idx}: layers {self.layer_begin} to {self.layer_end}")

        self.gpu = False
        self.INFERENCE_DEVICE = torch.device('cpu')
        if self.args.USE_CUDA:
            if torch.cuda.is_available():
                self.INFERENCE_DEVICE = torch.device('cuda')
            else:
                self.args.USE_CUDA = False
        elif self.args.USE_XPU:
            try:
                import intel_extension_for_pytorch as ipex
                self.INFERENCE_DEVICE = torch.device('xpu')
            except:
                self.args.USE_XPU = False
        if self.INFERENCE_DEVICE is not torch.device('cpu'):
            self.gpu = True
        w_new = {}
        for k in w.keys():
            if 'blocks' in k:
                parts = k.split('.')
                if int(parts[1]) < self.layer_begin or int(parts[1]) >= self.layer_end:
                    continue
            if self.args.fp16:
                w[k] = w[k].half() # convert to f16 type
            else:
                w[k] = w[k].float() # convert to f32 type
            if 'emb' in k or ('blocks.0.ln0' in k):
                if chunk_idx > 0:
                    continue
                if not self.args.USE_EMBEDDING:
                    w_new[k] = w[k]
                    continue
            if 'ln_out' in k or 'head' in k:
                if chunk_idx < chunks - 1:
                    continue

            w_new[k] = w[k].to(self.INFERENCE_DEVICE) if self.gpu else w[k]
        del w
        w = w_new

        for k in w.keys():
            if '.time_' in k: w[k] = w[k].squeeze()
            if '.time_decay' in k and '_w' not in k:
                if int(self.args.version) == 5:
                    w[k] = torch.exp(-torch.exp(w[k])).reshape(-1, 1, 1)
                    if self.args.version == 5.2:
                        w[k] = w[k].reshape(self.args.n_head, -1, 1)
                elif int(self.args.version) == 6:
                    w[k] = w[k].reshape(self.args.n_head, -1, 1)
            if '.time_first' in k: 
                if int(self.args.version) in [5, 6]:
                    if REAL_TIME_FIRST:
                        w[k] = w[k].reshape(-1, 1, 1)
                    else:
                        w[k] = torch.exp(w[k].float()).reshape(-1, 1, 1)
                        if self.args.fp16:
                            w[k] = w[k].half()
                    if self.args.version in [5.2, 6.0]:
                        w[k] = w[k].reshape(self.args.n_head, -1, 1)
            if 'ln' in k: w[k] = w[k].reshape(1, -1)
            if '.time_mix' in k or ('maa' in k and not 'w' in k): w[k] = w[k].reshape(1, -1)
        
        self.w = types.SimpleNamespace() # set self.w from w
        self.w.blocks = {}
        for k in w.keys(): # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
            parts = k.split('.')
            last = parts.pop()
            here = self.w
            for p in parts:
                if p.isdigit():
                    p = int(p)
                    if p not in here: here[p] = types.SimpleNamespace()
                    here = here[p]
                else:
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k])

        if self.chunk_idx == 0:
            self.w.emb.weight = F.layer_norm(self.w.emb.weight.float(), (self.args.n_embd,), weight=self.w.blocks[0].ln0.weight.flatten().float(), bias=self.w.blocks[0].ln0.bias.flatten().float())
            if self.args.fp16:
                self.w.emb.weight = self.w.emb.weight.half()
            if self.gpu:
                self.w.emb.weight = self.w.emb.weight.to(self.INFERENCE_DEVICE)
            if self.args.USE_EMBEDDING:
                self.embedding = torch.nn.Embedding.from_pretrained(self.w.emb.weight, freeze=True)

        if self.args.RESCALE_LAYER > 0:
            for i in range(self.layer_begin, self.layer_end):
                self.w.blocks[i].att.output.weight = self.w.blocks[i].att.output.weight / (2 ** int(i // self.args.RESCALE_LAYER))
                self.w.blocks[i].ffn.value.weight = self.w.blocks[i].ffn.value.weight / (2 ** int(i // self.args.RESCALE_LAYER))

    def layer_norm(self, x, w):
        # return F.instance_norm(x.view(1, 1, 1, -1), eps=1e-5).view(1, -1) * w.weight + w.bias
        return F.layer_norm(x, x.size()[-1:], weight=w.weight.flatten(), bias=w.bias.flatten())

    def wkv(self, k, v, r, state2, time_first, time_decay):
        kv = k @ v
        wkv = r @ (time_first * kv + state2)
        new_state2 = time_decay * state2 + kv
        return wkv, new_state2

    def channel_mixing_v5(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xk = (x * time_mix_k + state * (1 - time_mix_k))
        xr = (x * time_mix_r + state * (1 - time_mix_r))

        r = xr @ rw.t()
        k = xk @ kw.t()

        r = torch.sigmoid(r)
        # square relu, primer paper
        k = torch.pow(torch.relu(k), 2)
        v = k @ vw.t()
        return r * v

    def channel_mixing_v6(self, x, state, i:int, time_maa_k, time_maa_r, kw, vw, rw):
        sx = state - x
        xk = x + sx * time_maa_k
        xr = x + sx * time_maa_r

        # r = xr @ rw.t()
        # k = xk @ kw.t()
        N = self.args.n_embd
        r = F.conv2d(xr.view(1, N//8, 1, 8), rw.view(N, N//8, 1, 8)).view(-1, 8, 8)
        k = F.conv2d(xk.view(1, N//8, 1, 8), kw.view(-1, N//8, 1, 8)).view(-1, 8, 8)

        r = torch.sigmoid(r)
        # square relu, primer paper
        k = torch.pow(torch.relu(k), 2)
        # v = k @ vw.t()
        v = F.conv2d(k.view(1, -1, 1, 1), vw.view(N, -1, 1, 1)).view(-1, 8, 8)
        return (r * v).view(1, 1, N)
        # return r * v

    def time_mixing_v5(self, x, state1, state2, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow, ln_w, ln_b, calibrate=False):
        H = self.args.n_head
        S = self.args.head_size

        xk = x * time_mix_k + state1 * (1 - time_mix_k)
        xv = x * time_mix_v + state1 * (1 - time_mix_v)
        xr = x * time_mix_r + state1 * (1 - time_mix_r)

        r = (xr @ rw.t()).view(H, 1, S)
        k = (xk @ kw.t()).view(H, S, 1)
        v = (xv @ vw.t()).view(H, 1, S) / 32

        x, state2 = self.wkv_func(k, v, r, state2, time_first, time_decay)

        x = (F.instance_norm(x.view(1, H, 1, -1), eps=1e-5).view(1, -1) * ln_w + ln_b)
        return x @ ow.t(), state2

    def time_mixing_v5_1(self, x, state1, state2, i:int, time_mix_k, time_mix_v, time_mix_r, time_mix_g, time_first, time_decay, kw, vw, rw, gw, ow, ln_w, ln_b, calibrate=False):
        H = self.args.n_head
        S = self.args.head_size

        xk = x * time_mix_k + state1 * (1 - time_mix_k)
        xv = x * time_mix_v + state1 * (1 - time_mix_v)
        xr = x * time_mix_r + state1 * (1 - time_mix_r)
        xg = x * time_mix_g + state1 * (1 - time_mix_g)

        r = (xr @ rw.t()).view(H, 1, S)
        k = (xk @ kw.t()).view(H, S, 1)
        v = (xv @ vw.t()).view(H, 1, S)
        g = xg @ gw.t()
        g = g * F.sigmoid(g)

        # avoid fp16 overflow
        if ("ABC" in self.args.MODEL_NAME):
            v = v / 128

        x, state2 = self.wkv_func(k, v, r, state2, time_first, time_decay)

        x = (F.instance_norm(x.view(1, H, 1, -1), eps=1e-5).view(1, -1) * ln_w + ln_b) * g
        return x @ ow.t(), state2

    def time_mixing_v6(self, x, state1, state2, i:int, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, time_first, time_decay, kw, vw, rw, gw, ow, ln_w, ln_b, calibrate=False):
        H = self.args.n_head
        S = self.args.head_size

        sx = state1 - x
        xxx = x + sx * x_maa
        xxx = torch.tanh((xxx @ tm_w1.t().view(5, -1, H*S).transpose(1, 2)))
        xxx = torch.bmm(xxx, tm_w2)
        maa = torch.cat([w_maa.view(1, 1, -1), k_maa.view(1, 1, -1), v_maa.view(1, 1, -1), r_maa.view(1, 1, -1), g_maa.view(1, 1, -1)], dim=0)
        xxx = x + sx * (maa + xxx)
        xw, xk, xv, xr, xg = torch.split(xxx, 1, dim=0)

        N = H * S
        # r = F.conv2d(xr.view(1, N//8, 1, 8), rw.view(N, N//8, 1, 8)).view(H, 1, S)
        # k = F.conv2d(xk.view(1, N//8, 1, 8), kw.view(N, N//8, 1, 8) / 2).view(H, S, 1)
        # v = F.conv2d(xv.view(1, N//8, 1, 8), vw.view(N, N//8, 1, 8) / 4).view(H, 1, S)
        g = F.conv2d(xg.view(1, N//8, 1, 8), gw.view(N, N//8, 1, 8)).view(1, 1, N)
        r = (xr @ rw.view(H, S, H*S).transpose(1, 2))
        k = (xk @ (kw.view(H, S, H*S).transpose(1, 2) / 2)).view(H, S, 1)
        v = (xv @ (vw.view(H, S, H*S).transpose(1, 2) / 4))
        # g = xg @ gw.t()

        g = g * F.sigmoid(g)

        w = time_decay + (torch.tanh(xw @ td_w1) @ td_w2).view(H, S, 1)
        w = torch.exp(-torch.exp(w.clip(-9.72, 2.27)))
        # w = torch.exp(-torch.exp(w))

        x, state2 = self.wkv_func(k, v, r, state2, time_first, w)

        # x = (F.instance_norm(x.view(1, H, 1, S), eps=1e-5).view(1, 8, N//8) * ln_w.view(1, 8, N//8) + ln_b.view(1, 8, N//8))
        x = (F.instance_norm(x.view(1, H, 1, -1), eps=1e-5).view(1, -1) * ln_w + ln_b)
        x = x * g
        # return x @ ow.t(), state2
        return F.conv2d(x.view(1, N//8, 1, 8), ow.view(N, N//8, 1, 8)).view(1, 1, N), state2

    def forward(self, in0, *states, calibrate=False):
        with torch.no_grad():
            for i in range(self.layer_begin, self.layer_end):
                self.__dict__[f"state{3*i}"] = states[3*(i-self.layer_begin)]
                self.__dict__[f"state{3*i+1}"] = states[3*(i-self.layer_begin)+1]
                self.__dict__[f"state{3*i+2}"] = states[3*(i-self.layer_begin)+2]
            if self.args.USE_EMBEDDING and self.chunk_idx == 0:
                x = self.embedding(in0)
            else:
                x = in0
            if self.args.version == 5:
                for i in range(self.layer_begin, self.layer_end):
                    att = self.w.blocks[i].att
                    # x_ln = self.att[i].layernorm(x)
                    x_ln = self.layer_norm(x, self.w.blocks[i].ln1)
                    out, self.__dict__[f"state{3*i+1}"] = self.time_mixing_v5(x_ln, self.__dict__[f"state{3*i}"], self.__dict__[f"state{3*i+1}"], i, 
                        att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first, att.time_decay, 
                        att.key.weight, att.value.weight, att.receptance.weight, att.output.weight,
                        att.ln_x.weight, att.ln_x.bias)
                    self.__dict__[f"state{3*i}"] = x_ln
                    x = x + out
                    ffn = self.w.blocks[i].ffn
                    # x_ln = self.ffn[i].layernorm(x)
                    x_ln = self.layer_norm(x, self.w.blocks[i].ln2)
                    x = x + self.channel_mixing_v5(x_ln, self.__dict__[f"state{3*i+2}"], i, 
                        ffn.time_mix_k, ffn.time_mix_r, 
                        ffn.key.weight, ffn.value.weight, ffn.receptance.weight)
                    self.__dict__[f"state{3*i+2}"] = x_ln
                    if self.args.RESCALE_LAYER > 0:
                        if (i+1) % self.args.RESCALE_LAYER == 0:
                            x = x / 2
            elif self.args.version in [5.1, 5.2]:
                for i in range(self.layer_begin, self.layer_end):
                    att = self.w.blocks[i].att
                    # x_ln = self.att[i].layernorm(x)
                    x_ln = self.layer_norm(x, self.w.blocks[i].ln1)
                    out, self.__dict__[f"state{3*i+1}"] = self.time_mixing_v5_1(x_ln, self.__dict__[f"state{3*i}"], self.__dict__[f"state{3*i+1}"], i, 
                        att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_mix_g, att.time_first, att.time_decay, 
                        att.key.weight, att.value.weight, att.receptance.weight, att.gate.weight, att.output.weight,
                        att.ln_x.weight, att.ln_x.bias)
                    self.__dict__[f"state{3*i}"] = x_ln
                    x = x + out
                    ffn = self.w.blocks[i].ffn
                    # x_ln = self.ffn[i].layernorm(x)
                    x_ln = self.layer_norm(x, self.w.blocks[i].ln2)
                    x = x + self.channel_mixing_v5(x_ln, self.__dict__[f"state{3*i+2}"], i, 
                        ffn.time_mix_k, ffn.time_mix_r, 
                        ffn.key.weight, ffn.value.weight, ffn.receptance.weight)
                    self.__dict__[f"state{3*i+2}"] = x_ln
                    if self.args.RESCALE_LAYER > 0:
                        if (i+1) % self.args.RESCALE_LAYER == 0:
                            x = x / 2
            elif int(self.args.version) == 6:
                for i in range(self.layer_begin, self.layer_end):
                    att = self.w.blocks[i].att
                    # x_ln = self.att[i].layernorm(x)
                    x_ln = self.layer_norm(x, self.w.blocks[i].ln1)
                    out, self.__dict__[f"state{3*i+1}"] = self.time_mixing_v6(x_ln, self.__dict__[f"state{3*i}"], self.__dict__[f"state{3*i+1}"], i, 
                        att.time_maa_x, att.time_maa_w, att.time_maa_k, att.time_maa_v, att.time_maa_r, att.time_maa_g, att.time_maa_w1, att.time_maa_w2,
                        att.time_decay_w1, att.time_decay_w2, att.time_first, att.time_decay,
                        att.key.weight, att.value.weight, att.receptance.weight, att.gate.weight, att.output.weight,
                        att.ln_x.weight, att.ln_x.bias)
                    self.__dict__[f"state{3*i}"] = x_ln
                    x = x + out
                    ffn = self.w.blocks[i].ffn
                    # x_ln = self.ffn[i].layernorm(x)
                    x_ln = self.layer_norm(x, self.w.blocks[i].ln2)
                    x = x + self.channel_mixing_v6(x_ln, self.__dict__[f"state{3*i+2}"], i, 
                        ffn.time_maa_k, ffn.time_maa_r, 
                        ffn.key.weight, ffn.value.weight, ffn.receptance.weight)
                    self.__dict__[f"state{3*i+2}"] = x_ln
                    if self.args.RESCALE_LAYER > 0:
                        if (i+1) % self.args.RESCALE_LAYER == 0:
                            x = x / 2
            
            if self.chunk_idx == self.chunks - 1:
                x = self.layer_norm(x, self.w.ln_out)
                if "7B" in self.args.MODEL_NAME:
                    a, b = torch.chunk(self.w.head.weight.t(), 2, dim=1)
                    x = torch.cat([x @ a, x @ b], dim=1)
                else:
                    x = x @ self.w.head.weight.t()
                x = x.view(self.args.vocab_size)
            else:
                x = x.view(self.args.n_embd)
            return_list = [x]
            for i in range(self.layer_begin, self.layer_end):
                return_list.append(self.__dict__[f"state{3*i}"])
                return_list.append(self.__dict__[f"state{3*i+1}"])
                return_list.append(self.__dict__[f"state{3*i+2}"])
            return return_list

iteration_count = 0
def run_prompt(model, context, length=150, generate_samples=False, tokenizer=None, TEMPERATURE=1.0, TOP_P=0.8):
    global iteration_count
    if iteration_count == 0 and generate_samples:
        not os.path.exists("input_list.txt") or os.remove("input_list.txt")
        not os.path.exists("input_list_chunk0.txt") or os.remove("input_list_chunk0.txt")
        not os.path.exists("input_list_chunk1.txt") or os.remove("input_list_chunk1.txt")
        not os.path.exists("input_list_chunk2.txt") or os.remove("input_list_chunk2.txt")
        not os.path.exists("input_list_chunk3.txt") or os.remove("input_list_chunk3.txt")
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
    INFERENCE_DEVICE = model[0].INFERENCE_DEVICE if chunks > 0 else model.INFERENCE_DEVICE

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
        if INFERENCE_DEVICE is not torch.device('cpu'):
            states = [tensor.to(INFERENCE_DEVICE) for tensor in states]

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
                if INFERENCE_DEVICE is not torch.device('cpu'):
                    in0 = in0.to(INFERENCE_DEVICE)
                inputs = [in0] + [states[j] for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
                if generate_samples:
                    os.path.exists(f"samples_chunk{i}") or os.mkdir(f"samples_chunk{i}")
                    os.path.exists(f"samples_chunk{i}/{iteration_count}") or os.mkdir(f"samples_chunk{i}/{iteration_count}")
                    for j in range(len(inputs)):
                        inputs[j].cpu().numpy().astype(np.float32).tofile(f"samples_chunk{i}/{iteration_count}/input_{j}.bin")
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
            if INFERENCE_DEVICE is not torch.device('cpu'):
                in0 = in0.to(INFERENCE_DEVICE)
            inputs = [in0] + states
            if generate_samples:
                os.path.exists("samples") or os.mkdir("samples")
                os.path.exists(f"samples/{iteration_count}") or os.mkdir(f"samples/{iteration_count}")
                for j in range(len(inputs)):
                    inputs[j].cpu().numpy().astype(np.float32).tofile(f"samples/{iteration_count}/input_{j}.bin")
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
                if INFERENCE_DEVICE is not torch.device('cpu'):
                    in0 = in0.to(INFERENCE_DEVICE)
                inputs = [in0] + [states[j] for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
                if generate_samples:
                    os.path.exists(f"samples_chunk{i}/{iteration_count}") or os.mkdir(f"samples_chunk{i}/{iteration_count}")
                    for j in range(len(inputs)):
                        inputs[j].cpu().numpy().astype(np.float32).tofile(f"samples_chunk{i}/{iteration_count}/input_{j}.bin")
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
            if INFERENCE_DEVICE is not torch.device('cpu'):
                in0 = in0.to(INFERENCE_DEVICE)
            inputs = [in0] + states
            if generate_samples:
                os.path.exists(f"samples/{iteration_count}") or os.mkdir(f"samples/{iteration_count}")
                for j in range(len(inputs)):
                    inputs[j].cpu().numpy().astype(np.float32).tofile(f"samples/{iteration_count}/input_{j}.bin")
                input_list_lines.append(" ".join([f"samples/{iteration_count}/input_{j}.bin" for j in range(len(inputs))]) + "\n")
                iteration_count += 1
            inputs = model.forward(*inputs)
            states = inputs[1:]

    print('\n')
    if generate_samples:
        if chunks > 0:
            for i in range(chunks):
                with open(f"input_list_chunk{i}.txt", "a") as f:
                    f.writelines(input_list_lines[i])
        else:
            with open(f"input_list.txt", "a") as f:
                f.writelines(input_list_lines)

def make_chunks(chunks, args):
    return [RWKV_RNN(args, chunks=chunks, chunk_idx=i) for i in range(chunks)]
