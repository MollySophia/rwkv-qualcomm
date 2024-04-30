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
from rwkv_tokenizer import RWKV_TOKENIZER, ABCTokenizer
import scipy

def sample_logits(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out, dim=-1).numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    out = np.argmax(probs)
    return out

import matplotlib.pyplot as plt
def draw_distribution(tensor_list, file_name):
    plt.cla()
    tensor_array = np.stack([t.flatten().numpy() for t in tensor_list])
    max_values = np.max(tensor_array, axis=0)
    min_values = np.min(tensor_array, axis=0)
    low_99 = np.percentile(tensor_array, 1, axis=0)
    high_99 = np.percentile(tensor_array, 99, axis=0)
    low_75 = np.percentile(tensor_array, 25, axis=0)
    high_75 = np.percentile(tensor_array, 75, axis=0)
    plt.fill_between(range(len(max_values)), min_values, max_values, color='blue', alpha=0.5)
    plt.fill_between(range(len(max_values)), low_99, high_99, color='orange', alpha=0.5)
    plt.fill_between(range(len(max_values)), low_75, high_75, color='red', alpha=0.5)
    plt.savefig(file_name)

plot_tensors = []
class RWKV_RNN(torch.nn.Module):
    def __init__(self, args, chunks=1, chunk_idx=0, fp16=False):
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

        w_new = {}
        for k in w.keys():
            if 'blocks' in k:
                parts = k.split('.')
                if int(parts[1]) < self.layer_begin or int(parts[1]) >= self.layer_end:
                    continue
            w_new[k] = w[k]
        del w
        w = w_new

        for k in w.keys():
            if fp16:
                w[k] = w[k].half() # convert to f16 type
            else:
                w[k] = w[k].float() # convert to f32 type
            if      '.time_' in k: w[k] = w[k].squeeze()
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
                        if fp16:
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

        # r = torch.randint(low=0, high=2, size=(self.args.head_size,), dtype=torch.float16 if fp16 else torch.float32)
        # r = r * 2 - 1
        H = torch.tensor(scipy.linalg.hadamard(self.args.head_size), dtype=torch.float16 if fp16 else torch.float32)
        # H = H @ torch.diag(r) # randomize the Hadamard matrix
        self.Qt = H.t().unsqueeze(0) / (self.args.head_size / 8)
        self.Q = H.unsqueeze(0) / 8

        if self.chunk_idx == 0:
            self.w.emb.weight = F.layer_norm(self.w.emb.weight.float(), (self.args.n_embd,), weight=self.w.blocks[0].ln0.weight.flatten().float(), bias=self.w.blocks[0].ln0.bias.flatten().float())
            if fp16:
                self.w.emb.weight = self.w.emb.weight.half()

        if self.args.RESCALE_LAYER > 0:
            for i in range(self.layer_begin, self.layer_end):
                self.w.blocks[i].att.output.weight = self.w.blocks[i].att.output.weight / (2 ** int(i // self.args.RESCALE_LAYER))
                self.w.blocks[i].ffn.value.weight = self.w.blocks[i].ffn.value.weight / (2 ** int(i // self.args.RESCALE_LAYER))

    def layer_norm(self, x, w):
        # plot_tensors.append(x.flatten())
        return F.instance_norm(x.view(1, 1, 1, -1), eps=1e-5).view(1, -1) * w.weight + w.bias

    def wkv(self, k, v, r, state2, time_first, time_decay, scale=1, i=-1, use_hadamard=False):
        state2 = state2.view(self.args.n_head, self.args.head_size, self.args.head_size) * scale
        v = v * scale
        kv = k @ v
        if use_hadamard:
            wkv = r @ ((time_first * kv + state2) @ self.Qt)
        else:
            wkv = r @ (time_first * kv + state2)
        new_state2 = (time_decay * state2 + kv).view(self.args.n_head * self.args.head_size * self.args.head_size) / scale
        plot_tensors.append(wkv.flatten())
        if use_hadamard:
            return wkv @ self.Q, new_state2
        else:
            return wkv, new_state2

    def channel_mixing_v5(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xk = (x * time_mix_k + state * (1 - time_mix_k))
        xr = (x * time_mix_r + state * (1 - time_mix_r))

        r = xr @ rw.t()
        k = xk @ kw.t()

        r = torch.sigmoid(r)
        # square relu, primer paper
        k = torch.square(torch.relu(k))
        v = k @ vw.t()
        return r * v

    def channel_mixing_v6(self, x, state, i:int, time_maa_k, time_maa_r, kw, vw, rw):
        sx = state - x
        xk = x + sx * time_maa_k
        xr = x + sx * time_maa_r

        r = xr @ rw.t()
        k = xk @ kw.t()

        r = torch.sigmoid(r)
        # square relu, primer paper
        k = torch.square(torch.relu(k))
        v = k @ vw.t()
        return r * v

    def time_mixing_v5(self, x, state1, state2, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow, ln_w, ln_b, calibrate=False):
        H = self.args.n_head
        S = self.args.head_size

        xk = x * time_mix_k + state1 * (1 - time_mix_k)
        xv = x * time_mix_v + state1 * (1 - time_mix_v)
        xr = x * time_mix_r + state1 * (1 - time_mix_r)

        r = (xr @ rw.t()).view(H, 1, S)
        k = (xk @ kw.t()).view(H, S, 1)
        v = (xv @ vw.t()).view(H, 1, S)

        x, state2 = self.wkv(k, v, r, state2, time_first, time_decay, scale=1/32, use_hadamard=True)

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

        if ("ABC" in self.args.MODEL_NAME):
            x, state2 = self.wkv(k, v, r, state2, time_first, time_decay, scale=1)
        else:
            x, state2 = self.wkv(k, v, r, state2, time_first, time_decay, scale=1/128)

        x = (F.instance_norm(x.view(1, H, 1, -1), eps=1e-5).view(1, -1) * ln_w + ln_b) * g
        return x @ ow.t(), state2

    def time_mixing_v6(self, x, state1, state2, i:int, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, time_first, time_decay, kw, vw, rw, gw, ow, ln_w, ln_b, calibrate=False):
        H = self.args.n_head
        S = self.args.head_size

        sx = state1 - x
        xxx = x + sx * x_maa
        xxx = torch.tanh(xxx @ tm_w1).view(5, 1, -1)
        xxx = torch.bmm(xxx, tm_w2).view(5, -1)
        maa = torch.cat([w_maa.unsqueeze(0), k_maa, v_maa, r_maa, g_maa], dim=0)
        # print(x.shape, sx.shape, maa.shape)
        xxx = x + sx * (maa + xxx)
        xw, xk, xv, xr, xg = torch.split(xxx, 1, dim=-2)
        # mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        # xw = x + sx * (w_maa + mw)
        # xk = x + sx * (k_maa + mk)
        # xv = x + sx * (v_maa + mv)
        # xr = x + sx * (r_maa + mr)
        # xg = x + sx * (g_maa + mg)

        r = (xr @ rw.t()).view(H, 1, S)
        k = (xk @ kw.t()).view(H, S, 1)
        v = (xv @ vw.t()).view(H, 1, S)
        g = xg @ gw.t()
        g = g * F.sigmoid(g)

        w = time_decay + (torch.tanh(xw @ td_w1) @ td_w2).view(H, S, 1)
        # e^(-e^2.8) is near the finest precision of f16
        w = torch.exp(-torch.exp(w.clip(-100, 2.8)))

        x, state2 = self.wkv(k, v, r, state2, time_first, w, scale=1/8, i=i, use_hadamard=True)

        x = (F.instance_norm(x.view(1, H, 1, -1), eps=1e-5).view(1, -1) * ln_w + ln_b) * g
        return x @ ow.t(), state2

    def forward(self, in0, *states, calibrate=False):
        with torch.no_grad():
            for i in range(self.layer_begin, self.layer_end):
                self.__dict__[f"state{3*i}"] = states[3*(i-self.layer_begin)]
                self.__dict__[f"state{3*i+1}"] = states[3*(i-self.layer_begin)+1]
                self.__dict__[f"state{3*i+2}"] = states[3*(i-self.layer_begin)+2]
            # if self.chunk_idx == 0:
            #     # x = self.embedding(in0)
            # else:
            x = in0.view(1, 1, -1)
            if self.args.version == 5:
                for i in range(self.layer_begin, self.layer_end):
                    att = self.w.blocks[i].att
                    x_ln = self.layer_norm(x, self.w.blocks[i].ln1)
                    out, self.__dict__[f"state{3*i+1}"] = self.time_mixing_v5(x_ln, self.__dict__[f"state{3*i}"], self.__dict__[f"state{3*i+1}"], i, 
                        att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first, att.time_decay, 
                        att.key.weight, att.value.weight, att.receptance.weight, att.output.weight,
                        att.ln_x.weight, att.ln_x.bias, calibrate=calibrate)
                    self.__dict__[f"state{3*i}"] = x_ln
                    x = x + out
                    ffn = self.w.blocks[i].ffn
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
                    x_ln = self.layer_norm(x, self.w.blocks[i].ln1)
                    out, self.__dict__[f"state{3*i+1}"] = self.time_mixing_v5_1(x_ln, self.__dict__[f"state{3*i}"], self.__dict__[f"state{3*i+1}"], i, 
                        att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_mix_g, att.time_first, att.time_decay, 
                        att.key.weight, att.value.weight, att.receptance.weight, att.gate.weight, att.output.weight,
                        att.ln_x.weight, att.ln_x.bias, calibrate=calibrate)
                    self.__dict__[f"state{3*i}"] = x_ln
                    x = x + out
                    ffn = self.w.blocks[i].ffn
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
                    x_ln = self.layer_norm(x, self.w.blocks[i].ln1)
                    out, self.__dict__[f"state{3*i+1}"] = self.time_mixing_v6(x_ln, self.__dict__[f"state{3*i}"], self.__dict__[f"state{3*i+1}"], i, 
                        att.time_maa_x, att.time_maa_w, att.time_maa_k, att.time_maa_v, att.time_maa_r, att.time_maa_g, att.time_maa_w1, att.time_maa_w2,
                        att.time_decay_w1, att.time_decay_w2, att.time_first, att.time_decay,
                        att.key.weight, att.value.weight, att.receptance.weight, att.gate.weight, att.output.weight,
                        att.ln_x.weight, att.ln_x.bias, calibrate=calibrate)
                    self.__dict__[f"state{3*i}"] = x_ln
                    x = x + out
                    ffn = self.w.blocks[i].ffn
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
                x = F.conv2d(x.view(1, self.args.n_embd, 1, 1), self.w.head.weight.view(self.args.vocab_size, self.args.n_embd, 1, 1))
            return_list = [x.flatten()]
            for i in range(self.layer_begin, self.layer_end):
                return_list.append(self.__dict__[f"state{3*i}"])
                return_list.append(self.__dict__[f"state{3*i+1}"])
                return_list.append(self.__dict__[f"state{3*i+2}"])
            return return_list

def run_prompt(model, context, length=150, calibrate=False, generate_samples=False, tokenizer=None, fp16=False):
    assert tokenizer != None
    iteration_count = 0
    input_list_lines = []
    print(context, end="")
    if type(model) == list:
        args = model[0].args
        chunks = len(model)
    else:
        args = model.args
        chunks = -1

    states = []
    for i in range(args.n_layer):
        if fp16:
            states.append(torch.zeros(1, args.n_embd, dtype=torch.float16))
            states.append(torch.zeros(args.n_head * args.head_size * args.head_size, dtype=torch.float16))
            states.append(torch.zeros(1, args.n_embd, dtype=torch.float16))
        else:
            states.append(torch.zeros(1, args.n_embd))
            states.append(torch.zeros(args.n_head * args.head_size * args.head_size))
            states.append(torch.zeros(1, args.n_embd))

    def encode(tokenizer, x):
        if 'Tokenizer' in str(type(tokenizer)):
            return tokenizer.encode(x).ids
        else:
            return tokenizer.encode(x)

    for token in encode(tokenizer, context):
        if chunks > 0:
            if generate_samples:
                input_list_lines = [[] for i in range(chunks)]
            for i in range(chunks):
                in0 = model[0].w.emb.weight[token] if i == 0 else inputs[0]
                inputs = [in0] + [states[j] for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
                if generate_samples:
                    os.path.exists(f"samples_chunk{i}") or os.mkdir(f"samples_chunk{i}")
                    os.path.exists(f"samples_chunk{i}/{iteration_count}") or os.mkdir(f"samples_chunk{i}/{iteration_count}")
                    for j in range(len(inputs)):
                        inputs[j].numpy().astype(np.float32).tofile(f"samples_chunk{i}/{iteration_count}/input_{j}.bin")
                    input_list_lines[i].append(" ".join([f"samples_chunk{i}/{iteration_count}/input_{j}.bin" for j in range(len(inputs))]) + "\n")
                inputs = model[i].forward(*inputs, calibrate=calibrate)
                for j in range(3*model[i].layer_begin, 3*model[i].layer_end):
                    states[j] = inputs[j - 3*model[i].layer_begin + 1]
            if generate_samples:
                os.path.exists("sample_outputs") or os.mkdir("sample_outputs")
                inputs[0].numpy().astype(np.float32).tofile(f"sample_outputs/{iteration_count}.bin")
                iteration_count += 1
        else:
            in0 = model.w.emb.weight[token]
            inputs = [in0] + states
            if generate_samples:
                os.path.exists(f"samples/{iteration_count}") or os.mkdir(f"samples/{iteration_count}")
                for j in range(len(inputs)):
                    inputs[j].numpy().astype(np.float32).tofile(f"samples/{iteration_count}/input_{j}.bin")
                input_list_lines.append(" ".join([f"samples/{iteration_count}/input_{j}.bin" for j in range(len(inputs))]) + "\n")
            inputs = model.forward(*inputs, calibrate=calibrate)
            states = inputs[1:]
            if generate_samples:
                os.path.exists("sample_outputs") or os.mkdir("sample_outputs")
                inputs[0].numpy().astype(np.float32).tofile(f"sample_outputs/{iteration_count}.bin")
                iteration_count += 1

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
            if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
                print(tmp, end="", flush=True)
                out_last = i + 1
        except:
            pass

        if chunks > 0:
            for i in range(chunks):
                in0 = model[0].w.emb.weight[token] if i == 0 else inputs[0]
                inputs = [in0] + [states[j] for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
                if generate_samples:
                    os.path.exists(f"samples_chunk{i}/{iteration_count}") or os.mkdir(f"samples_chunk{i}/{iteration_count}")
                    for j in range(len(inputs)):
                        inputs[j].numpy().astype(np.float32).tofile(f"samples_chunk{i}/{iteration_count}/input_{j}.bin")
                    input_list_lines[i].append(" ".join([f"samples_chunk{i}/{iteration_count}/input_{j}.bin" for j in range(len(inputs))]) + "\n")
                inputs = model[i].forward(*inputs, calibrate=calibrate)
                for j in range(3*model[i].layer_begin, 3*model[i].layer_end):
                    states[j] = inputs[j - 3*model[i].layer_begin + 1]
            if generate_samples:
                os.path.exists("sample_outputs") or os.mkdir("sample_outputs")
                inputs[0].numpy().astype(np.float32).tofile(f"sample_outputs/{iteration_count}.bin")
                iteration_count += 1
        else:
            in0 = model.w.emb.weight[token]
            inputs = [in0] + states
            if generate_samples:
                os.path.exists(f"samples/{iteration_count}") or os.mkdir(f"samples/{iteration_count}")
                for j in range(len(inputs)):
                    inputs[j].numpy().astype(np.float32).tofile(f"samples/{iteration_count}/input_{j}.bin")
                input_list_lines.append(" ".join([f"samples/{iteration_count}/input_{j}.bin" for j in range(len(inputs))]) + "\n")
            inputs = model.forward(*inputs, calibrate=calibrate)
            states = inputs[1:]
            if generate_samples:
                os.path.exists("sample_outputs") or os.mkdir("sample_outputs")
                inputs[0].numpy().astype(np.float32).tofile(f"sample_outputs/{iteration_count}.bin")
                iteration_count += 1

    print('\n')
    if generate_samples:
        if chunks > 0:
            for i in range(chunks):
                with open(f"input_list_chunk{i}.txt", "w") as f:
                    f.writelines(input_list_lines[i])
        else:
            with open(f"input_list.txt", "w") as f:
                f.writelines(input_list_lines)

def make_chunks(chunks, fp16=False):
    return [RWKV_RNN(args, chunks=chunks, chunk_idx=i, fp16=fp16) for i in range(chunks)]

args = types.SimpleNamespace()
model_dir = '/home/molly/workspace/models/'
# args.MODEL_NAME = model_dir + 'RWKV-6-ABC-85M-v1-20240217-ctx1024'
args.MODEL_NAME = model_dir + 'RWKV-6-MIDI-120M-20240220-ctx4096'
# args.MODEL_NAME = model_dir + 'RWKV-x060-World-3B-v2.1-20240417-ctx4096'
# args.MODEL_NAME = model_dir + 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096'
# args.MODEL_NAME = model_dir + 'RWKV-5-ABC-82M-v1-20230901-ctx1024'
# args.MODEL_NAME = model_dir + 'RWKV-5-MIDI-120M-v1-20230728-ctx4096'
# args.MODEL_NAME = model_dir + 'RWKV-5-World-0.4B-v2-20231113-ctx4096'
# args.MODEL_NAME = model_dir + 'RWKV-5-World-1B5-v2-20231025-ctx4096'
# args.MODEL_NAME = model_dir + 'RWKV-5-World-3B-v2-20231118-ctx16k'

if 'ABC' in args.MODEL_NAME:
    args.RESCALE_LAYER = 0
    tokenizer = ABCTokenizer()
    prompt = """S:3
B:9
E:4
B:9
E:4
E:4
B:9
L:1/8
M:3/4
K:D
 Bc | d2 cB A2 FE | F2 B4 F^G |
"""
    prompt = chr(tokenizer.bos_token_id) + prompt
elif 'MIDI' in args.MODEL_NAME:
    args.RESCALE_LAYER = 0
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file("./tokenizer-midi.json")
    prompt = "<pad> " + "v:5b:3 v:5b:2 t125 t125 t125 t106 pi:43:5 t24 pi:4a:7 t15 pi:4f:7 t17 pi:56:7 t18 pi:54:7 t125 t49 pi:51:7 t117 pi:4d:7 t125 t125 t111 pi:37:7 t14 pi:3e:6 t15 pi:43:6 t12 pi:4a:7 t17 pi:48:7 t125 t60 pi:45:7 t121 pi:41:7 t125 t117 s:46:5 s:52:5 f:46:5 f:52:5 t121 s:45:5 s:46:0 s:51:5 s:52:0 f:45:5 f:46:0 f:51:5 f:52:0 t121 s:41:5 s:45:0 s:4d:5 s:51:0 f:41:5 f:45:0 f:4d:5 f:51:0 t102 pi:37:0 pi:3e:0 pi:41:0 pi:43:0 pi:45:0 pi:48:0 pi:4a:0 pi:4d:0 pi:4f:0 pi:51:0 pi:54:0 pi:56:0 t19 s:3e:5 s:41:0 s:4a:5 s:4d:0 f:3e:5 f:41:0 f:4a:5 f:4d:0 t121 v:3a:5 t121 v:39:7 t15 v:3a:0 t106 v:35:8 t10 v:39:0 t111 v:30:8 v:35:0 t125 t117 v:32:8 t10 v:30:0 t125 t125 t103 v:5b:0 v:5b:0 t9 pi:4a:7"
else:
    args.RESCALE_LAYER = 2
    tokenizer = RWKV_TOKENIZER("./rwkv_vocab_v20230424.txt")
    prompt = "\n我们发现"

TEMPERATURE = 1.0
TOP_P = 0.7

model = RWKV_RNN(args)
# model = make_chunks(4)
model_fp16 = RWKV_RNN(args, fp16=True)
# model_fp16 = make_chunks(4, fp16=True)

# run_prompt(model, prompt, tokenizer=tokenizer, length=1000, generate_samples=True)
# run_prompt(model, prompt, tokenizer=tokenizer, length=100)
run_prompt(model_fp16, prompt, tokenizer=tokenizer, length=100, fp16=True)
# draw_distribution(plot_tensors, "wkv.png")
# quit()

qnn_sdk_root = os.environ["QNN_SDK_ROOT"]
if not qnn_sdk_root:
    print("Please set QNN_SDK_ROOT environment variable to the root of the Qualcomm Neural Processing SDK")
    exit(1)
os.path.exists("onnx") or os.mkdir("onnx")

if type(model) == list:
    model[0].w.emb.weight.numpy().astype(np.float32).tofile("onnx/" + args.MODEL_NAME.split("/")[-1] + ".emb")
    args = model[0].args
    states = []
    for i in range(args.n_layer):
        states.append(torch.zeros(1, args.n_embd))
        states.append(torch.zeros(args.n_head * args.head_size * args.head_size))
        states.append(torch.zeros(1, args.n_embd))

    for i in range(len(model)):
        dirname = "onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i}"
        os.path.exists(dirname) or os.mkdir(dirname)
        in0 = model[0].w.emb.weight[0] if i == 0 else torch.zeros(args.n_embd)
        inputs = [in0] + [states[j] for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
        input_names = ['in'] + [f'state{j}_in' for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
        output_names = ['out'] + [f'state{j}_out' for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
        torch.onnx.export(model[i], tuple(inputs), dirname + "/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i}.onnx", input_names=input_names, output_names=output_names, opset_version=15)
        print(f"onnx model chunk{i} saved to {dirname}" + "/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i}.onnx")

    config = f"""version: {args.version}
head_size: {args.head_size}
n_layer: {args.n_layer}
n_embd: {args.n_embd}
n_att: {args.n_att}
n_ffn: {args.n_ffn}
vocab_size: {args.vocab_size}
"""
    with open("onnx/" + args.MODEL_NAME.split("/")[-1] + ".config", "w") as f:
        f.write(config)
    
    print("Converting and compiling QNN models...")
    for i in range(len(model)):
        dirname = "onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i}"
        os.path.exists(dirname) or os.mkdir(dirname)
        os.system(f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-onnx-converter -i {dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i}.onnx --float_bw 32 --no_simplification --converter_op_package_lib ./customop/CustomOpPackage_Converter_Op_Package/ConverterOpPackage/libConverterOpPackage.so --op_package_config ./customop/CustomOpPackage.xml")
        os.system(f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-model-lib-generator -c {dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i}.cpp -b {dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i}.bin -t x86_64-linux-clang")
else:
    model.w.emb.weight.numpy().astype(np.float32).tofile("onnx/" + args.MODEL_NAME.split("/")[-1] + ".emb")
    args = model.args
    inputs = [model.w.emb.weight[0]]
    for i in range(model.args.n_layer):
        inputs.append(torch.zeros(1, model.args.n_embd))
        inputs.append(torch.zeros(model.args.n_head, model.args.head_size, model.args.head_size))
        inputs.append(torch.zeros(1, model.args.n_embd))
    input_names = ['id'] + [f'state{i}_in' for i in range(3*model.args.n_layer)]
    output_names = ['logits'] + [f'state{i}_out' for i in range(3*model.args.n_layer)]
    torch.onnx.export(model, tuple(inputs), "onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx", input_names=input_names, output_names=output_names, opset_version=15)
    print(f"onnx model saved to onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx")
    config = f"""version: {args.version}
head_size: {args.head_size}
n_layer: {args.n_layer}
n_embd: {args.n_embd}
n_att: {args.n_att}
n_ffn: {args.n_ffn}
vocab_size: {args.vocab_size}
"""
    with open("onnx/" + args.MODEL_NAME.split("/")[-1] + ".config", "w") as f:
        f.write(config)

    print("Converting to QNN model...")
    os.system(f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-onnx-converter -i onnx/{args.MODEL_NAME.split('/')[-1]}.onnx --float_bw 32 --no_simplification --converter_op_package_lib ./customop/CustomOpPackage_Converter_Op_Package/ConverterOpPackage/libConverterOpPackage.so --op_package_config ./customop/CustomOpPackage.xml")
    print("Compiling QNN model library...")
    os.system(f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-model-lib-generator -c onnx/{args.MODEL_NAME.split('/')[-1]}.cpp -b onnx/{args.MODEL_NAME.split('/')[-1]}.bin -t x86_64-linux-clang")

# from onnxsim import simplify
# import onnx
# import json
# onnx_model = onnx.load("onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx")
# model_simplified, check = simplify("onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx")
# assert check, "Simplified ONNX model could not be validated"

# encodings_dict = {'activation_encodings': {}, 'param_encodings': {}}
# graph = onnx_model.graph
# for i in range(len(graph.node)):
    # if "Pow" in graph.node[i].op_type or "Sqrt" in graph.node[i].op_type or "Div" in graph.node[i].op_type \
    #         or "Mul" in graph.node[i].op_type: 
    #     for j in graph.node[i].input:
    #         encodings_dict['activation_encodings'][j] = [{"bitwidth": 16, "dtype": "float"}]
    #     for j in graph.node[i].output:
    #         encodings_dict['activation_encodings'][j] = [{"bitwidth": 16, "dtype": "float"}]

# with open('onnx/quant_override.json', 'w') as encoding_json:
#     json.dump(encodings_dict, encoding_json, sort_keys=True, indent=4)

# onnx.save(model_simplified, "onnx/" + args.MODEL_NAME.split("/")[-1] + "_simplified.onnx")
