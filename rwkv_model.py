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
                    w[k] = w[k].reshape(-1, 1, 1)
                    if self.args.version in [5.2, 6.0]:
                        w[k] = w[k].reshape(self.args.n_head, -1, 1)
            if 'ln_x' in k: w[k] = w[k].reshape(1, -1)
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
            for i in range(self.args.vocab_size):
                self.w.emb.weight[i] = self.layer_norm(self.w.emb.weight[i].reshape(1, -1), self.w.blocks[0].ln0).flatten()

        self.embedding = torch.nn.Embedding(self.args.vocab_size, self.args.n_embd, _weight=self.w.emb.weight, norm_type=None)

        if self.args.RESCALE_LAYER > 0:
            for i in range(self.layer_begin, self.layer_end):
                self.w.blocks[i].att.output.weight = self.w.blocks[i].att.output.weight / (2 ** int(i // self.args.RESCALE_LAYER))
                self.w.blocks[i].ffn.value.weight = self.w.blocks[i].ffn.value.weight / (2 ** int(i // self.args.RESCALE_LAYER))

        self.norm_scales = [1.0 for i in range(self.args.n_layer)]

    def layer_norm(self, x, w):
        return F.layer_norm(x.flatten(), (self.args.n_embd,), weight=w.weight, bias=w.bias, eps=1e-5).view(1, -1)

    def group_norm(self, x, weight, bias, eps: float, i=-1, calibrate=False):
        # if calibrate:
        x = x.view(1, self.args.n_head, -1)
        mean = x.mean(-1, keepdim=True)
        tmp = (x - mean)
        if calibrate:
            if 65504.0 / torch.max(torch.abs(tmp**2)).item() < self.norm_scales[i]:
                self.norm_scales[i] = 65504.0 / torch.max(torch.abs(tmp**2)).item()
        else:
            tmp = tmp * np.sqrt(self.norm_scales[i])
            eps = eps * self.norm_scales[i]
        
        var = torch.mean(tmp ** 2, -1, keepdim=True)
        x = tmp / (var + eps).sqrt()
        x = x.view(1, -1)
        return x * weight + bias

    def wkv(self, k, v, r, state2, time_first, time_decay, scale=1):
        state2 = state2 * scale
        v = v * scale
        kv = k @ v
        wkv = r @ (time_first * kv + state2)
        new_state2 = time_decay * state2 + kv
        return wkv, new_state2 / scale

    def channel_mixing_v5(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xk = (x * time_mix_k + state * (1 - time_mix_k))
        xr = (x * time_mix_r + state * (1 - time_mix_r))

        ffn_size = kw.shape[0]
        a, b = self.args.n_head, int(ffn_size / self.args.n_head)
        r = F.conv2d(xr.view(1, 1, self.args.n_head, self.args.head_size), rw.view(self.args.n_embd, 1, self.args.n_head, self.args.head_size)).view(1, -1)
        k = F.conv2d(xk.view(1, 1, self.args.n_head, self.args.head_size), kw.view(ffn_size, 1, self.args.n_head, self.args.head_size)).view(1, 1, a, b)

        r = torch.sigmoid(r)
        # square relu, primer paper
        k = torch.square(torch.relu(k))
        v = F.conv2d(k, vw.view(self.args.n_embd, 1, a, b)).view(1, -1)
        return r * v

    def channel_mixing_v6(self, x, state, i:int, time_maa_k, time_maa_r, kw, vw, rw):
        sx = state - x
        xk = x + sx * time_maa_k
        xr = x + sx * time_maa_r

        r = rw @ xr.view(-1, 1)
        k = kw @ xk.view(-1, 1)

        r = torch.sigmoid(r)
        # square relu, primer paper
        k = torch.square(torch.relu(k))
        v = vw @ k
        return (r * v).view(1, -1)

    def time_mixing_v5(self, x, state1, state2, i:int, time_mix_k, time_mix_v, time_mix_r, time_mix_g, time_first, time_decay, kw, vw, rw, gw, ow, ln_w, ln_b, calibrate=False):
        H = self.args.n_head
        S = self.args.head_size

        xk = x * time_mix_k + state1 * (1 - time_mix_k)
        xv = x * time_mix_v + state1 * (1 - time_mix_v)
        xr = x * time_mix_r + state1 * (1 - time_mix_r)
        xg = x * time_mix_g + state1 * (1 - time_mix_g)

        r = F.conv2d(xr.view(1, 1, self.args.n_head, self.args.head_size), rw.view(self.args.n_embd, 1, self.args.n_head, self.args.head_size)).view(H, 1, S)
        k = F.conv2d(xk.view(1, 1, self.args.n_head, self.args.head_size), kw.view(self.args.n_embd, 1, self.args.n_head, self.args.head_size)).view(H, S, 1)
        v = F.conv2d(xv.view(1, 1, self.args.n_head, self.args.head_size), vw.view(self.args.n_embd, 1, self.args.n_head, self.args.head_size)).view(H, 1, S)
        g = F.conv2d(xg.view(1, 1, self.args.n_head, self.args.head_size), gw.view(self.args.n_embd, 1, self.args.n_head, self.args.head_size)).view(1, self.args.n_embd)

        g = g * torch.sigmoid(g)

        x, state2 = self.wkv(k, v, r, state2, time_first, time_decay, scale=1/8)
        x = x.reshape(1, -1)

        x = self.group_norm(x, weight=ln_w, bias=ln_b, eps = 1e-5, i=i, calibrate=calibrate) * g
        return F.conv2d(x.view(1, 1, self.args.n_head, self.args.head_size), ow.view(self.args.n_embd, 1, self.args.n_head, self.args.head_size)).view(1, -1), state2

    def time_mixing_v6(self, x, state1, state2, i:int, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, time_first, time_decay, kw, vw, rw, gw, ow, ln_w, ln_b, calibrate=False):
        H = self.args.n_head
        S = self.args.head_size

        sx = state1 - x
        xxx = x + sx * x_maa
        xxx = torch.tanh(xxx @ tm_w1).view(5, 1, -1)
        xxx = torch.bmm(xxx, tm_w2).view(5, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + sx * (w_maa + mw)
        xk = x + sx * (k_maa + mk)
        xv = x + sx * (v_maa + mv)
        xr = x + sx * (r_maa + mr)
        xg = x + sx * (g_maa + mg)

        r = (rw @ xr.view(-1, 1)).view(H, 1, S)
        k = (kw @ xk.view(-1, 1)).view(H, S, 1)
        v = (vw @ xv.view(-1, 1)).view(H, 1, S)
        g = (gw @ xg.view(-1, 1)).view(1, -1)
        g = g * F.sigmoid(g)

        w = time_decay + (torch.tanh(xw @ td_w1) @ td_w2).float().view(H, S, 1)
        w = torch.exp(-torch.exp(w.float()))

        x, state2 = self.wkv(k, v, r, state2, time_first, w, scale=1/8)

        x = self.group_norm(x, weight=ln_w, bias=ln_b, eps = 1e-5, i=i, calibrate=calibrate) * g
        return (ow @ x.view(-1, 1)).view(1, -1), state2
        
    def forward(self, in0, *states, calibrate=False):
        with torch.no_grad():
            for i in range(self.layer_begin, self.layer_end):
                self.__dict__[f"state{3*i}"] = states[3*(i-self.layer_begin)]
                self.__dict__[f"state{3*i+1}"] = states[3*(i-self.layer_begin)+1]
                self.__dict__[f"state{3*i+2}"] = states[3*(i-self.layer_begin)+2]
            if self.chunk_idx == 0:
                x = self.embedding(in0)
            else:
                x = in0
            if int(self.args.version) == 5:
                for i in range(self.layer_begin, self.layer_end):
                    att = self.w.blocks[i].att
                    x_ln = self.layer_norm(x, self.w.blocks[i].ln1)
                    out, self.__dict__[f"state{3*i+1}"] = self.time_mixing_v5(x_ln, self.__dict__[f"state{3*i}"], self.__dict__[f"state{3*i+1}"], i, 
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

def run_prompt(model, context, length=150, calibrate=False, generate_samples=False, tokenizer=None):
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
        states.append(torch.zeros(1, args.n_embd))
        states.append(torch.zeros(args.n_head, args.head_size, args.head_size))
        states.append(torch.zeros(1, args.n_embd))

    for token in tokenizer.encode(context):
        if chunks > 0:
            for i in range(chunks):
                in0 = torch.LongTensor([token]) if i == 0 else inputs[0]
                inputs = [in0] + [states[j] for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
                inputs = model[i].forward(*inputs, calibrate=calibrate)
                for j in range(3*model[i].layer_begin, 3*model[i].layer_end):
                    states[j] = inputs[j - 3*model[i].layer_begin + 1]
        else:
            in0 = torch.LongTensor([token])
            inputs = [in0] + states
            inputs = model.forward(*inputs, calibrate=calibrate)
            states = inputs[1:]

    all_tokens = []
    out_last = 0
    for i in range(length):
        token = sample_logits(inputs[0], TEMPERATURE, TOP_P)
        all_tokens += [token]
        try:
            tmp = tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
                print(tmp, end="", flush=True)
                out_last = i + 1
        except:
            pass

        if chunks > 0:
            for i in range(chunks):
                in0 = torch.LongTensor([token]) if i == 0 else inputs[0]
                inputs = [in0] + [states[j] for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
                inputs = model[i].forward(*inputs, calibrate=calibrate)
                for j in range(3*model[i].layer_begin, 3*model[i].layer_end):
                    states[j] = inputs[j - 3*model[i].layer_begin + 1]
        else:
            in0 = torch.LongTensor([token])
            inputs = [in0] + states
            inputs = model.forward(*inputs, calibrate=calibrate)
            states = inputs[1:]

    print('\n')

def make_chunks(chunks):
    return [RWKV_RNN(args, chunks=chunks, chunk_idx=i) for i in range(chunks)]

tokenizer = RWKV_TOKENIZER("./rwkv_vocab_v20230424.txt")
abctokenizer = ABCTokenizer()

args = types.SimpleNamespace()
# args.MODEL_NAME = '/home/molly/workspace/models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096'
# args.MODEL_NAME = '/home/molly/workspace/models/RWKV-5-ABC-82M-v1-20230901-ctx1024'
# args.MODEL_NAME = '/home/molly/workspace/models/RWKV-5-World-0.4B-v2-20231113-ctx4096'
args.MODEL_NAME = '/home/molly/workspace/models/RWKV-5-World-1B5-v2-20231025-ctx4096'

if 'ABC' in args.MODEL_NAME:
    args.RESCALE_LAYER = 0
else:
    args.RESCALE_LAYER = 2

TEMPERATURE = 1.0
TOP_P = 0.7

# model = RWKV_RNN(args)
model = make_chunks(2)

# prompt = """S:3
# B:9
# E:4
# B:9
# E:4
# E:4
# B:9
# L:1/8
# M:3/4
# K:D
#  Bc |"G" d2 cB"A" A2 FE |"Bm" F2 B4 F^G |"""
# prompt = chr(abctokenizer.bos_token_id) + prompt
# run_prompt(prompt, tokenizer = abctokenizer, calibrate=True)

# print("Calibrating norm scales...")
# run_prompt(model, "\n我们发现", tokenizer=tokenizer, length=100, calibrate=True)

qnn_sdk_root = os.environ["QNN_SDK_ROOT"]
if not qnn_sdk_root:
    print("Please set QNN_SDK_ROOT environment variable to the root of the Qualcomm Neural Processing SDK")
    exit(1)
os.path.exists("onnx") or os.mkdir("onnx")

import onnx, io
from onnxsim import simplify
from onnxmltools.utils import float16_converter

if type(model) == list:
    args = model[0].args
    states = []
    for i in range(args.n_layer):
        states.append(torch.zeros(1, args.n_embd))
        states.append(torch.zeros(args.n_head, args.head_size, args.head_size))
        states.append(torch.zeros(1, args.n_embd))

    for i in range(len(model)):
        in0 = torch.LongTensor([0]) if i == 0 else torch.zeros(args.n_embd)
        inputs = [in0] + [states[j] for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
        torch.onnx.export(model[i], tuple(inputs), "onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i}.onnx", opset_version=17)
        print(f"onnx model chunk{i} saved to onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i}.onnx")
    
    print("Converting and compiling QNN models...")
    for i in range(len(model)):
        os.system(f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-onnx-converter -i onnx/{args.MODEL_NAME.split('/')[-1]}_chunk{i}.onnx --float_bw 16 --no_simplification")
        os.system(f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-model-lib-generator -c onnx/{args.MODEL_NAME.split('/')[-1]}_chunk{i}.cpp -b onnx/{args.MODEL_NAME.split('/')[-1]}_chunk{i}.bin -t aarch64-android")
else:
    inputs = [torch.LongTensor([0])]
    for i in range(model.args.n_layer):
        inputs.append(torch.zeros(1, model.args.n_embd))
        inputs.append(torch.zeros(model.args.n_head, model.args.head_size, model.args.head_size))
        inputs.append(torch.zeros(1, model.args.n_embd))
    torch.onnx.export(model, tuple(inputs), "onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx", opset_version=17)
    print(f"onnx model saved to onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx")
    print("Converting to QNN model...")
    os.system(f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-onnx-converter -i onnx/{args.MODEL_NAME.split('/')[-1]}.onnx --float_bw 16")
    print("Compiling QNN model library...")
    os.system(f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-model-lib-generator -c onnx/{args.MODEL_NAME.split('/')[-1]}.cpp -b onnx/{args.MODEL_NAME.split('/')[-1]}.bin -t aarch64-android")

# from onnxsim import simplify
# import onnx
# import json
# onnx_model = onnx.load("onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx")
# model_simplified, check = simplify("onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx")
# assert check, "Simplified ONNX model could not be validated"

# encodings_dict = {'activation_encodings': {}, 'param_encodings': {}}
# graph = onnx_model.graph
# for i in range(len(graph.node)):
#     if "Pow" in graph.node[i].op_type or "Reduce" in graph.node[i].op_type or "Sqrt" in graph.node[i].op_type \
#             or "Clip" in graph.node[i].op_type or "Div" in graph.node[i].op_type:
#         for j in graph.node[i].input:
#             encodings_dict['activation_encodings'][j] = [{"bitwidth": 16, "dtype": "float"}]
#         for j in graph.node[i].output:
#             encodings_dict['activation_encodings'][j] = [{"bitwidth": 16, "dtype": "float"}]

# with open('onnx/quant_override.json', 'w') as encoding_json:
#     json.dump(encodings_dict, encoding_json, sort_keys=True, indent=4)

# onnx.save(model_simplified, "onnx/" + args.MODEL_NAME.split("/")[-1] + "_simplified.onnx")