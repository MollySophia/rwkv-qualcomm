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

class RWKV_TOKENIZER():
    table: List[List[List[bytes]]]
    good: List[Set[int]]
    wlen: List[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> List[int]:
        src_len: int = len(src)
        tokens: List[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()

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
    # out = np.argmax(probs)
    return out

class RWKV_RNN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eval() # set torch to inference mode
        
        w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        for k in w.keys():
            w[k] = w[k].float() # convert to f32 type
            if      '.time_' in k: w[k] = w[k].squeeze()
            if '.time_decay' in k: w[k] = torch.exp(-torch.exp(w[k])).unsqueeze(-1)
            if '.time_faaaa' in k: w[k] = w[k].unsqueeze(-1)
            if 'ln' in k: w[k] = w[k].reshape(1, -1)
            if '.time_mix' in k: w[k] = w[k].reshape(1, -1)

        self.n_head = w['blocks.0.att.time_decay'].shape[0]
        self.head_size = w['blocks.0.ln1.weight'].shape[1] // self.n_head
        self.embd_sqrt = int(np.sqrt(self.args.n_embd))
        
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
        for i in range(self.args.vocab_size):
            self.w.emb.weight[i] = self.layer_norm(self.w.emb.weight[i].reshape(1, -1), self.w.blocks[0].ln0).flatten()

        self.embedding = torch.nn.Embedding(self.args.vocab_size, self.args.n_embd, _weight=self.w.emb.weight, norm_type=None)

        if self.args.RESCALE_LAYER > 0:
            for i in range(self.args.n_layer):
                self.w.blocks[i].att.output.weight = self.w.blocks[i].att.output.weight / (2 ** int(i // self.args.RESCALE_LAYER))
                self.w.blocks[i].ffn.value.weight = self.w.blocks[i].ffn.value.weight / (2 ** int(i // self.args.RESCALE_LAYER))

        self.norm_scales = [1.0 for i in range(self.args.n_layer)]

    def layer_norm(self, x, w):
        eps = 1e-5
        mean = x.mean(-1, keepdim=True)
        tmp = (x - mean)
        var = torch.mean(tmp ** 2, -1, keepdim=True)
        x = tmp / (var + eps).sqrt()
        return x * w.weight + w.bias
        # return F.layer_norm(x.flatten(), (self.args.n_embd,), w.weight.flatten(), w.bias.flatten(), 1e-5).view(1, -1)

    def group_norm(self, x, weight, bias, eps: float, i=-1, calibrate=False):
        if calibrate == False:
            x = x * np.sqrt(self.norm_scales[i])
            eps = eps * self.norm_scales[i]
        x = x.view(1, self.n_head, -1)
        mean = x.mean(-1, keepdim=True)
        tmp = (x - mean)
        if calibrate:
            if 65504.0 / torch.max(torch.abs(tmp**2)).item() < self.norm_scales[i]:
                self.norm_scales[i] = 65504.0 / torch.max(torch.abs(tmp**2)).item()
        
        # if calibrate == False:
        #     var = torch.mean(torch.clamp(tmp**2, min=-65504.0, max=65504.0), -1, keepdim=True)
        # else:
        var = torch.mean(tmp ** 2, -1, keepdim=True)
        x = tmp / (var + eps).sqrt()
        x = x.view(1, -1)
        return x * weight + bias

    def f_wkv(self, k, v, r, state2, time_first, time_decay):
        H = self.n_head
        S = self.head_size
        scale = 1 / 128
        state2 = state2 * scale
        v = v * scale
        # [H, S, 1] x [H, 1, S]
        kv = k @ v
        # [H, 1, S] x [H, S, S]
        wkv = r @ (time_first * kv + state2)
        new_state2 = time_decay * state2 + kv
        return wkv, new_state2 / scale

    def channel_mixing(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xk = (x * time_mix_k + state * (1 - time_mix_k))
        xr = (x * time_mix_r + state * (1 - time_mix_r))
        ffn_size = kw.shape[0]
        a, b = 1, ffn_size
        if int(ffn_size / 8) * 8 == ffn_size:
            a, b = 8, int(ffn_size / 8)

        r = F.conv2d(xr.view(1, 1, self.embd_sqrt, self.embd_sqrt), rw.view(self.args.n_embd, 1, self.embd_sqrt, self.embd_sqrt)).view(1, -1)
        k = F.conv2d(xk.view(1, 1, self.embd_sqrt, self.embd_sqrt), kw.view(ffn_size, 1, self.embd_sqrt, self.embd_sqrt)).view(1, 1, a, b)

        r = torch.sigmoid(r)
        # square relu, primer paper
        k = torch.relu(k)
        k = torch.square(k)
        v = F.conv2d(k, vw.view(self.args.n_embd, 1, a, b)).view(1, -1)
        return r * v

    def time_mixing(self, x, state1, state2, i:int, time_mix_k, time_mix_v, time_mix_r, time_mix_g, time_first, time_decay, kw, vw, rw, gw, ow, ln_w, ln_b, calibrate=False):
        H = self.n_head
        S = self.head_size

        xk = x * time_mix_k + state1 * (1 - time_mix_k)
        xv = x * time_mix_v + state1 * (1 - time_mix_v)
        xr = x * time_mix_r + state1 * (1 - time_mix_r)
        xg = x * time_mix_g + state1 * (1 - time_mix_g)

        r = F.conv2d(xr.view(1, 1, self.embd_sqrt, self.embd_sqrt), rw.view(self.args.n_embd, 1, self.embd_sqrt, self.embd_sqrt)).view(H, 1, S)
        k = F.conv2d(xk.view(1, 1, self.embd_sqrt, self.embd_sqrt), kw.view(self.args.n_embd, 1, self.embd_sqrt, self.embd_sqrt)).view(H, S, 1)
        v = F.conv2d(xv.view(1, 1, self.embd_sqrt, self.embd_sqrt), vw.view(self.args.n_embd, 1, self.embd_sqrt, self.embd_sqrt)).view(H, 1, S)
        g = F.conv2d(xg.view(1, 1, self.embd_sqrt, self.embd_sqrt), gw.view(self.args.n_embd, 1, self.embd_sqrt, self.embd_sqrt)).view(1, self.args.n_embd)

        g = g * torch.sigmoid(g)

        x, state2 = self.f_wkv(k, v, r, state2, time_first, time_decay)
        x = x.reshape(1, -1)

        x = self.group_norm(x, weight=ln_w, bias=ln_b, eps = 64e-5, i=i, calibrate=calibrate) * g
        # x = F.group_norm(x.unsqueeze(0), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).squeeze() * g # same as gn(x/8, eps=1e-5)
        return F.conv2d(x.view(1, 1, self.embd_sqrt, self.embd_sqrt), ow.view(self.args.n_embd, 1, self.embd_sqrt, self.embd_sqrt)).view(1, -1), state2
        
    def forward(self, token, *states, calibrate=False):
        with torch.no_grad():
            for i in range(self.args.n_layer):
                self.__dict__[f"state{3*i}"] = states[3*i]
                self.__dict__[f"state{3*i+1}"] = states[3*i+1]
                self.__dict__[f"state{3*i+2}"] = states[3*i+2]

            x = self.embedding(token)
            # x = self.w.emb.weight[token]
            # x = self.layer_norm(x, self.w.blocks[0].ln0)
            for i in range(self.args.n_layer):
                att = self.w.blocks[i].att
                x_ln = self.layer_norm(x, self.w.blocks[i].ln1)
                out, self.__dict__[f"state{3*i+1}"] = self.time_mixing(x_ln, self.__dict__[f"state{3*i}"], self.__dict__[f"state{3*i+1}"], i, 
                    att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_mix_g, att.time_faaaa, att.time_decay, 
                    att.key.weight, att.value.weight, att.receptance.weight, att.gate.weight, att.output.weight,
                    att.ln_x.weight, att.ln_x.bias, calibrate=calibrate)
                self.__dict__[f"state{3*i}"] = x_ln
                x = x + out
                ffn = self.w.blocks[i].ffn
                x_ln = self.layer_norm(x, self.w.blocks[i].ln2)
                x = x + self.channel_mixing(x_ln, self.__dict__[f"state{3*i+2}"], i, 
                    ffn.time_mix_k, ffn.time_mix_r, 
                    ffn.key.weight, ffn.value.weight, ffn.receptance.weight)
                self.__dict__[f"state{3*i+2}"] = x_ln
                if self.args.RESCALE_LAYER > 0:
                    if (i+1) % self.args.RESCALE_LAYER == 0:
                        x = x / 2
            
            x = self.layer_norm(x, self.w.ln_out)
            x = F.conv2d(x.view(1, self.args.n_embd, 1, 1), self.w.head.weight.view(self.args.vocab_size, self.args.n_embd, 1, 1)).flatten()
            return_list = [x]
            for i in range(self.args.n_layer):
                return_list.append(self.__dict__[f"state{3*i}"])
                return_list.append(self.__dict__[f"state{3*i+1}"])
                return_list.append(self.__dict__[f"state{3*i+2}"])
            return return_list

tokenizer = RWKV_TOKENIZER("/home/molly/workspace/rwkv_vocab_v20230424.txt")

args = types.SimpleNamespace()
args.MODEL_NAME = '/home/molly/workspace/RWKV-5-World-0.4B-v2-20231113-ctx4096'
args.n_layer = 24
args.n_embd = 1024
args.vocab_size = 65536
# args.MODEL_NAME = '/home/molly/workspace/RWKV-5-World-1B5-v2-20231025-ctx4096'
# args.n_layer = 24
# args.n_embd = 2048
# args.vocab_size = 65536

args.RESCALE_LAYER = 2
TEMPERATURE = 1.0
TOP_P = 0.7

model = RWKV_RNN(args)

def run_context(context, length=150, calibrate=False, generate_samples=False):
    iteration_count = 0
    input_list_lines = []
    print(context, end="")
    for i in range(model.args.n_layer):
        globals()[f"state{i*3}"] = torch.zeros(1, model.args.n_embd)
        globals()[f"state{i*3+1}"] = torch.zeros(model.n_head, model.head_size, model.head_size)
        globals()[f"state{i*3+2}"] = torch.zeros(1, model.args.n_embd)

    inputs = [torch.LongTensor([0])]
    for i in range(model.args.n_layer):
        inputs.append(globals()[f"state{i*3}"])
        inputs.append(globals()[f"state{i*3+1}"])
        inputs.append(globals()[f"state{i*3+2}"])

    for token in tokenizer.encode(context):
        inputs[0] = torch.LongTensor([token])
        if generate_samples:
            os.exist(f"data") or os.mkdir(f"data")
            os.mkdir(f"data/iteration{iteration_count}")
            index = 0
            for tensor in inputs:
                tensor.numpy().astype(np.float32).tofile(f"data/iteration{iteration_count}/input{index}.bin")
                input_list_lines.append(f"data/iteration{iteration_count}/input{index}.bin ")
                index += 1
            iteration_count += 1
            input_list_lines.append("\n")
        inputs = model.forward(*inputs, calibrate=calibrate)

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

        inputs[0] = torch.LongTensor([token])
        if generate_samples:
            os.mkdir(f"data/iteration{iteration_count}")
            index = 0
            for tensor in inputs:
                tensor.numpy().astype(np.float32).tofile(f"data/iteration{iteration_count}/input{index}.bin")
                input_list_lines.append(f"data/iteration{iteration_count}/input{index}.bin ")
                index += 1
            iteration_count += 1
            input_list_lines.append("\n")
        inputs = model.forward(*inputs, calibrate=calibrate)

    print('\n')
    if generate_samples:
        with open("input_list_samples.txt", "w") as f:
            f.writelines(input_list_lines)

for i in range(model.args.n_layer):
    globals()[f"state{i*3}"] = torch.zeros(model.args.n_embd)
    globals()[f"state{i*3+1}"] = torch.zeros(model.n_head, model.head_size, model.head_size)
    globals()[f"state{i*3+2}"] = torch.zeros(model.args.n_embd)


print("Calibrating norm scales...")
run_context("\n我们发现", calibrate=True)
run_context("\nElon Musk has", calibrate=True)
os.system("rm -rf data/iteration*")
# run_context("\n我们发现", length=300, generate_samples=True)

print("norm_scales = [", end="")
for i in range(model.args.n_layer):
    print(f"{model.norm_scales[i]:.7f}", end=", ")
print("]")

inputs = [torch.LongTensor([0])]
for i in range(model.args.n_layer):
    inputs.append(globals()[f"state{i*3}"])
    inputs.append(globals()[f"state{i*3+1}"])
    inputs.append(globals()[f"state{i*3+2}"])
inputs_tuple = tuple(inputs)
torch.onnx.export(model, inputs_tuple, "onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx", opset_version=16)
print(f"onnx model saved to onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx")

qnn_sdk_root = os.environ["QNN_SDK_ROOT"]
if not qnn_sdk_root:
    print("Please set QNN_SDK_ROOT environment variable to the root of the Qualcomm Neural Processing SDK")
    exit(1)

print("Converting to QNN model...")
os.system(f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-onnx-converter -i onnx/{args.MODEL_NAME.split('/')[-1]}.onnx --float_bw 16")
print("Compiling QNN model library...")
os.system(f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-model-lib-generator -c onnx/{args.MODEL_NAME.split('/')[-1]}.cpp -b onnx/{args.MODEL_NAME.split('/')[-1]}.bin")

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