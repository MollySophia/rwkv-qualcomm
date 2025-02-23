from rwkv_src.rwkv_model import RWKV_RNN
from rwkv_src.rwkv_tokenizer import RWKV_TOKENIZER
from rwkv_src.rwkv_v7_modules import Wkv7
import types
import torch
import torch.nn as nn
import math
import os
from tqdm import tqdm

from utils.model_utils import get_dummy_input_for_rwkv_causal_llm, get_dummy_state_kvcache

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Compute param encodings for linear modules')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
args_parser = parser.parse_args()

model_args = types.SimpleNamespace()
model_args.USE_CUDA = torch.cuda.is_available()
model_args.fp16 = False
model_args.USE_EMBEDDING = True
model_args.RESCALE_LAYER = 0
model_args.wkv_customop = True

model_args.MODEL_NAME = str(args_parser.model)

model = RWKV_RNN(model_args)
model_args = model.args

from aimet_common.defs import QuantScheme
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.v2.mixed_precision import MixedPrecisionConfigurator
from aimet_torch.v2.nn import QuantizationMixin

@QuantizationMixin.implements(Wkv7)
class QuantizedWkv7(QuantizationMixin, Wkv7):
    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers
        self.input_quantizers = torch.nn.ModuleList([None, None, None, None, None, None, None, None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, seq_length, r, w, k, v, a, b, state2):
        # not quantizing wkv module
        return super().forward(seq_length, r, w, k, v, a, b, state2)

from torch.onnx import register_custom_op_symbolic
def onnx_custom_wkv6(g, k, v, r, state2, time_first, time_decay):
    n_head = state2.type().sizes()[0]
    head_size = state2.type().sizes()[1]
    out1, out2 = g.op("rwkv::wkv6", k, v, r, state2, time_first, time_decay, outputs=2)
    return out1.setType(k.type().with_dtype(torch.float32).with_sizes([k.type().sizes()[0], head_size])),\
        out2.setType(k.type().with_dtype(torch.float32).with_sizes([n_head, head_size, head_size]))

def onnx_custom_wkv7(g, r, w, k, v, a, b, state):
    n_head = state.type().sizes()[0]
    head_size = state.type().sizes()[1]
    out1, out2 = g.op("rwkv::wkv7", r, w, k, v, a, b, state, outputs=2)
    return out1.setType(k.type().with_dtype(torch.float32).with_sizes([k.type().sizes()[0], head_size])),\
        out2.setType(k.type().with_dtype(torch.float32).with_sizes([n_head, head_size, head_size]))
register_custom_op_symbolic("rwkv::wkv6", onnx_custom_wkv6, 9)
register_custom_op_symbolic("rwkv::wkv7", onnx_custom_wkv7, 9)

dummy_input = get_dummy_input_for_rwkv_causal_llm(1, 1, "cuda", model.args)
dummy_input = (dummy_input['in0'], dummy_input['state'])

sim = QuantizationSimModel(model, dummy_input=dummy_input,
                           quant_scheme=QuantScheme.post_training_tf,
                           default_param_bw=8,
                           default_output_bw=16,
                        #    config_file=get_path_for_per_channel_config(),
                           config_file="/home/molly/miniconda3/envs/py310/lib/python3.10/site-packages/aimet_common/quantsim_config/backend_aware_htp_quantsim_config_v75.json",
                        #    in_place=True)
)

# mp_configurator = MixedPrecisionConfigurator(sim)
# for block in sim.model.blocks:
#     mp_configurator.set_precision(block.att.ln_1, activation='fp16')
#     mp_configurator.set_precision(block.ffn.ln_2, activation='fp16')

# mp_configurator.apply()

tokenizer = RWKV_TOKENIZER("./assets/rwkv_vocab_v20230424.txt")

from datasets import load_from_disk, load_dataset, IterableDataset
from torch.utils.data import DataLoader, Dataset
from typing import Any, Optional
train_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
dataloader = DataLoader(
    train_dataset, batch_size=5
)

def pass_calibration_data(model: torch.nn.Module, forward_pass_args: Optional[Any]=None):
    data_loader = forward_pass_args

    num_batches = 6

    model.eval()
    with torch.no_grad():
        for batch, input_data in tqdm(enumerate(data_loader)):
            text = "\n\n".join(input_data['text'])
            if len(text) == 0:
                num_batches += 1
                continue
            state = get_dummy_state_kvcache(1, model_args, model.device)
            input_data = torch.LongTensor([tokenizer.encode(text)]).to(model.device)
            model(input_data, state)

            if batch >= num_batches:
                break

sim.compute_encodings(pass_calibration_data, forward_pass_callback_args=dataloader)

sim.model.eval()

lambada_texts = None
with open("assets/lambada_test.txt") as f:
    lambada_texts = ''.join(f.readlines()).split('|')

xsum = 0
xacc = 0
xcnt = 0
with torch.no_grad():
    for text in tqdm(lambada_texts[:300]):
        if len(text) == 0:
            continue
        targets = tokenizer.encode(' ' + text.split(' ')[-1])
        state = get_dummy_state_kvcache(1, model_args, model.device)
        input_data = torch.LongTensor([[0] + tokenizer.encode(text)]).to(model.device)
        logits, _ = sim.model(input_data, state)
        # logits, _ = model(input_data, state)
        logits = logits[:, -1-len(targets):-1, :]
        logits = torch.nn.functional.softmax(logits, dim=-1)
        results = torch.argmax(logits, dim=-1).squeeze().cpu().numpy().tolist()
        if type(results) == int:
            results = [results]
        if results == targets:
            xacc += 1

        for i in range(len(targets)):
            xsum += logits[0, i, targets[i]].log().item()
        xcnt += 1

print(math.exp(-xsum/xcnt))
print(xacc/xcnt)

# 12.279, 0.52 for 300 samples v7 0.1B fp32
# 7.142, 0.593 for 300 samples v7 0.4B fp32
# 4.336, 0.68 for 300 samples v7 1.5B fp32

sim.model.to("cpu")
torch.cuda.empty_cache()

input_names = ['in'] + [f'state{j}_in' for j in range(3*model.layer_begin, 3*model.layer_end)]
output_names = ['out'] + [f'state{j}_out' for j in range(3*model.layer_begin, 3*model.layer_end)]

dummy_input = get_dummy_input_for_rwkv_causal_llm(1, 1, "cpu", model.args)
dummy_input = (dummy_input['in0'], dummy_input['state'])
os.path.exists('./tmp') or os.makedirs('./tmp')
sim.export(path='./tmp', filename_prefix='quantized_test', dummy_input=dummy_input, onnx_export_args={'input_names': input_names, 'output_names': output_names})