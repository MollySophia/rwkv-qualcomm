from rwkv_src.rwkv_model import RWKV_RNN
from rwkv_src.rwkv_tokenizer import RWKV_TOKENIZER
from rwkv_src.rwkv_v7_modules_conv import Wkv7State, Wkv7Output, L2Norm
import types
import torch
import math
import os
from tqdm import tqdm
from typing import Any, Optional
import numpy as np
import random
import copy
from utils.model_utils import get_dummy_input_for_rwkv_causal_llm, get_dummy_state_kvcache, register_customop_symbols

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Compute param encodings for linear modules')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser.add_argument('encodings', type=Path, help='Path to load encodings from')
parser.add_argument('--use_cpu', action='store_true', default=False, help='Use cpu to compute')
args_parser = parser.parse_args()

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False if args_parser.use_cpu else torch.cuda.is_available()
model_args.fp16 = True if model_args.USE_CUDA else False
model_args.USE_EMBEDDING = True
model_args.RESCALE_LAYER = 0
model_args.wkv_customop = True
model_args.use_single_head_wkv = False
model_args.output_last = False
model_args.EXTERNAL_HEAD = False

model_args.MODEL_NAME = str(args_parser.model)

model = RWKV_RNN(model_args)
model_args = model.args

device = torch.device("cuda" if model_args.USE_CUDA else "cpu")

from aimet_common import quantsim
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.v2.quantsim.config_utils import set_blockwise_quantization_for_weights, set_grouped_blockwise_quantization_for_weights, set_activation_quantizers_to_float
# from aimet_torch.quantization.affine import GroupedBlockQuantizeDequantize, QuantizeDequantize
# from aimet_torch.v2.mixed_precision import MixedPrecisionConfigurator
from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.v2.nn.true_quant import QuantizedConv2d
from aimet_torch.v2 import quantization as Q

from aimet_torch.seq_mse import apply_seq_mse, SeqMseParams
from utils.dataset_builder import DatasetBuilder

@QuantizationMixin.implements(Wkv7State)
class QuantizedWkv7State(QuantizationMixin, Wkv7State):
    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers
        self.input_quantizers = torch.nn.ModuleList([None, None, None, None, None, None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, w, k, v, a, b, state2):

        with self._patch_quantized_parameters():
            ret = super().forward(w, k, v, a, b, state2)

        return ret

@QuantizationMixin.implements(Wkv7Output)
class QuantizedWkv7Output(QuantizationMixin, Wkv7Output):
    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers
        self.input_quantizers = torch.nn.ModuleList([None, None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, r, state2):

        with self._patch_quantized_parameters():
            ret = super().forward(r, state2)

        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret

@QuantizationMixin.implements(L2Norm)
class QuantizedL2Norm(QuantizationMixin, L2Norm):
    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, x):
        # Quantize input tensors
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        # Run forward with quantized inputs and parameters
        with self._patch_quantized_parameters():
            ret = super().forward(x)

        # Quantize output tensors
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret

register_customop_symbols()

dummy_input = get_dummy_input_for_rwkv_causal_llm(1, 1, device, model.args)
dummy_input = (dummy_input['in0'], dummy_input['state'])

model = model.eval()
sim = QuantizationSimModel(model, dummy_input=dummy_input,
                        quant_scheme=QuantScheme.post_training_tf_enhanced,
                        # quant_scheme=QuantScheme.post_training_percentile,
                        default_param_bw=8,
                        default_output_bw=16,
                        config_file="quantizers/configs/htp_quantsim_config_v75.json",
                        #    in_place=True,
)
# sim.set_percentile_value(99.999)
torch.cuda.empty_cache()


# mp_configurator = MixedPrecisionConfigurator(sim)
def set_linear_weight_quantizer_to_4bit(module):
    if module.param_quantizers['weight'] is not None:
        module.param_quantizers['weight'].bitwidth = 4
        module.param_quantizers['weight'].symmetric = True

for block in sim.model.blocks:
    block.att.pre_permute_r.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.pre_permute_w.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.pre_permute_k.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.pre_permute_v.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.post_permute_a.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.post_permute_g.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.post_permute_r.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.post_permute_w.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.post_permute_k.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.post_permute_v.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.post_permute_a.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.post_permute_g.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.post_permute_v1.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.ffn.pre_conv_transpose.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.ffn.post_conv_transpose.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.ffn.pre_conv_transpose2.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.ffn.post_conv_transpose2.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)

    block.att.wkv7.reshape_r.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.wkv7.reshape_w.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.wkv7.reshape_k.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.wkv7.reshape_v.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.wkv7.reshape_a.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.wkv7.reshape_b.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.wkv7.reshape_x.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)

    block.att.wkv7.wkv_state.output_quantizers[0] = None
    for i in range(6):
        block.att.wkv7.wkv_state.input_quantizers[i] = None

    block.att.wkv7.wkv_output.input_quantizers[0] = None
    block.att.wkv7.wkv_output.input_quantizers[1] = None
    block.att.wkv7.wkv_output.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)

    # if args_parser.use_w4_seq_mse or args_parser.blockwise_quant:
    #     set_linear_weight_quantizer_to_4bit(block.ffn.key)
    #     set_linear_weight_quantizer_to_4bit(block.ffn.value)
    #     set_linear_weight_quantizer_to_4bit(block.att.output)
    #     set_linear_weight_quantizer_to_4bit(block.att.key)
    #     set_linear_weight_quantizer_to_4bit(block.att.value)
    #     set_linear_weight_quantizer_to_4bit(block.att.receptance)

    # somehow it doesn't want to quantize ffn.key Linear by default
    block.ffn.key.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)

sim.model.head_pre_permute.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
sim.model.head_post_permute.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
sim.model.head_pre_reshape.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
sim.model.head_post_reshape.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
# mp_configurator.apply()

tokenizer = RWKV_TOKENIZER("./assets/rwkv_vocab_v20230424.txt")

if args_parser.encodings:
    sim.load_encodings(args_parser.encodings, allow_overwrite=False)

model = model.to('cpu').float()
torch.cuda.empty_cache()

sim.model.float()
model.args.fp16 = False
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

        src_text = text.rsplit(' ', 1)[0]
        target_text = " " + text.rsplit(' ', 1)[1]
        targets = tokenizer.encode(target_text)
        state = get_dummy_state_kvcache(1, model.args, model.device)
        input_data = torch.LongTensor([[0] + tokenizer.encode(src_text)]).to(model.device)
        logits_list = []
        logits, state = sim.model(input_data, state)
        # logits, state = model(input_data, state)
        logits = logits[:, -1, :]
        for token in tokenizer.encode(target_text):
            logits = logits.reshape(1, -1, logits.shape[-1])
            logits_list.append(logits)
            logits, state = sim.model(torch.LongTensor([[token]]).to(model.device), state)
            # logits, state = model(torch.LongTensor([[token]]).to(model.device), state)

        logits = torch.cat(logits_list, dim=1)
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
with open("lambada_results.txt", "w") as f:
    f.write(f"{math.exp(-xsum/xcnt)}\n")
    f.write(f"{xacc/xcnt}\n")
