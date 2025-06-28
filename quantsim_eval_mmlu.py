from rwkv_src.rwkv_model import RWKV_RNN
from rwkv_src.rwkv_tokenizer import RWKV_TOKENIZER
from rwkv_src.rwkv_v7_modules_conv import Wkv7OutputState, Wkv7Op, Wkv7OutputX, L2Norm
import types
import torch
import math
import os
from tqdm import tqdm
from typing import Any, Optional
import numpy as np
import random
import copy
import datetime, json
from utils.model_utils import get_dummy_input_for_rwkv_causal_llm, get_dummy_state_kvcache, register_customop_symbols

import argparse
from pathlib import Path
from torch.nn import functional as F
from datasets import load_dataset, load_from_disk

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
model_args.output_last = False

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

@QuantizationMixin.implements(Wkv7Op)
class QuantizedWkv7Op(QuantizationMixin, Wkv7Op):
    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers
        self.input_quantizers = torch.nn.ModuleList([None, None, None, None, None, None, None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, r, w, k, v, a, b, state2):
        with self._patch_quantized_parameters():
            ret = super().forward(r, w, k, v, a, b, state2)

        return ret

@QuantizationMixin.implements(Wkv7OutputState)
class QuantizedWkv7OutputState(QuantizationMixin, Wkv7OutputState):
    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, input):
        with self._patch_quantized_parameters():
            ret = super().forward(input)

        return ret

@QuantizationMixin.implements(Wkv7OutputX)
class QuantizedWkv7OutputX(QuantizationMixin, Wkv7OutputX):
    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, input):
        with self._patch_quantized_parameters():
            ret = super().forward(input)

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

    block.att.wkv7.wkv.output_quantizers[0] = None
    for i in range(7):
        block.att.wkv7.wkv.input_quantizers[i] = None

    block.att.wkv7.wkv_output_x.input_quantizers[0] = None
    block.att.wkv7.wkv_output_x.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)

    block.att.wkv7.wkv_output_state.input_quantizers[0] = None
    block.att.wkv7.wkv_output_state.output_quantizers[0] = None

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
    sim.load_encodings(args_parser.encodings, allow_overwrite=False, strict=False)

model = model.to('cpu').float()
torch.cuda.empty_cache()

sim.model.float()
model.args.fp16 = False
sim.model.eval()

sim.fold_param_quantizers()

TEMPLATE = """User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
A. <|A|>
B. <|B|>
C. <|C|>
D. <|D|>

Assistant: The answer is"""

# choices
CHOICES = [" A", " B", " C", " D"]

########################################################################################################
# MMLU DATASET
# mmlu = load_dataset("cais/mmlu", 'all')

# mmlu_test = mmlu['test']
# mmlu_dev = mmlu['dev']

# mmlu_test.save_to_disk('mmlu_test_dataset')
# mmlu_dev.save_to_disk('mmlu_dev_dataset')

mmlu_test = load_from_disk("assets/mmlu_test_dataset")
mmlu_dev = load_from_disk("assets/mmlu_dev_dataset")

########################################################################################################
# SET RANDOM SEED
SHUFFLE = False
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

########################################################################################################
# RUN EVALUATION
correct = 0
total = 0
pbar = tqdm(total=len(mmlu_test))

choices_token = [tokenizer.encode(x) for x in CHOICES]
assert all([len(x) == 1 for x in choices_token])
choices_token = [x[0] for x in choices_token]

score_by_subject = {}
for idx, sample in enumerate(mmlu_test):
    question = sample["question"]
    choices = sample["choices"]
    subject = sample["subject"]
    gt = sample["answer"]

    if SHUFFLE and not any(["Both" in x for x in choices]):  # exclude choices like "Both A and B"
        original_gt_text = choices[gt]
        np.random.shuffle(choices)
        gt = choices.index(original_gt_text)

    all_prefix = (
        TEMPLATE.replace("<Q>", question)
        .replace("<|A|>", choices[0])
        .replace("<|B|>", choices[1])
        .replace("<|C|>", choices[2])
        .replace("<|D|>", choices[3])
        .replace("<SUBJECT>", subject.replace("_", " "))
    )

    if idx == 0:
        print(f"Format example:")
        print("-" * 100)
        print(all_prefix)
        print("-" * 100)
        format_example = all_prefix

    state = get_dummy_state_kvcache(1, model.args, model.device)
    input_data = torch.LongTensor([[0] + tokenizer.encode(all_prefix.strip())]).to(model.device)
    logits, _ = sim.model(input_data, state)
    logits = logits[:, -1, :]

    log_prob = F.log_softmax(logits, dim=-1).squeeze()
    target_prob = log_prob[choices_token]
    if subject not in score_by_subject:
        score_by_subject[subject] = {"correct": 0, "total": 0}
    if torch.argmax(target_prob).item() == gt:
        correct += 1
        score_by_subject[subject]["correct"] += 1
    total += 1
    score_by_subject[subject]["total"] += 1
    pbar.set_description(f"Correct: {correct} - Total: {total} - Accuracy: {correct / total:.5f}")
    pbar.update(1)
pbar.close()

# Save results
for subject in score_by_subject:
    score_by_subject[subject]["accuracy"] = score_by_subject[subject]["correct"] / score_by_subject[subject]["total"]
now = datetime.datetime.now()
file_name = f'mmlu_test_results_{now.strftime("%Y%m%d%H%M%S")}.json'
with open(file_name, "w") as f:
    json.dump(
        {
            "model": model_args.MODEL_NAME,
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
            "template": TEMPLATE,
            "example": format_example,
            "shuffle": SHUFFLE,
            "seed": SEED,
            "score_by_subject": score_by_subject,
        },
        f,
        indent=4,
    )
print(f"Results saved to {file_name}")
