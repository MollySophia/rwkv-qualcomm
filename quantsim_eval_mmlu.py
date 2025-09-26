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
parser.add_argument('--load_encodings', type=Path, default=None, help='Path to load encodings from')
parser.add_argument('--use_cpu', action='store_true', default=False, help='Use cpu to compute')
parser.add_argument('--blockwise_quant', action='store_true', help='Use blockwise quantization')
parser.add_argument('--calib_num_batches', type=int, default=10, help='Number of batches to calibrate')
parser.add_argument('--binidx_dataset', type=Path, default=None, help='Path to binidx dataset')
parser.add_argument('--w8_embedding', action='store_true', help='Use int8 quantization for embedding')
parser.add_argument('--heads_per_split', type=int, default=8, help='Number of heads per split')
args_parser = parser.parse_args()

# SET RANDOM SEED
SHUFFLE = False
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False if args_parser.use_cpu else torch.cuda.is_available()
model_args.fp16 = False
model_args.bf16 = True
model_args.USE_EMBEDDING = True
model_args.RESCALE_LAYER = 0
model_args.wkv_customop = True
model_args.output_last = False
model_args.EXTERNAL_HEAD = False
model_args.heads_per_split = args_parser.heads_per_split

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

in_place = True

model = model.eval()
sim = QuantizationSimModel(model, dummy_input=dummy_input,
                        quant_scheme=QuantScheme.post_training_tf_enhanced,
                        # quant_scheme=QuantScheme.post_training_percentile,
                        default_param_bw=8,
                        default_output_bw=16,
                        config_file="quantizers/configs/htp_quantsim_config_v75.json",
                        in_place=in_place,
)
# sim.set_percentile_value(99.999)
torch.cuda.empty_cache()

if args_parser.w8_embedding:
    sim.model.embedding.param_quantizers['weight'] = Q.affine.Quantize((model.args.n_embd,), bitwidth=8, symmetric=True).to(device)

# mp_configurator = MixedPrecisionConfigurator(sim)
def set_linear_weight_quantizer_to_4bit(module):
    if module.param_quantizers['weight'] is not None:
        module.param_quantizers['weight'].bitwidth = 4
        module.param_quantizers['weight'].symmetric = True

num_head_splits = len(sim.model.blocks[0].att.heads)
for block in sim.model.blocks:
    block.att.pre_permute_r.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.pre_permute_w.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.pre_permute_k.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.pre_permute_v.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)

    for head in block.att.heads:
        head.post_permute_a.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
        head.post_permute_g.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
        head.post_permute_r.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
        head.post_permute_w.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
        head.post_permute_k.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
        head.post_permute_v.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
        head.post_permute_a.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
        head.post_permute_g.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
        head.post_permute_v1.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)

        head.mul_ln_x.input_quantizers[1] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
        head.add_ln_x.input_quantizers[1] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
        head.mul_gate.input_quantizers[1] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
        head.scale_w.input_quantizers[1] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)

        head.mix_kk.input_quantizers[1] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
        head.mix_ka_add.input_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
        head.mix_ka_sub.input_quantizers[1] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
        head.mix_ka_mul_a.input_quantizers[1] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
        head.mul_r_k.input_quantizers[1] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)

        head.wkv7.wkv.output_quantizers[0] = None
        for i in range(7):
            head.wkv7.wkv.input_quantizers[i] = None

        head.wkv7.wkv_output_x.input_quantizers[0] = None
        head.wkv7.wkv_output_x.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)

        head.wkv7.wkv_output_state.input_quantizers[0] = None
        head.wkv7.wkv_output_state.output_quantizers[0] = None

    block.ffn.pre_conv_transpose.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.ffn.post_conv_transpose.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.ffn.pre_conv_transpose2.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.ffn.post_conv_transpose2.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)

    block.ffn.mul_x_k.input_quantizers[1] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)

    # somehow it doesn't quantize the affine parameters of Mul/Add layers with aimet 2.8.0
    block.att.lerp_mul_r.input_quantizers[1] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.lerp_mul_w.input_quantizers[1] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.lerp_mul_k.input_quantizers[1] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.lerp_mul_v.input_quantizers[1] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.lerp_mul_a.input_quantizers[1] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
    block.att.lerp_mul_g.input_quantizers[1] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)

sim.model.head_pre_permute.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
sim.model.head_post_permute.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
sim.model.head_pre_reshape.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
sim.model.head_post_reshape.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)

tokenizer = RWKV_TOKENIZER("./assets/rwkv_vocab_v20230424.txt")

dataloader = None
if args_parser.binidx_dataset is not None:
    from utils.indexed_dataset import MMapIndexedDataset
    dataset = MMapIndexedDataset(str(args_parser.binidx_dataset))
    block_size = 2048
    len_total = 1
    took = []
    tokens = np.array([0])
    while len_total < args_parser.calib_num_batches * block_size:
        if len(took) >= len(dataset):
            break
        idx = random.randint(0, len(dataset) - 1)
        while idx in took:
            if len(took) >= len(dataset):
                break
            idx = random.randint(0, len(dataset) - 1)
        took.append(idx)
        len_total += len(dataset[idx])
        tokens = np.concatenate((tokens, np.array([0]), np.array(dataset[idx])), axis=0)

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, tokens, block_size=2048):
            self.full_tokens = tokens
            self.block_size = block_size
            self._len = len(tokens) // block_size

        def __len__(self):
            return self._len

        def __getitem__(self, idx):
            start_idx = idx * self.block_size
            end_idx = (idx+1) * self.block_size

            input_ids = self.full_tokens[start_idx:end_idx]
            return input_ids
    dataset = CustomDataset(tokens, block_size)
    def collate_fn(x):
        return {'input_ids': torch.LongTensor(np.array(x, dtype=np.int64))}
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
else:
    dataset_args = types.SimpleNamespace()
    dataset_args.calib_dataset_name = "wikitext"
    dataset_args.calib_dataset_config_name = "wikitext-2-raw-v1"
    dataset_args.dataset_cache_dir = "./dataset_cache"
    dataset_args.calib_dataset_split = None
    dataset_args.calib_dataset_preprocessor = "gpt2"
    dataset_args.eval_dataset_name = "wikitext"
    dataset_args.eval_dataset_config_name = "wikitext-103-raw-v1"
    dataset_args.eval_dataset_split = "test"
    dataset_args.eval_dataset_preprocessor = "gptq"
    dataset_args.per_device_calib_batch_size = 1
    dataset_args.per_device_eval_batch_size = 1
    dataset_args.block_size = 1024
    dataset_args.seed = 42

    dataset_builder = DatasetBuilder(dataset_args)
    dataset_builder.make_dataset(tokenizer=tokenizer, args=dataset_args, column_name="text", shuffle=True)
    dataloader = dataset_builder.train_dataloader

def pass_calibration_data(model: torch.nn.Module, forward_pass_args: Optional[Any]=None):
    model.eval()
    with torch.no_grad():
        state = get_dummy_state_kvcache(1, model.args, model.device)
        model(forward_pass_args['input_ids'], state)

def pass_calibration_data_calib(model: torch.nn.Module, forward_pass_args: Optional[Any]=None):
    data_loader = forward_pass_args

    num_batches = args_parser.calib_num_batches

    model.eval()
    with torch.no_grad():
        for batch, input_data in tqdm(enumerate(data_loader)):
            state = get_dummy_state_kvcache(1, model.args, model.device)
            model(input_data['input_ids'].to(model.device), state)

            if batch >= num_batches:
                break

if args_parser.load_encodings:
    sim.load_encodings(args_parser.load_encodings, allow_overwrite=False)
    if args_parser.binidx_dataset is not None:
        sim.compute_encodings(pass_calibration_data_calib, forward_pass_callback_args=dataloader)
elif args_parser.blockwise_quant:
    fn = lambda module: isinstance(module, QuantizedConv2d) and module.param_quantizers['weight'].bitwidth == 4

    BLOCK_QUANT_SIZE = 16
    BITWIDTH = 4
    DECOMPRESSED_BITWIDTH = 8

    set_grouped_blockwise_quantization_for_weights(sim = sim,
                                                arg = fn,
                                                bitwidth = BITWIDTH,
                                                symmetric = True,
                                                decompressed_bw = DECOMPRESSED_BITWIDTH,
                                                block_size = BLOCK_QUANT_SIZE,
                                                block_grouping = -1)

    print(sim.model.blocks[0])
    sim.compute_encodings(pass_calibration_data_calib, forward_pass_callback_args=dataloader)

print(sim.model.blocks[0])
print(sim.model.head)

# model = model.to('cpu').float()
if not in_place:
    model = model.to('cpu').float()
model = model.float()
model.args.fp16 = False
model.args.bf16 = False
torch.cuda.empty_cache()

# sim.fold_param_quantizers()

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

state = get_dummy_state_kvcache(1, model.args, model.device)
input_data = torch.LongTensor([[0]]).to(model.device)
logits, _ = sim.model(input_data, state)
print(logits.flatten()[:10])
for i in range(len(state)):
    print("state", i, state[i].flatten()[:10])

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
