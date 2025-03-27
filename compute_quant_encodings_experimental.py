from rwkv_src.rwkv_model import RWKV_RNN
from rwkv_src.rwkv_tokenizer import RWKV_TOKENIZER
from rwkv_src.rwkv_v7_modules_conv import Wkv7, L2Norm
import types
import torch
import math
import os
from tqdm import tqdm
from typing import Any, Optional
import numpy as np

from utils.model_utils import get_dummy_input_for_rwkv_causal_llm, get_dummy_state_kvcache, register_customop_symbols

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Compute param encodings for linear modules')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser.add_argument('--lambada_test', action='store_true', help='Run lambada test')
# parser.add_argument('--use_old_format', action='store_true', help='Use old format for encodings')
parser.add_argument('--output_folder', type=Path, help='Output folder for encodings')
parser.add_argument('--use_w4_seq_mse', action='store_true', help='Use int4 quantization with SeqMse optimization')
parser.add_argument('--binidx_dataset', type=Path, default=None, help='Path to binidx dataset')
args_parser = parser.parse_args()

model_args = types.SimpleNamespace()
model_args.USE_CUDA = torch.cuda.is_available()
model_args.fp16 = True if model_args.USE_CUDA else False
model_args.USE_EMBEDDING = True
model_args.RESCALE_LAYER = 0
model_args.wkv_customop = True
model_args.split_wkv = True
model_args.output_last = False

model_args.MODEL_NAME = str(args_parser.model)

model = RWKV_RNN(model_args)
model_args = model.args

device = torch.device("cuda" if model_args.USE_CUDA else "cpu")

from aimet_common import quantsim
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
# from aimet_torch.v2.quantsim.config_utils import set_blockwise_quantization_for_weights, set_grouped_blockwise_quantization_for_weights, set_activation_quantizers_to_float
# from aimet_torch.quantization.affine import GroupedBlockQuantizeDequantize, QuantizeDequantize
from aimet_torch.v2.mixed_precision import MixedPrecisionConfigurator
from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.v2 import quantization as Q

from aimet_torch.seq_mse import apply_seq_mse, SeqMseParams
from utils.dataset_builder import DatasetBuilder

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

model.eval()
with torch.no_grad():
    sim = QuantizationSimModel(model, dummy_input=dummy_input,
                            quant_scheme=QuantScheme.post_training_tf_enhanced,
                            default_param_bw=8,
                            default_output_bw=16,
                            config_file="quantizers/configs/htp_quantsim_config_v75.json",
                            #    in_place=True,
    )
torch.cuda.empty_cache()

mp_configurator = MixedPrecisionConfigurator(sim)
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

    block.att.exp_w.output_quantizers[0] = None
    block.att.mix_ka_sub.output_quantizers[0] = None
    block.att.mix_ka_mul_key.output_quantizers[0] = None

    block.att.wkv7.split_state.input_quantizers[0] = None
    block.att.wkv7.split_state.output_quantizers[0] = None
    block.att.wkv7.concat_state.input_quantizers[0] = None
    block.att.wkv7.concat_state.output_quantizers[0] = None
    block.att.wkv7.concat_x.input_quantizers[0] = None
    block.att.wkv7.concat_x.output_quantizers[0] = None

    if args_parser.use_w4_seq_mse:
        set_linear_weight_quantizer_to_4bit(block.ffn.key)
        set_linear_weight_quantizer_to_4bit(block.ffn.value)
        set_linear_weight_quantizer_to_4bit(block.att.output)
        set_linear_weight_quantizer_to_4bit(block.att.key)
        set_linear_weight_quantizer_to_4bit(block.att.value)
        set_linear_weight_quantizer_to_4bit(block.att.receptance)

    # somehow it doesn't want to quantize ffn.key Linear by default
    block.ffn.key.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)

sim.model.head_pre_permute.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
sim.model.head_post_permute.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
sim.model.head_pre_reshape.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
sim.model.head_post_reshape.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
mp_configurator.apply()

print(sim)

tokenizer = RWKV_TOKENIZER("./assets/rwkv_vocab_v20230424.txt")

dataloader = None
if args_parser.binidx_dataset is not None:
    from utils.indexed_dataset import MMapIndexedDataset
    dataset = MMapIndexedDataset(str(args_parser.binidx_dataset))
    def collate_fn(x):
        return {'input_ids': torch.LongTensor(np.array(x, dtype=np.int64)).to(device)}
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

def pass_calibration_data_seq_mse(model: torch.nn.Module, forward_pass_args: Optional[Any]=None):
    model.eval()
    with torch.no_grad():
        state = get_dummy_state_kvcache(1, model.args, model.device)
        model(forward_pass_args['input_ids'].to(model.device), state)

def pass_calibration_data_calib(model: torch.nn.Module, forward_pass_args: Optional[Any]=None):
    data_loader = forward_pass_args

    num_batches = 20

    model.eval()
    with torch.no_grad():
        for batch, input_data in tqdm(enumerate(data_loader)):
            state = get_dummy_state_kvcache(1, model.args, model.device)
            model(input_data['input_ids'].to(model.device), state)

            if batch >= num_batches:
                break

if args_parser.use_w4_seq_mse:
    with torch.no_grad():
        params = SeqMseParams(num_batches=20,
                            num_candidates=20,
                            inp_symmetry='symqt',
                            loss_fn='mse',
                            forward_fn=pass_calibration_data_seq_mse)

        apply_seq_mse(model=model, sim=sim, data_loader=dataloader, params=params)

model = model.to('cpu').float()
torch.cuda.empty_cache()

# NOTE: looks unusable with QNN yet
# for block in sim.model.blocks:
#     # set_activation_quantizers_to_float(sim=sim, arg=[block.ffn.value], dtype=torch.float16)
#     # set_blockwise_quantization_for_weights(sim=sim,
#     #                                           arg=[block.ffn.value],
#     #                                           bitwidth=4,
#     #                                           symmetric=True,
#     #                                           block_size=128)
#     set_grouped_blockwise_quantization_for_weights(
#         sim, [block.ffn.value], bitwidth=4, symmetric=True, decompressed_bw=8, block_size=128, block_grouping=-1
#     )


sim.compute_encodings(pass_calibration_data_calib, forward_pass_callback_args=dataloader)

sim.model.float()
model.args.fp16 = False
sim.model.eval()

if args_parser.lambada_test:
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

# 12.279, 0.52 for 300 samples v7 0.1B fp32
# 7.142, 0.593 for 300 samples v7 0.4B fp32

# 4.336, 0.68 for 300 samples v7 1.5B fp32
# 4.86, 0.65 for 300 samples v7 1.5B a16w8 in AIMET quantsim (post_training_tf_enhanced schema)
# 4.03, 0.676 for 300 samples v7 1.5B [ffn.key/value, att.output] a16w4 + others a16w8 in AIMET quantsim (post_training_tf_enhanced schema)
# (TODO) for 300 samples v7 1.5B a16w8 on actual hardware

input_names = ['in'] + [f'state{j}_in' for j in range(3*model.layer_begin, 3*model.layer_end)]
output_names = ['out'] + [f'state{j}_out' for j in range(3*model.layer_begin, 3*model.layer_end)]

input_names_prefill = ['in_prefill'] + [f'state{j}_in' for j in range(3*model.layer_begin, 3*model.layer_end)]
output_names_prefill = ['out_prefill'] + [f'state{j}_out' for j in range(3*model.layer_begin, 3*model.layer_end)]

dummy_input = get_dummy_input_for_rwkv_causal_llm(1, 1, "cpu", model.args)
dummy_input = (dummy_input['in0'], dummy_input['state'])

dummy_input_prefill = get_dummy_input_for_rwkv_causal_llm(1, 128, "cpu", model.args)
dummy_input_prefill = (dummy_input_prefill['in0'], dummy_input_prefill['state'])

filename = model.args.MODEL_NAME.split('/')[-1].replace('.pth', '')
prefill_filename = filename + '_prefill'
output_path = './tmp' if args_parser.output_folder is None else str(args_parser.output_folder)

# if not args_parser.use_old_format:
if False:
    # for exporting Qualcomm's LPBQ parameters
    quantsim.encoding_version = '1.0.0'

sim.model.to('cpu')
for module in sim.model.modules():
    module.to('cpu')

for q in sim.qmodules():
    for _, item in q.param_quantizers.items():
        if item is not None:
            item.min.to('cpu')
            item.max.to('cpu')

torch.cuda.empty_cache()

# why the fuck is it sticking to CUDA
print(sim.model.device)
os.path.exists(output_path) or os.makedirs(output_path)
# sim.export(path=output_path, filename_prefix=filename, dummy_input=dummy_input, onnx_export_args={'input_names': input_names, 'output_names': output_names})
# sim.export(path=output_path, filename_prefix=prefill_filename, dummy_input=dummy_input_prefill, onnx_export_args={'input_names': input_names_prefill, 'output_names': output_names_prefill})

sim.export_onnx_model_and_encodings(output_path, filename, model, sim.model,
                                    dummy_input, {'input_names': input_names, 'output_names': output_names}, False,
                                    sim._module_marker_map, sim._is_conditional,
                                    sim._excluded_layer_names, quantizer_args=sim.quant_args,
                                    export_model=True,
                                    filename_prefix_encodings=filename)
sim.export_onnx_model_and_encodings(output_path, prefill_filename, model, sim.model,
                                    dummy_input_prefill, {'input_names': input_names, 'output_names': output_names}, False,
                                    sim._module_marker_map, sim._is_conditional,
                                    sim._excluded_layer_names, quantizer_args=sim.quant_args,
                                    export_model=True,
                                    filename_prefix_encodings=prefill_filename)

# set corresponding state_in/out to the same parameters if quantized
import json
with open(output_path + '/' + filename + '.encodings', 'r') as f:
    encodings = json.load(f)

with open(output_path + '/' + prefill_filename + '.encodings', 'r') as f:
    encodings_prefill = json.load(f)

# if args_parser.use_old_format:
if True:
    act_fp_override = [{"bitwidth": 16, "dtype": "float"}]
    keys = list(encodings['activation_encodings'].keys())
    for key in keys:
        if 'state' in key and 'out' in key:
            encodings['activation_encodings'][key.replace('out', 'in')] = encodings['activation_encodings'][key]

    for i in range(model.args.n_layer):
        for j in range(4):
            for o in ['r', 'w', 'k', 'v', 'a', 'b', 'state']:
                encodings['activation_encodings'][f'/blocks.{i}/att/wkv7/split_{o}/Split_output_{j}'] = act_fp_override
        # try:
        #     del encodings['activation_encodings'][f'state{3*i+1}_in']
        #     del encodings['activation_encodings'][f'state{3*i+1}_out']
        # except:
        #     pass

    for k in keys:
        if 'state' in k:
            encodings_prefill['activation_encodings'][k] = encodings['activation_encodings'][k]

    for i in range(model.args.n_layer):
        for j in range(4):
            for o in ['r', 'w', 'k', 'v', 'a', 'b', 'state']:
                encodings_prefill['activation_encodings'][f'/blocks.{i}/att/wkv7/split_{o}/Split_output_{j}'] = act_fp_override
        # try:
        #     del encodings_prefill['activation_encodings'][f'state{3*i+1}_in']
        #     del encodings_prefill['activation_encodings'][f'state{3*i+1}_out']
        # except:
        #     pass
else:
    for entry in encodings['activation_encodings']:
        if 'state' in entry['name'] and 'out' in entry['name']:
            idx = int(entry['name'].split('_')[0].replace('state', ''))
            encodings_prefill['activation_encodings'].append(entry)
            if idx % 3 == 0:
                for n in encodings_prefill['activation_encodings']:
                    if n['name'] == f'/blocks.{idx//3}/att/concat_shift/Concat_output_0':
                        n['offset'] = entry['offset']
                        n['scale'] = entry['scale']
            if idx % 3 == 2:
                for n in encodings_prefill['activation_encodings']:
                    if n['name'] == f'/blocks.{idx//3}/ffn/concat_shift/Concat_output_0':
                        n['offset'] = entry['offset']
                        n['scale'] = entry['scale']
            for n in encodings['activation_encodings']:
                if n['name'] == entry['name'].replace('out', 'in'):
                    n['offset'] = entry['offset']
                    n['scale'] = entry['scale']
                    encodings_prefill['activation_encodings'].append(n)
    for i in range(model.args.n_layer):
        for j in range(4):
            for o in ['r', 'w', 'k', 'v', 'a', 'b', 'state']:
                encodings['activation_encodings'].append({
                    "name": f'/blocks.{i}/att/wkv7/split_{o}/Split_output_{j}',
                    "bw": 16,
                    "dtype": "FLOAT",
                })
                encodings_prefill['activation_encodings'].append({
                    "name": f'/blocks.{i}/att/wkv7/split_{o}/Split_output_{j}',
                    "bw": 16,
                    "dtype": "FLOAT",
                })

with open(output_path + '/' + filename + '.encodings', 'w') as f:
    json.dump(encodings, f, indent=4)

with open(output_path + '/' + prefill_filename + '.encodings', 'w') as f:
    json.dump(encodings_prefill, f, indent=4)
