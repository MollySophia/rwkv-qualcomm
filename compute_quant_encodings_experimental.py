from rwkv_src.rwkv_model import RWKV_RNN
from rwkv_src.rwkv_tokenizer import RWKV_TOKENIZER
from rwkv_src.rwkv_v7_modules_conv import Wkv7OutputState, Wkv7Op, Wkv7OutputX, L2Norm
import types
import torch
import math
import os
import json
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
parser.add_argument('--blockwise_quant', action='store_true', help='Use blockwise quantization')
parser.add_argument('--blockwise_block_size', type=int, default=16,
                    help='LPBQ block size used with --blockwise_quant')
parser.add_argument('--output_folder', type=Path, help='Output folder for encodings')
parser.add_argument('--use_w4_seq_mse', action='store_true', help='Use int4 quantization with SeqMse optimization')
parser.add_argument('--use_w4_omniquant', action='store_true', help='Use int4 quantization with Omniquant optimization')
parser.add_argument('--use_w4_adascale', action='store_true', help='Use int4 quantization with Adascale optimization')
parser.add_argument('--binidx_dataset', type=Path, default=None, help='Path to binidx dataset')
parser.add_argument('--calib_num_batches', type=int, default=10, help='Number of batches to calibrate on')
parser.add_argument('--seqmse_num_batches', type=int, default=20, help='Number of batches to calibrate on')
parser.add_argument('--use_cpu', action='store_true', default=False, help='Use cpu to compute')
parser.add_argument('--load_encodings', type=Path, default=None, help='Path to load encodings from')
parser.add_argument('--w8_embedding', action='store_true', help='Use int8 quantization for embedding')
parser.add_argument('--w4_head', action='store_true', help='Use int4 quantization for head')
parser.add_argument('--heads_per_split', type=int, default=8, help='Number of heads per split')
parser.add_argument('--data_no_padding', action='store_true', help='Do not pad the data')
parser.add_argument('--reuse_encodings_folder', type=Path, default=None,
                    help='Reuse activation and non-W4 parameter encodings from this folder')
parser.add_argument('--blockwise_extra_modules', type=str, default='',
                    help='Comma-separated extra module groups to export as LPBQ: att_output,lora,head')
parser.add_argument('--blockwise_extra_att_output_layers', type=str, default='',
                    help='Optional comma-separated layer ids/ranges for att_output, e.g. 0-3,8')
parser.add_argument('--blockwise_skip_att_lpbq_layers', type=str, default='',
                    help='Optional comma-separated layer ids/ranges where attention LPBQ projections stay W8')
parser.add_argument('--blockwise_skip_att_lpbq_projections', type=str, default='',
                    help='Comma-separated attention projections to keep W8 on skipped layers: receptance/key/value or r/k/v')
parser.add_argument('--allow_extra_lpbq', action='store_true',
                    help='Allow blockwise export to add LPBQ params beyond the reused baseline W4 set')
parser.add_argument('--calib_stream_chunk_size', type=int, default=0,
                    help='If >0, run each calibration sample as stateful chunks of this token length')
parser.add_argument('--no_cleanup', action='store_true',
                    help='Do not delete temporary export files from output_folder')
args_parser = parser.parse_args()
VALID_BLOCKWISE_EXTRA_MODULES = {'att_output', 'lora', 'head'}
BLOCKWISE_EXTRA_MODULES = {
    item.strip()
    for item in args_parser.blockwise_extra_modules.split(',')
    if item.strip()
}
unknown_blockwise_extras = BLOCKWISE_EXTRA_MODULES - VALID_BLOCKWISE_EXTRA_MODULES
if unknown_blockwise_extras:
    parser.error(
        '--blockwise_extra_modules only supports '
        f'{sorted(VALID_BLOCKWISE_EXTRA_MODULES)}, got {sorted(unknown_blockwise_extras)}'
    )

def parse_layer_set(value):
    layers = set()
    if not value:
        return layers
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        if '-' in item:
            start, end = item.split('-', 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(item))
    return layers

BLOCKWISE_EXTRA_ATT_OUTPUT_LAYERS = parse_layer_set(args_parser.blockwise_extra_att_output_layers)
BLOCKWISE_SKIP_ATT_LPBQ_LAYERS = parse_layer_set(args_parser.blockwise_skip_att_lpbq_layers)

ATT_PROJECTION_ALIASES = {
    'r': 'receptance',
    'receptance': 'receptance',
    'k': 'key',
    'key': 'key',
    'v': 'value',
    'value': 'value',
}

def parse_att_projection_set(value):
    projections = set()
    if not value:
        return projections
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        if item not in ATT_PROJECTION_ALIASES:
            parser.error(
                '--blockwise_skip_att_lpbq_projections only supports '
                f'{sorted(ATT_PROJECTION_ALIASES)}, got {item}'
            )
        projections.add(ATT_PROJECTION_ALIASES[item])
    return projections

BLOCKWISE_SKIP_ATT_LPBQ_PROJECTIONS = parse_att_projection_set(
    args_parser.blockwise_skip_att_lpbq_projections
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False #if args_parser.use_cpu else torch.cuda.is_available()
model_args.fp16 = True if model_args.USE_CUDA else False
# model_args.bf16 = True if model_args.USE_CUDA else False
# model_args.fp16 = False
model_args.bf16 = False
model_args.USE_EMBEDDING = True
model_args.RESCALE_LAYER = 0
model_args.wkv_customop = True
model_args.output_last = False
model_args.EXTERNAL_HEAD = False
model_args.heads_per_split = args_parser.heads_per_split

model_args.MODEL_NAME = str(args_parser.model)

model = RWKV_RNN(model_args)
model_args = model.args
has_deep_embedding = model_args.has_deepemb

device = torch.device("cuda" if model_args.USE_CUDA else "cpu")

import aimet_torch
from aimet_torch import onnx_utils
onnx_utils.EXPORT_TO_ONNX_DIRECT = True
from aimet_common import quantsim
from aimet_common.defs import QuantScheme
import aimet_torch._base.quantsim as aimet_base_quantsim
import onnx
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.v2.quantsim.config_utils import set_blockwise_quantization_for_weights, set_grouped_blockwise_quantization_for_weights, set_activation_quantizers_to_float
# from aimet_torch.quantization.affine import GroupedBlockQuantizeDequantize, QuantizeDequantize
# from aimet_torch.v2.mixed_precision import MixedPrecisionConfigurator
from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.v2.nn.true_quant import QuantizedConv2d
from aimet_torch.v2 import quantization as Q
from aimet_torch.experimental.omniquant import apply_omniquant
# from aimet_torch.experimental.adascale import apply_adascale
from aimet_torch.v2.experimental.quantsim_utils import clip_weights_to_7f7f
# modified to support rwkv
from quantizers.adascale_optimizer import apply_adascale

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

in_place = False

sim = QuantizationSimModel(model, dummy_input=dummy_input,
                        quant_scheme=QuantScheme.post_training_tf_enhanced if not args_parser.use_w4_omniquant else QuantScheme.training_range_learning_with_tf_init,
                        # quant_scheme=QuantScheme.post_training_percentile,
                        default_param_bw=8,
                        default_output_bw=16,
                        config_file="quantizers/configs/htp_quantsim_config_v75.json",
                        in_place=in_place,
)
# sim.set_percentile_value(99.999)
torch.cuda.empty_cache()

if args_parser.w8_embedding:
    sim.model.embedding.param_quantizers['weight'] = Q.affine.Quantize((), bitwidth=8, symmetric=False).to(device)

# mp_configurator = MixedPrecisionConfigurator(sim)
def set_linear_weight_quantizer_to_4bit(module):
    if module.param_quantizers['weight'] is not None:
        module.param_quantizers['weight'].bitwidth = 4
        module.param_quantizers['weight'].symmetric = True

def set_lpbq_weight_quantizers_to_match_omniquant(
        block, extra_modules=None, att_output_layers=None,
        skip_att_layers=None, skip_att_projections=None):
    """Match convert_omniquant_parameters.py's exported W4 parameter list."""
    extra_modules = extra_modules or set()
    att_output_layers = att_output_layers or set()
    skip_att_layers = skip_att_layers or set()
    skip_att_projections = skip_att_projections or set()

    def should_skip_att_projection(projection):
        return (
            block.att.layer_id in skip_att_layers
            and projection in skip_att_projections
        )

    set_linear_weight_quantizer_to_4bit(block.ffn.key)
    set_linear_weight_quantizer_to_4bit(block.ffn.value)
    for head in block.att.heads:
        if not should_skip_att_projection('receptance'):
            set_linear_weight_quantizer_to_4bit(head.receptance)
        if not should_skip_att_projection('key'):
            set_linear_weight_quantizer_to_4bit(head.key)
        if block.att.layer_id != 0 and not should_skip_att_projection('value'):
            set_linear_weight_quantizer_to_4bit(head.value)
    if 'att_output' in extra_modules and (not att_output_layers or block.att.layer_id in att_output_layers):
        set_linear_weight_quantizer_to_4bit(block.att.output)
    if 'lora' in extra_modules:
        set_linear_weight_quantizer_to_4bit(block.att.matmul_a1)
        set_linear_weight_quantizer_to_4bit(block.att.matmul_g1)
        set_linear_weight_quantizer_to_4bit(block.att.matmul_time_decay_w1)
        if block.att.layer_id != 0:
            set_linear_weight_quantizer_to_4bit(block.att.matmul_v1)

def count_w4_weight_quantizers(model):
    count = 0
    for module in model.modules():
        if not hasattr(module, 'param_quantizers'):
            continue
        if 'weight' not in module.param_quantizers:
            continue
        weight_quantizer = module.param_quantizers['weight']
        if weight_quantizer is not None and weight_quantizer.bitwidth == 4:
            count += 1
    return count

def iter_lpbq_weight_modules(model):
    for module_name, module in model.named_modules():
        if not hasattr(module, 'param_quantizers'):
            continue
        if 'weight' not in module.param_quantizers:
            continue
        weight_quantizer = module.param_quantizers['weight']
        if weight_quantizer is None:
            continue
        if weight_quantizer.__class__.__name__ == 'GroupedBlockQuantizeDequantize':
            yield module_name, module

def write_json_value(file, value):
    json.dump(value, file, separators=(',', ':'))

def write_lpbq_encodings_from_reused_baseline(
        sim, output_path, filename, prefill_filename, reuse_encodings_folder,
        allow_extra_lpbq=False):
    decode_baseline_path = os.path.join(reuse_encodings_folder, filename + '.encodings')
    prefill_baseline_path = os.path.join(reuse_encodings_folder, prefill_filename + '.encodings')
    with open(decode_baseline_path, 'r') as f:
        decode_baseline = json.load(f)
    with open(prefill_baseline_path, 'r') as f:
        prefill_baseline = json.load(f)

    decode_w4_names = {
        entry['name'] for entry in decode_baseline['param_encodings']
        if entry.get('bw') == 4 or entry.get('compressed_bw') == 4
    }
    prefill_w4_names = {
        entry['name'] for entry in prefill_baseline['param_encodings']
        if entry.get('bw') == 4 or entry.get('compressed_bw') == 4
    }
    if decode_w4_names != prefill_w4_names:
        raise RuntimeError('Decode and prefill baseline W4 parameter sets differ')

    lpbq_modules = list(iter_lpbq_weight_modules(sim.model))
    lpbq_names = {module_name + '.weight' for module_name, _ in lpbq_modules}
    if allow_extra_lpbq:
        if not decode_w4_names.issubset(lpbq_names):
            missing = sorted(decode_w4_names - lpbq_names)
            raise RuntimeError(f'LPBQ parameter set is missing baseline W4 params. missing={missing[:10]}')
        extra = sorted(lpbq_names - decode_w4_names)
        print(f'Extra LPBQ params beyond baseline: {len(extra)}')
        for name in extra[:20]:
            print(f'  {name}')
    elif lpbq_names != decode_w4_names:
        missing = sorted(decode_w4_names - lpbq_names)
        extra = sorted(lpbq_names - decode_w4_names)
        raise RuntimeError(f'LPBQ parameter set does not match baseline W4 set. missing={missing[:10]} extra={extra[:10]}')

    decode_non_w4_params = [
        entry for entry in decode_baseline['param_encodings']
        if entry['name'] not in lpbq_names
    ]
    prefill_non_w4_params = [
        entry for entry in prefill_baseline['param_encodings']
        if entry['name'] not in lpbq_names
    ]

    os.makedirs(output_path, exist_ok=True)
    decode_tmp_path = os.path.join(output_path, filename + '.encodings.tmp')
    prefill_tmp_path = os.path.join(output_path, prefill_filename + '.encodings.tmp')
    decode_path = os.path.join(output_path, filename + '.encodings')
    prefill_path = os.path.join(output_path, prefill_filename + '.encodings')

    def write_header(file, baseline, non_w4_params):
        file.write('{"activation_encodings":')
        write_json_value(file, baseline['activation_encodings'])
        file.write(',"excluded_layers":')
        write_json_value(file, baseline.get('excluded_layers', []))
        file.write(',"param_encodings":[')
        first = True
        for entry in non_w4_params:
            if not first:
                file.write(',')
            write_json_value(file, entry)
            first = False
        return first

    def write_footer(file, baseline):
        file.write('],"quantizer_args":')
        write_json_value(file, baseline.get('quantizer_args', {}))
        file.write(',"version":')
        write_json_value(file, baseline.get('version', '1.0.0'))
        file.write('}\n')

    with open(decode_tmp_path, 'w') as decode_file, open(prefill_tmp_path, 'w') as prefill_file:
        first_decode = write_header(decode_file, decode_baseline, decode_non_w4_params)
        first_prefill = write_header(prefill_file, prefill_baseline, prefill_non_w4_params)

        total = len(lpbq_modules)
        for idx, (module_name, module) in enumerate(lpbq_modules, 1):
            lpbq_encoding = module.export_param_encodings('1.0.0')['weight']
            lpbq_entry = {'name': module_name + '.weight', **lpbq_encoding}
            if not first_decode:
                decode_file.write(',')
            write_json_value(decode_file, lpbq_entry)
            first_decode = False
            if not first_prefill:
                prefill_file.write(',')
            write_json_value(prefill_file, lpbq_entry)
            first_prefill = False
            if idx == 1 or idx % 50 == 0 or idx == total:
                print(f'Exported LPBQ param encodings: {idx}/{total}')

        write_footer(decode_file, decode_baseline)
        write_footer(prefill_file, prefill_baseline)

    os.replace(decode_tmp_path, decode_path)
    os.replace(prefill_tmp_path, prefill_path)
    print(f'Wrote LPBQ encodings: {decode_path}')
    print(f'Wrote LPBQ prefill encodings: {prefill_path}')

def export_onnx_model_and_encodings_light(sim, path, filename_prefix, original_model, sim_model,
                                          dummy_input, onnx_export_args, propagate_encodings,
                                          module_marker_map=None, is_conditional=False,
                                          excluded_layer_names=None, quantizer_args=None,
                                          export_model=True, filename_prefix_encodings=None):
    """Avoid loading large ONNX external weight data when only graph metadata is needed."""
    if not filename_prefix_encodings:
        filename_prefix_encodings = filename_prefix
    onnx_path = os.path.join(path, filename_prefix + ".onnx")
    if export_model:
        onnx_utils.OnnxSaver.create_onnx_model_with_pytorch_layer_names(
            onnx_path,
            original_model,
            dummy_input,
            is_conditional,
            module_marker_map,
            onnx_export_args,
        )

    assert os.path.exists(onnx_path), (
        "The onnx model does not exist in the location specified. Please re-run export "
        "with export_model flag as True or check path/file_name"
    )
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    onnx_node_to_io_tensor_map, valid_param_set = (
        onnx_utils.OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)
    )

    sim.__class__._export_encodings_to_files(
        sim_model,
        path,
        filename_prefix_encodings,
        onnx_node_to_io_tensor_map,
        valid_param_set,
        excluded_layer_names,
        propagate_encodings,
        quantizer_args=quantizer_args,
    )

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

    if args_parser.blockwise_quant:
        set_lpbq_weight_quantizers_to_match_omniquant(
            block,
            BLOCKWISE_EXTRA_MODULES,
            BLOCKWISE_EXTRA_ATT_OUTPUT_LAYERS,
            BLOCKWISE_SKIP_ATT_LPBQ_LAYERS,
            BLOCKWISE_SKIP_ATT_LPBQ_PROJECTIONS,
        )
    elif args_parser.use_w4_seq_mse or args_parser.use_w4_omniquant or args_parser.use_w4_adascale:
        set_linear_weight_quantizer_to_4bit(block.ffn.key)
        set_linear_weight_quantizer_to_4bit(block.ffn.value)
        set_linear_weight_quantizer_to_4bit(block.att.output)
        set_linear_weight_quantizer_to_4bit(block.att.key)
        set_linear_weight_quantizer_to_4bit(block.att.value)
        set_linear_weight_quantizer_to_4bit(block.att.receptance)
        if True:
            set_linear_weight_quantizer_to_4bit(block.att.matmul_a1)
            set_linear_weight_quantizer_to_4bit(block.att.matmul_g1)
            set_linear_weight_quantizer_to_4bit(block.att.matmul_time_decay_w1)
            if block.att.layer_id != 0:
                set_linear_weight_quantizer_to_4bit(block.att.matmul_v1)

sim.model.head_pre_permute.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
sim.model.head_post_permute.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
sim.model.head_pre_reshape.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
sim.model.head_post_reshape.output_quantizers[0] = Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)
# mp_configurator.apply()
if args_parser.use_w4_seq_mse or args_parser.blockwise_quant or args_parser.use_w4_omniquant or args_parser.use_w4_adascale:
    if args_parser.w4_head or 'head' in BLOCKWISE_EXTRA_MODULES:
        set_linear_weight_quantizer_to_4bit(sim.model.head)

if args_parser.blockwise_quant:
    print(f"LPBQ W4 weight quantizers: {count_w4_weight_quantizers(sim.model)}")

if has_deep_embedding:
    for i in range(len(sim.model.deep_emb)):
        sim.model.deep_emb[i].param_quantizers['weight'] = Q.affine.Quantize((), bitwidth=8, symmetric=False).to(device)

tokenizer = RWKV_TOKENIZER("./assets/rwkv_vocab_v20230424.txt")

dataloader = None
if args_parser.binidx_dataset is not None:
    from utils.indexed_dataset import MMapIndexedDataset
    dataset = MMapIndexedDataset(str(args_parser.binidx_dataset))
    if not args_parser.data_no_padding:
        block_size = 4096
        print(f"Padding data to {block_size} tokens")
        len_total = 1
        took = []
        tokens = np.array([0])
        while len_total < max(args_parser.calib_num_batches, args_parser.seqmse_num_batches) * block_size:
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
    stream_chunk_size = args_parser.calib_stream_chunk_size

    model.eval()
    with torch.no_grad():
        for batch, input_data in tqdm(enumerate(data_loader)):
            state = get_dummy_state_kvcache(1, model.args, model.device)
            input_ids = input_data['input_ids'].to(model.device)
            if stream_chunk_size and input_ids.shape[1] > stream_chunk_size:
                for start in range(0, input_ids.shape[1], stream_chunk_size):
                    end = min(start + stream_chunk_size, input_ids.shape[1])
                    ret = model(input_ids[:, start:end], state)
                    state = ret[1]
                    state = [item.detach() for item in state]
            else:
                model(input_ids, state)

            if batch >= num_batches:
                break

if args_parser.load_encodings:
    sim.load_encodings(args_parser.load_encodings, allow_overwrite=False)
elif args_parser.use_w4_adascale:
    sim.model.config = types.SimpleNamespace()
    sim.model.config.use_cache = False
    sim.model.args.fp16 = False
    sim.model.args.bf16 = True
    apply_adascale(qsim=sim,
               data_loader=dataloader,
               forward_fn=pass_calibration_data,
               num_iterations=300)
    output_path = './tmp' if args_parser.output_folder is None else str(args_parser.output_folder)
    os.path.exists(output_path) or os.makedirs(output_path)
    sim.save_encodings_to_json(output_path, 'quant_encodings_checkpoint_adascale')
    sim.model.args.fp16 = True
    sim.model.args.bf16 = False
elif args_parser.use_w4_omniquant:
    sim.model.config = types.SimpleNamespace()
    sim.model.config.use_cache = False
    apply_omniquant(quant_sim=sim,
               dataloader=dataloader,
               forward_fn=pass_calibration_data,
               num_iterations=800)
    output_path = './tmp' if args_parser.output_folder is None else str(args_parser.output_folder)
    os.path.exists(output_path) or os.makedirs(output_path)
    sim.save_encodings_to_json(output_path, 'quant_encodings_checkpoint_omniquant')
elif args_parser.use_w4_seq_mse:
    with torch.no_grad():
        params = SeqMseParams(num_batches=args_parser.seqmse_num_batches,
                            num_candidates=20,
                            inp_symmetry='symqt',
                            loss_fn='mse',
                            forward_fn=pass_calibration_data)

        apply_seq_mse(model=model, sim=sim, data_loader=dataloader, params=params)
    output_path = './tmp' if args_parser.output_folder is None else str(args_parser.output_folder)
    os.path.exists(output_path) or os.makedirs(output_path)
    sim.save_encodings_to_json(output_path, 'quant_encodings_checkpoint_seqmse')
elif args_parser.blockwise_quant:
    fn = lambda module: isinstance(module, QuantizedConv2d) and module.param_quantizers['weight'].bitwidth == 4

    BLOCK_QUANT_SIZE = args_parser.blockwise_block_size
    BITWIDTH = 4
    DECOMPRESSED_BITWIDTH = 8

    set_grouped_blockwise_quantization_for_weights(sim = sim,
                                                arg = fn,
                                                bitwidth = BITWIDTH,
                                                symmetric = True,
                                                decompressed_bw = DECOMPRESSED_BITWIDTH,
                                                block_size = BLOCK_QUANT_SIZE,
                                                block_grouping = -1)

# print(sim)
if os.getenv('RWKV_PRINT_QSIM_ATT'):
    print(sim.model.blocks[0].att)

if not in_place:
    model = model.to('cpu').float()
torch.cuda.empty_cache()

if torch.cuda.is_available() and not args_parser.use_cpu:
    device = "cuda"
    model.args.fp16 = False
    model.args.bf16 = True
    sim.model = sim.model.bfloat16().to(device)
    sim.model.device = device
    sim.model.args.fp16 = False
    sim.model.args.bf16 = True

sim.compute_encodings(pass_calibration_data_calib, forward_pass_callback_args=dataloader)

clip_weights_to_7f7f(sim)

from aimet_torch.v2.experimental import propagate_output_encodings
import aimet_torch._base.nn.modules.custom as aimet_ops
propagate_output_encodings(sim, aimet_ops.Concat)

output_path = './tmp' if args_parser.output_folder is None else str(args_parser.output_folder)
os.path.exists(output_path) or os.makedirs(output_path)
if not args_parser.blockwise_quant:
    # too slow to save encodings for blockwise quant
    sim.save_encodings_to_json(output_path, 'quant_encodings_checkpoint_calib')

sim.model.to('cpu')
for module in sim.model.modules():
    module.to('cpu')

for q in sim.qmodules():
    for _, item in q.param_quantizers.items():
        if item is not None:
            item.min.to('cpu')
            item.max.to('cpu')

sim.model.float()
model.args.fp16 = False
model.args.bf16 = False
sim.model.eval()

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

if args_parser.blockwise_quant and args_parser.reuse_encodings_folder is not None:
    quantsim.encoding_version = '1.0.0'
    write_lpbq_encodings_from_reused_baseline(
        sim,
        output_path,
        filename,
        prefill_filename,
        str(args_parser.reuse_encodings_folder),
        allow_extra_lpbq=args_parser.allow_extra_lpbq,
    )
    raise SystemExit(0)

# if not args_parser.use_old_format:
if args_parser.blockwise_quant:
    # for exporting Qualcomm's LPBQ parameters
    quantsim.encoding_version = '1.0.0'
    aimet_base_quantsim.SKIP_TORCH_ENCODINGS_EXPORT = True

torch.cuda.empty_cache()

# print(sim.model.device)
os.path.exists(output_path) or os.makedirs(output_path)
# sim.export(path=output_path, filename_prefix=filename, dummy_input=dummy_input, onnx_export_args={'input_names': input_names, 'output_names': output_names})
# sim.export(path=output_path, filename_prefix=prefill_filename, dummy_input=dummy_input_prefill, onnx_export_args={'input_names': input_names_prefill, 'output_names': output_names_prefill})

if args_parser.blockwise_quant:
    export_fn = export_onnx_model_and_encodings_light
else:
    export_fn = lambda sim, *args, **kwargs: sim.export_onnx_model_and_encodings(*args, **kwargs)
export_fn(sim, output_path, filename, model, sim.model,
          dummy_input, {'input_names': input_names, 'output_names': output_names}, False,
          sim._module_marker_map, sim._is_conditional,
          sim._excluded_layer_names, quantizer_args=sim.quant_args,
          export_model=True,
          filename_prefix_encodings=filename)
export_fn(sim, output_path, prefill_filename, model, sim.model,
          dummy_input_prefill, {'input_names': input_names_prefill, 'output_names': output_names_prefill}, False,
          sim._module_marker_map, sim._is_conditional,
          sim._excluded_layer_names, quantizer_args=sim.quant_args,
          export_model=True,
          filename_prefix_encodings=prefill_filename)

# set corresponding state_in/out to the same parameters if quantized
with open(output_path + '/' + filename + '.encodings', 'r') as f:
    encodings = json.load(f)

with open(output_path + '/' + prefill_filename + '.encodings', 'r') as f:
    encodings_prefill = json.load(f)

for entry in encodings['activation_encodings']:
    if 'state' in entry['name'] and 'out' in entry['name']:
        idx = int(entry['name'].split('_')[0].replace('state', ''))
        if idx % 3 == 0:
            for n in encodings_prefill['activation_encodings']:
                if n['name'] == f'/blocks/blocks.{idx//3}/att/concat_shift/Concat_output_0':
                    n['offset'] = entry['offset']
                    n['scale'] = entry['scale']
        if idx % 3 == 2:
            for n in encodings_prefill['activation_encodings']:
                if n['name'] == f'/blocks/blocks.{idx//3}/ffn/concat_shift/Concat_output_0':
                    n['offset'] = entry['offset']
                    n['scale'] = entry['scale']
        if idx % 3 != 1:
            encodings_prefill['activation_encodings'].append(entry)
            for n in encodings['activation_encodings']:
                if n['name'] == entry['name'].replace('out', 'in'):
                    n['offset'] = entry['offset']
                    n['scale'] = entry['scale']
                    encodings_prefill['activation_encodings'].append(n)
for entry in encodings['param_encodings']:
    if 'embedding.weight' in entry['name']:
        tmp = copy.deepcopy(entry)
        tmp['name'] = '/pre_ln/LayerNormalization_output_0'
        encodings['activation_encodings'].append(tmp)
        encodings_prefill['activation_encodings'].append(tmp)

with open(output_path + '/' + filename + '.encodings', 'w') as f:
    json.dump(encodings, f, indent=4)

with open(output_path + '/' + prefill_filename + '.encodings', 'w') as f:
    json.dump(encodings_prefill, f, indent=4)

if not args_parser.no_cleanup:
    # Delete all files in output_path except .encodings and .json files
    for file in os.listdir(output_path):
        filepath = os.path.join(output_path, file)
        if not (file.endswith('.encodings') or file.endswith('.json')):
            if os.path.isfile(filepath):
                os.remove(filepath)
