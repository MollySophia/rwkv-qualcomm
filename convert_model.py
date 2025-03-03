from rwkv_src.rwkv_model import RWKV_RNN, make_chunks
from rwkv_src.rwkv_v7_modules import Wkv7, L2Norm
from utils.model_utils import register_customop_symbols, get_dummy_state_kvcache
import types
import os
import torch
import numpy as np
import argparse
import json
import copy
from pathlib import Path
import onnx
from onnx import shape_inference
from aimet_torch.onnx_utils import OnnxSaver

register_customop_symbols()

parser = argparse.ArgumentParser(description='Convert model')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser.add_argument('--chunks', type=int, default=2, help='Number of chunks')
parser.add_argument('--qnn_float_width', type=int, default=32, help='QNN float width')
parser.add_argument('--ext_embedding', action='store_true', default=False, help='Use external embedding')
parser.add_argument('--quant_encodings', type=Path, help='Path to quant encodings')
parser.add_argument('--prefill_model', action='store_true', help='Convert model for sequential prefill')
parser.add_argument('--wkv_customop', action='store_true', help='Use custom op for wkv')
parser_args = parser.parse_args()

seq_length = 128 if parser_args.prefill_model else 1

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False
model_args.fp16 = False
model_args.wkv_customop = parser_args.wkv_customop
model_args.USE_EMBEDDING = False if parser_args.ext_embedding else True
model_args.MODEL_NAME = str(parser_args.model)
model_args.split_wkv = True
model_args.output_last = True

if 'ABC' in model_args.MODEL_NAME or 'MIDI' in model_args.MODEL_NAME or parser_args.quant_encodings or 'x070' in model_args.MODEL_NAME:
    model_args.RESCALE_LAYER = 0
else:
    model_args.RESCALE_LAYER = 6

model = make_chunks(parser_args.chunks, model_args) if parser_args.chunks > 1 else RWKV_RNN(model_args)

qnn_sdk_root = os.environ["QNN_SDK_ROOT"]
if not qnn_sdk_root:
    print("Please set QNN_SDK_ROOT environment variable to the root of the Qualcomm Neural Processing SDK")
    exit(1)
os.path.exists("onnx") or os.mkdir("onnx")
if os.name == 'nt':
    qnn_tools_target = 'x86_64-windows-msvc'
else:
    qnn_tools_target = 'x86_64-linux-clang'

if type(model) == list:
    args = model[0].args
    filename = args.MODEL_NAME.split("/")[-1]
    if not args.USE_EMBEDDING:
        model[0].emb_weight.cpu().numpy().astype(np.float16 if args.fp16 else np.float32).tofile("onnx/" + filename + f"_chunk1of{len(model)}.emb")
    input_dtype = torch.float16 if args.fp16 else torch.float32

    states = get_dummy_state_kvcache(1, args, model[0].device)
    if parser_args.quant_encodings:
        with open(parser_args.quant_encodings, 'r') as f:
            encodings_all = json.load(f)

    for i in range(len(model)):
        dirname = "onnx/" + filename + f"_chunk{i+1}of{len(model)}"
        os.path.exists(dirname) or os.mkdir(dirname)
        if i == 0 and args.USE_EMBEDDING:
            in0 = torch.LongTensor([[1]*seq_length], device=model[i].device)
        else:
            in0 = torch.zeros(1, seq_length, args.n_embd, dtype=input_dtype)

        inputs = [in0, [states[j] for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]]
        input_name = ('in' if not parser_args.prefill_model else 'in_prefill') + f'_chunk{i+1}'
        output_name = ('out' if not parser_args.prefill_model else 'out_prefill') + f'_chunk{i+1}'
        input_names = [input_name] + [f'state{j}_in' for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
        output_names = [output_name] + [f'state{j}_out' for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]

        if i == 0:
            encodings_chunk = copy.deepcopy(encodings_all)
            encodings_chunk["activation_encodings"][output_name] = encodings_all["activation_encodings"][f'/blocks.{model[1].layer_begin-1}/ffn/add_feed_forward/Add_output_0']
            with open(f"{dirname}/quant_encodings_chunk{i}.encodings", "w") as f:
                json.dump(encodings_chunk, f, sort_keys=True, indent=4)
        else:
            encodings_chunk = {"activation_encodings": {}, "excluded_layers": [], "param_encodings": {}, "quantizer_args": encodings_all["quantizer_args"], "version": encodings_all["version"]}
            for k, v in encodings_all["activation_encodings"].items():
                try:
                    encoding_block_id = int(k.split(".")[-1].split("/")[0]) if 'block' in k else args.n_layer-1
                except ValueError:
                    encoding_block_id = int(k.split(".")[1]) if 'block' in k else args.n_layer-1
                if any([f'state{j}' in k for j in range(3*model[i].layer_begin, 3*model[i].layer_end)] + [
                    (encoding_block_id >= model[i].layer_begin and encoding_block_id < model[i].layer_end),
                    k == '/blocks.0/att/value/MatMul_output_0',
                    k == f'/blocks.{model[i].layer_begin-1}/ffn/add_feed_forward/Add_output_0'
                ]):
                    if k == '/blocks.0/att/value/MatMul_output_0':
                        encodings_chunk["activation_encodings"][f'v_first_in{"_prefill" if parser_args.prefill_model else ""}_chunk{i+1}'] = v
                    elif k == f'/blocks.{model[i].layer_begin-1}/ffn/add_feed_forward/Add_output_0':
                        encodings_chunk["activation_encodings"][input_name] = v
                    else:
                        encodings_chunk["activation_encodings"][k.replace(f"blocks.{encoding_block_id}", f"blocks.{encoding_block_id-model[i].layer_begin}")] = v
            for k, v in encodings_all["param_encodings"].items():
                if 'embedding' in k:
                    encoding_block_id = 0
                else:
                    encoding_block_id = int(k.split(".")[1]) if 'block' in k else args.n_layer-1
                if any([f'state{j}' in k for j in range(3*model[i].layer_begin, 3*model[i].layer_end)] + [
                    (encoding_block_id >= model[i].layer_begin and encoding_block_id < model[i].layer_end)
                ]):
                    encodings_chunk["param_encodings"][k.replace(f"blocks.{encoding_block_id}", f"blocks.{encoding_block_id-model[i].layer_begin}")] = v
            with open(f"{dirname}/quant_encodings_chunk{i}.encodings", 'w') as f:
                json.dump(encodings_chunk, f, sort_keys=True, indent=4)

        if args.version == 7:
            if i == 0:
                output_names += [('v_first_out' if not parser_args.prefill_model else 'v_first_out_prefill') + f'_chunk{i+1}']
            else:
                input_names += [('v_first_in' if not parser_args.prefill_model else 'v_first_in_prefill') + f'_chunk{i+1}']
                inputs += [torch.zeros(seq_length, args.n_head, args.head_size, dtype=input_dtype)]

        if parser_args.prefill_model:
            onnx_output_path = f"{dirname}/{filename}_prefill_chunk{i+1}of{len(model)}.onnx"
        else:
            onnx_output_path = f"{dirname}/{filename}_chunk{i+1}of{len(model)}.onnx"
        OnnxSaver.create_onnx_model_with_pytorch_layer_names(onnx_output_path, model[i], tuple(inputs),
            False, None, {'input_names': input_names, 'output_names': output_names, 'opset_version': 17})
        shape_inference.infer_shapes_path(onnx_output_path)
        print(f"onnx model chunk{i} saved to {onnx_output_path}")

    print("Converting and compiling QNN models...")
    for i in range(len(model)):
        dirname = "onnx/" + filename + f"_chunk{i+1}of{len(model)}"
        if parser_args.prefill_model:
            onnx_path = f"{dirname}/{filename}_prefill_chunk{i+1}of{len(model)}.onnx"
        else:
            onnx_path = f"{dirname}/{filename}_chunk{i+1}of{len(model)}.onnx"
        os.path.exists(dirname) or os.mkdir(dirname)

        states_layout = "NONTRIVIAL"
        converter_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qnn-onnx-converter -i {onnx_path} --float_bitwidth {parser_args.qnn_float_width} "
        converter_cmd += " ".join([f'--input_layout "state{3*j+1}_in" "{states_layout}"' for j in range(model[i].layer_begin, model[i].layer_end)])
        if args.version == 7:
            converter_cmd += f' --input_layout "v_first_in{"_prefill" if parser_args.prefill_model else ""}_chunk{i+1}" "NONTRIVIAL"'
        if i != 0:
            converter_cmd += f' --input_layout "{input_name}" "NFC"'

        if parser_args.quant_encodings:
            converter_cmd += f" --quantization_overrides {dirname}/quant_encodings_chunk{i}.encodings --float_fallback"

        if model_args.wkv_customop:
            converter_cmd += " --op_package_config hexagon/RwkvWkvOpPackageCPU.xml --op_package_lib hexagon/CPU/RwkvWkvOpPackage/libs/x86_64-linux-clang/libRwkvWkvOpPackage.so:RwkvWkvOpPackageInterfaceProvider"
        print(converter_cmd)

        if os.name == 'nt':
            converter_cmd = "python " + converter_cmd
        os.system(converter_cmd)

        print("Compiling QNN model library...")
        compiling_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qnn-model-lib-generator -c {onnx_path.replace('.onnx', '.cpp')} -b {onnx_path.replace('.onnx', '.bin')}"
        if os.name == 'nt':
            compiling_cmd = "python " + compiling_cmd
        os.system(compiling_cmd)
else:
    args = model.args
    filename = args.MODEL_NAME.split("/")[-1]
    dirname = "onnx/" + filename
    os.path.exists(dirname) or os.mkdir(dirname)
    if not args.USE_EMBEDDING:
        model.emb_weight.cpu().numpy().astype(np.float16 if args.fp16 else np.float32).tofile(dirname + '/' + filename + ".emb")
    input_dtype = torch.float16 if args.fp16 else torch.float32

    in0 = torch.LongTensor([[1]*seq_length]) if args.USE_EMBEDDING else [torch.zeros(1, seq_length, args.n_embd, dtype=input_dtype)]
    states = get_dummy_state_kvcache(1, args, model.device)

    input_name = 'in' if not parser_args.prefill_model else 'in_prefill'
    output_name  = 'out' if not parser_args.prefill_model else 'out_prefill'
    input_names = [input_name] + [f'state{i}_in' for i in range(3*args.n_layer)]
    output_names = [output_name] + [f'state{i}_out' for i in range(3*args.n_layer)]

    if parser_args.prefill_model:
        onnx_output_path = f"{dirname}/{filename}_prefill.onnx"
    else:
        onnx_output_path = f"{dirname}/{filename}.onnx"
    OnnxSaver.create_onnx_model_with_pytorch_layer_names(onnx_output_path, model, (in0, states),
        False, None, {'input_names': input_names, 'output_names': output_names, 'opset_version': 17})
    shape_inference.infer_shapes_path(onnx_output_path)
    print(f"onnx model saved to {onnx_output_path}")

    print("Converting to QNN model...")
    states_layout = "NONTRIVIAL"
    converter_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qnn-onnx-converter -i {onnx_output_path} --float_bitwidth {parser_args.qnn_float_width} " + " ".join([f'--input_layout "state{3*j+1}_in" "{states_layout}"' for j in range(model.layer_begin, model.layer_end)])

    if parser_args.quant_encodings:
        converter_cmd += f" --quantization_overrides {str(parser_args.quant_encodings)} --float_fallback"

    if model_args.wkv_customop:
        converter_cmd += " --op_package_config hexagon/RwkvWkvOpPackageCPU.xml --op_package_lib hexagon/CPU/RwkvWkvOpPackage/libs/x86_64-linux-clang/libRwkvWkvOpPackage.so:RwkvWkvOpPackageInterfaceProvider"

    if os.name == 'nt':
        converter_cmd = "python " + converter_cmd
    print(converter_cmd)
    os.system(converter_cmd)
    print("Compiling QNN model library...")
    compiling_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qnn-model-lib-generator -c {onnx_output_path.replace('.onnx', '.cpp')} -b {onnx_output_path.replace('.onnx', '.bin')}"
    if os.name == 'nt':
        compiling_cmd = "python " + compiling_cmd
    print(compiling_cmd)
    os.system(compiling_cmd)
