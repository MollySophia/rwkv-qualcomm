from rwkv_src.rwkv_model import RWKV_RNN, make_chunks
import types
import os
import torch
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Convert model')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser.add_argument('--chunks', type=int, default=2, help='Number of chunks')
parser.add_argument('--use_qnn_quant', action='store_true', help='Use QNN quantization')
parser.add_argument('--qnn_float_width', type=int, default=16, help='QNN float width')
parser.add_argument('--act_bitwidth', type=int, default=16, help='Activation bitwidth')
parser.add_argument('--weights_bitwidth', type=int, default=8, help='Weights bitwidth')
parser.add_argument('--ext_embedding', action='store_true', default=False, help='Use external embedding')
parser.add_argument('--calib_data_path', type=Path, help='Path to calibration data')
parser_args = parser.parse_args()

USE_QNN_QUANT = parser_args.use_qnn_quant
ACT_BITWIDTH = parser_args.act_bitwidth
WEIGHTS_BITWIDTH = parser_args.weights_bitwidth

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False
model_args.fp16 = False
model_args.wkv_customop = False
model_args.USE_EMBEDDING = False if parser_args.ext_embedding else True

model_args.MODEL_NAME = str(parser_args.model)

if 'ABC' in model_args.MODEL_NAME or 'MIDI' in model_args.MODEL_NAME or USE_QNN_QUANT == True:
    model_args.RESCALE_LAYER = 0
else:
    model_args.RESCALE_LAYER = 6

model = make_chunks(parser_args.chunks, model_args) if parser_args.chunks > 1 else RWKV_RNN(model_args)

qnn_sdk_root = os.environ["QNN_SDK_ROOT"]
if not qnn_sdk_root:
    print("Please set QNN_SDK_ROOT environment variable to the root of the Qualcomm Neural Processing SDK")
    exit(1)
os.path.exists("onnx") or os.mkdir("onnx")

def quant_override(model):
    def calc_quant_override(model, args):
        encodings_dict = {'activation_encodings': {}, 'param_encodings': {}}
        graph = model.graph
        float_override = [{"bitwidth": parser_args.qnn_float_width, "dtype": "float"}]
        for i in range(len(graph.node)):
            if "matmul_kv" in graph.node[i].name \
                or "mul_time_decay" in graph.node[i].name \
                or "add_time_decay1" in graph.node[i].name \
                or "att_tanh1" in graph.node[i].name:
                for j in graph.node[i].output:
                    encodings_dict['activation_encodings'][j] = float_override
            if "add_time_first" in graph.node[i].name:
                for j in graph.node[i].input:
                    if "state" in j:
                        encodings_dict['activation_encodings'][j] = float_override
                for j in graph.node[i].output:
                    encodings_dict['activation_encodings'][j] = float_override

        return encodings_dict

    args = model[0].args if type(model) == list else model.args
    import onnx
    import json
    if type(model) == list:
        for i in range(len(model)):
            onnx_model = onnx.load("onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}.onnx")
            encodings_dict = calc_quant_override(onnx_model, args)
            with open("onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}/" + "quant_override.json", 'w') as encoding_json:
                json.dump(encodings_dict, encoding_json, sort_keys=True, indent=4)
    else:
        onnx_model = onnx.load("onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx")
        encodings_dict = calc_quant_override(onnx_model, args)
        with open("onnx/" + args.MODEL_NAME.split("/")[-1] + "_quant_override.json", 'w') as encoding_json:
            json.dump(encodings_dict, encoding_json, sort_keys=True, indent=4)

if type(model) == list:
    args = model[0].args
    if not args.USE_EMBEDDING:
        model[0].emb_weight.cpu().numpy().astype(np.float32).tofile("onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk1of{len(model)}.emb")
    args = model[0].args
    fp16 = args.fp16
    states = []
    for i in range(args.n_layer):
        states.append(torch.zeros(1, args.n_embd, dtype=torch.float16 if fp16 else torch.float32))
        states.append(torch.zeros(args.n_head, args.head_size, args.head_size, dtype=torch.float16 if fp16 else torch.float32))
        states.append(torch.zeros(1, args.n_embd, dtype=torch.float16 if fp16 else torch.float32))
    if model[0].device is not torch.device('cpu'):
        states = [i.to(model[0].device) for i in states]

    for i in range(len(model)):
        dirname = "onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}"
        os.path.exists(dirname) or os.mkdir(dirname)
        if i == 0:
            in0 = torch.LongTensor([[1]]) if args.USE_EMBEDDING else model[0].emb_weight[0].view(1, 1, -1)
        else:
            in0 = torch.zeros(1, 1, args.n_embd, dtype=torch.float16 if fp16 else torch.float32)
        if model[0].device is not torch.device('cpu'):
            in0 = in0.to(model[0].device)
        inputs = {'in0': in0, 'state': [states[j] for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]}
        input_names = ['in'] + [f'state{j}_in' for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
        output_names = ['out'] + [f'state{j}_out' for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]

        if args.wkv_customop:
            def onnx_custom_wkv(g, k, v, r, state2, time_first, time_decay, scale):
                out1, out2 = g.op("rwkv::custom_wkv", k, v, r, state2, time_first, time_decay, scale, outputs=2)
                return out1.setType(k.type().with_dtype(torch.float32).with_sizes([args.n_head, 1, args.head_size])),\
                 out2.setType(k.type().with_dtype(torch.float32).with_sizes([args.n_head, args.head_size, args.head_size]))
            from torch.onnx import register_custom_op_symbolic
            register_custom_op_symbolic('rwkv::custom_wkv', onnx_custom_wkv, 9)

        torch.onnx.export(model[i], inputs, dirname + "/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}.onnx", input_names=input_names, output_names=output_names, opset_version=17)
        print(f"onnx model chunk{i} saved to {dirname}" + "/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}.onnx")
    
    quant_override(model)

    print("Converting and compiling QNN models...")
    for i in range(len(model)):
        dirname = "onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}"
        os.path.exists(dirname) or os.mkdir(dirname)
        converter_cmd = f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-onnx-converter -i {dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i+1}of{len(model)}.onnx --float_bw {parser_args.qnn_float_width} " + " ".join([f'--input_layout "state{j}_in" NONTRIVIAL' for j in range(3*model[i].layer_begin, 3*model[i].layer_end)])
        converter_cmd += ' --input_layout "in" NONTRIVIAL'
        if USE_QNN_QUANT:
            converter_cmd += f" --use_per_row_quantization --use_per_channel_quantization --act_bitwidth {ACT_BITWIDTH} --weights_bitwidth {WEIGHTS_BITWIDTH} --bias_bitwidth {WEIGHTS_BITWIDTH} --quantization_overrides {dirname}/quant_override.json --input_list {parser_args.calib_data_path}/input_list_chunk{i}.txt"
        print(converter_cmd)
        os.system(converter_cmd)
        # Set state{id}_in to have the same encoding as state{id}_out
        with open(f"{dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i+1}of{len(model)}.cpp", "r") as f:
            cpp_lines = f.readlines()
        
        for state_id in range(3*model[i].layer_begin, 3*model[i].layer_end):
            for j in range(len(cpp_lines)):
                if f'.name= "state{state_id}_out",' in cpp_lines[j]:
                    addTensor_line_idx = j
                    break
            for j in range(addTensor_line_idx, addTensor_line_idx + 100):
                if 'scaleOffsetEncoding' in cpp_lines[j]:
                    state_out_encoding = cpp_lines[j]
                    break

            for j in range(len(cpp_lines)):
                if f'"state{state_id}_in"' in cpp_lines[j] and 'model.addTensor' in cpp_lines[j]:
                    addTensor_line_idx = j
                    break
            for j in range(addTensor_line_idx, addTensor_line_idx + 100):
                if 'scaleOffsetEncoding' in cpp_lines[j]:
                    cpp_lines[j] = state_out_encoding
                    break

        with open(f"{dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i+1}of{len(model)}.cpp", "w") as f:
            f.writelines(cpp_lines)

        print("Compiling QNN model library...")
        compiling_cmd = f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-model-lib-generator -c {dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i+1}of{len(model)}.cpp -b {dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i+1}of{len(model)}.bin"
        os.system(compiling_cmd)
else:
    args = model.args
    if not args.USE_EMBEDDING:
        model.emb_weight.cpu().numpy().astype(np.float32).tofile("onnx/" + args.MODEL_NAME.split("/")[-1] + ".emb")
    args = model.args
    fp16 = args.fp16
    in0 = [torch.LongTensor([[1]])] if args.USE_EMBEDDING else [model.emb_weight[0].view(1, 1, -1)]
    states = []
    for i in range(model.args.n_layer):
        states.append(torch.zeros(1, model.args.n_embd, dtype=torch.float16 if fp16 else torch.float32))
        states.append(torch.zeros(model.args.n_head, model.args.head_size, model.args.head_size, dtype=torch.float16 if fp16 else torch.float32))
        states.append(torch.zeros(1, model.args.n_embd, dtype=torch.float16 if fp16 else torch.float32))
    if model.device is not torch.device('cpu'):
        states = [tensor.to(model.device) for tensor in states]
    inputs = {'in0': in0, 'state': states}
    input_names = ['in'] + [f'state{i}_in' for i in range(3*model.args.n_layer)]
    output_names = ['logits'] + [f'state{i}_out' for i in range(3*model.args.n_layer)]
    torch.onnx.export(model, inputs, "onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx", input_names=input_names, output_names=output_names, opset_version=17)
    print(f"onnx model saved to onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx")

    quant_override(model)

    print("Converting to QNN model...")
    converter_cmd = f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-onnx-converter -i onnx/{args.MODEL_NAME.split('/')[-1]}.onnx --float_bw {parser_args.qnn_float_width} " + " ".join([f'--input_layout "state{j}_in" NONTRIVIAL' for j in range(3*model.args.n_layer)])
    if USE_QNN_QUANT:
        converter_cmd += f" --use_per_row_quantization --use_per_channel_quantization --act_bitwidth {ACT_BITWIDTH} --weights_bitwidth {WEIGHTS_BITWIDTH} --quantization_overrides onnx/{args.MODEL_NAME.split('/')[-1]}_quant_override.json --input_list {parser_args.calib_data_path}/input_list.txt"
    print(converter_cmd)
    os.system(converter_cmd)
    print("Compiling QNN model library...")
    os.system(f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-model-lib-generator -c onnx/{args.MODEL_NAME.split('/')[-1]}.cpp -b onnx/{args.MODEL_NAME.split('/')[-1]}.bin")
