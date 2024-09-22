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
parser.add_argument('--act_bitwidth', type=int, default=16, help='Activation bitwidth')
parser.add_argument('--weights_bitwidth', type=int, default=8, help='Weights bitwidth')
args = parser.parse_args()

USE_QNN_QUANT = args.use_qnn_quant
ACT_BITWIDTH = args.act_bitwidth
WEIGHTS_BITWIDTH = args.weights_bitwidth

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False
model_args.fp16 = False
model_args.wkv_customop = False
model_args.USE_EMBEDDING = True

model_args.MODEL_NAME = str(args.model)

if 'ABC' in model_args.MODEL_NAME or 'MIDI' in model_args.MODEL_NAME or USE_QNN_QUANT == True:
    model_args.RESCALE_LAYER = 0
else:
    model_args.RESCALE_LAYER = 6

model = make_chunks(args.chunks, model_args) if args.chunks > 1 else RWKV_RNN(model_args)

qnn_sdk_root = os.environ["QNN_SDK_ROOT"]
if not qnn_sdk_root:
    print("Please set QNN_SDK_ROOT environment variable to the root of the Qualcomm Neural Processing SDK")
    exit(1)
os.path.exists("onnx") or os.mkdir("onnx")

def quant_override(model):
    def calc_quant_override(model, args):
        encodings_dict = {'activation_encodings': {}, 'param_encodings': {}}
        graph = model.graph
        for i in range(len(graph.node)):
            if "MatMul" == graph.node[i].op_type:
                if ("MatMul" in graph.node[i].input[0] and ("Add" in graph.node[i].input[1])) or ("Reshape" in graph.node[i].input[0] and ("MatMul" in graph.node[i].input[1] or "Reshape" in graph.node[i].input[1])):
                    for j in graph.node[i].input:
                        if not "Split" in j:
                            if "Constant" in j:
                                encodings_dict['param_encodings'][j] = [{"bitwidth": WEIGHTS_BITWIDTH, "dtype": "int"}]
                            else:
                                encodings_dict['activation_encodings'][j] = [{"bitwidth": 32, "dtype": "float"}]
                    for j in graph.node[i].output:
                        encodings_dict['activation_encodings'][j] = [{"bitwidth": 32, "dtype": "float"}]

            if "Mul" == graph.node[i].op_type and "Exp" in graph.node[i].input[0]:
                for j in graph.node[i].output:
                    encodings_dict['activation_encodings'][j] = [{"bitwidth": 32, "dtype": "float"}]

            if "Add" == graph.node[i].op_type:
                if ("Mul" in graph.node[i].input[1] or "Reshape" in graph.node[i].input[1]) and ("Add" in graph.node[i].input[0]):
                    encodings_dict['activation_encodings'][graph.node[i].input[0]] = [{"bitwidth": 32, "dtype": "float"}]
                    for j in graph.node[i].output:
                        encodings_dict['activation_encodings'][j] = [{"bitwidth": 32, "dtype": "float"}]
                if "Mul_" in graph.node[i].input[0] and "MatMul" in graph.node[i].input[1]:
                    for j in graph.node[i].output:
                        encodings_dict['activation_encodings'][j] = [{"bitwidth": 32, "dtype": "float"}]

        for i in range(len(graph.input)):
            if i == 0 and args.USE_EMBEDDING and graph.input[i].type.tensor_type.elem_type != 1:
                continue
            encodings_dict['activation_encodings'][graph.input[i].name] = [{"bitwidth": 32, "dtype": "float"}]
        for i in range(len(graph.output)):
            encodings_dict['activation_encodings'][graph.output[i].name] = [{"bitwidth": 32, "dtype": "float"}]

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
        encodings_dict = calc_quant_override(onnx_model)
        with open("onnx/" + args.MODEL_NAME.split("/")[-1] + "_quant_override.json", 'w') as encoding_json:
            json.dump(encodings_dict, encoding_json, sort_keys=True, indent=4)

if type(model) == list:
    args = model[0].args
    if not args.USE_EMBEDDING:
        model[0].w.emb.weight.cpu().numpy().astype(np.float32).tofile("onnx/" + args.MODEL_NAME.split("/")[-1] + ".emb")
    args = model[0].args
    fp16 = args.fp16
    states = []
    for i in range(args.n_layer):
        states.append(torch.zeros(1, args.n_embd, dtype=torch.float16 if fp16 else torch.float32))
        states.append(torch.zeros(args.n_head, args.head_size, args.head_size, dtype=torch.float16 if fp16 else torch.float32))
        states.append(torch.zeros(1, args.n_embd, dtype=torch.float16 if fp16 else torch.float32))
    if model[0].INFERENCE_DEVICE is not torch.device('cpu'):
        states = [i.to(model[0].INFERENCE_DEVICE) for i in states]

    for i in range(len(model)):
        dirname = "onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}"
        os.path.exists(dirname) or os.mkdir(dirname)
        if i == 0:
            in0 = torch.LongTensor([1]) if args.USE_EMBEDDING else model[0].w.emb.weight[0]
        else:
            in0 = torch.zeros(args.n_embd, dtype=torch.float16 if fp16 else torch.float32)
        if model[0].INFERENCE_DEVICE is not torch.device('cpu'):
            in0 = in0.to(model[0].INFERENCE_DEVICE)
        inputs = [in0] + [states[j] for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
        input_names = ['in'] + [f'state{j}_in' for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
        output_names = ['out'] + [f'state{j}_out' for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]

        if args.wkv_customop:
            def onnx_custom_wkv(g, k, v, r, state2, time_first, time_decay, scale):
                out1, out2 = g.op("rwkv::custom_wkv", k, v, r, state2, time_first, time_decay, scale, outputs=2)
                return out1.setType(k.type().with_dtype(torch.float32).with_sizes([args.n_head, 1, args.head_size])),\
                 out2.setType(k.type().with_dtype(torch.float32).with_sizes([args.n_head, args.head_size, args.head_size]))
            from torch.onnx import register_custom_op_symbolic
            register_custom_op_symbolic('rwkv::custom_wkv', onnx_custom_wkv, 9)

        torch.onnx.export(model[i], tuple(inputs), dirname + "/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}.onnx", input_names=input_names, output_names=output_names, opset_version=17)
        print(f"onnx model chunk{i} saved to {dirname}" + "/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}.onnx")
    
    quant_override(model)

    print("Converting and compiling QNN models...")
    for i in range(len(model)):
        dirname = "onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}"
        os.path.exists(dirname) or os.mkdir(dirname)
        converter_cmd = f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-onnx-converter -i {dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i+1}of{len(model)}.onnx --float_bw 32 " + " ".join([f'--input_layout "state{3*j+1}_in" NONTRIVIAL' for j in range(model[i].layer_begin, model[i].layer_end)])
        if USE_QNN_QUANT:
            converter_cmd += f" --use_per_row_quantization --use_per_channel_quantization --act_bitwidth {ACT_BITWIDTH} --weights_bitwidth {WEIGHTS_BITWIDTH} --bias_bitwidth {WEIGHTS_BITWIDTH} --quantization_overrides {dirname}/quant_override.json --input_list input_list_chunk{i}.txt"
        print(converter_cmd)
        os.system(converter_cmd)
        print("Compiling QNN model library...")
        compiling_cmd = f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-model-lib-generator -c {dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i+1}of{len(model)}.cpp -b {dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i+1}of{len(model)}.bin"
        os.system(compiling_cmd)
else:
    args = model.args
    if not args.USE_EMBEDDING:
        model.w.emb.weight.cpu().numpy().astype(np.float32).tofile("onnx/" + args.MODEL_NAME.split("/")[-1] + ".emb")
    args = model.args
    fp16 = args.fp16
    inputs = [torch.LongTensor([1])] if args.USE_EMBEDDING else [model.w.emb.weight[0]]
    for i in range(model.args.n_layer):
        inputs.append(torch.zeros(1, model.args.n_embd, dtype=torch.float16 if fp16 else torch.float32))
        inputs.append(torch.zeros(model.args.n_head, model.args.head_size, model.args.head_size, dtype=torch.float16 if fp16 else torch.float32))
        inputs.append(torch.zeros(1, model.args.n_embd, dtype=torch.float16 if fp16 else torch.float32))
    if model.INFERENCE_DEVICE is not torch.device('cpu'):
        inputs = [tensor.to(model.INFERENCE_DEVICE) for tensor in inputs]
    input_names = ['id'] + [f'state{i}_in' for i in range(3*model.args.n_layer)]
    output_names = ['logits'] + [f'state{i}_out' for i in range(3*model.args.n_layer)]
    torch.onnx.export(model, tuple(inputs), "onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx", input_names=input_names, output_names=output_names, opset_version=17)
    print(f"onnx model saved to onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx")

    quant_override(model)

    print("Converting to QNN model...")
    converter_cmd = f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-onnx-converter -i onnx/{args.MODEL_NAME.split('/')[-1]}.onnx --float_bw 32 " + " ".join([f'--input_layout "state{3*j+1}_in" NONTRIVIAL' for j in range(model.args.n_layer)])
    if USE_QNN_QUANT:
        converter_cmd += f" --use_per_row_quantization --use_per_channel_quantization --act_bitwidth {ACT_BITWIDTH} --weights_bitwidth {WEIGHTS_BITWIDTH} --quantization_overrides onnx/{args.MODEL_NAME.split('/')[-1]}_quant_override.json --input_list input_list.txt"
    print(converter_cmd)
    os.system(converter_cmd)
    print("Compiling QNN model library...")
    os.system(f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-model-lib-generator -c onnx/{args.MODEL_NAME.split('/')[-1]}.cpp -b onnx/{args.MODEL_NAME.split('/')[-1]}.bin")
