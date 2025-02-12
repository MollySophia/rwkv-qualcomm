from rwkv_src.rwkv_model import RWKV_RNN, make_chunks
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

parser = argparse.ArgumentParser(description='Convert model')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser.add_argument('--chunks', type=int, default=2, help='Number of chunks')
parser.add_argument('--use_qnn_quant', action='store_true', help='Use QNN quantization')
parser.add_argument('--qnn_float_width', type=int, default=32, help='QNN float width')
parser.add_argument('--act_bitwidth', type=int, default=16, help='Activation bitwidth')
parser.add_argument('--ext_embedding', action='store_true', default=False, help='Use external embedding')
parser.add_argument('--calib_data_path', type=Path, help='Path to calibration data')
parser.add_argument('--linear_param_encodings', type=Path, default=None, help='Path to linear param encodings')
parser.add_argument('--prefill_model', action='store_true', help='Convert model for sequential prefill')
parser.add_argument('--wkv_customop', action='store_true', help='Use custom op for wkv')
parser_args = parser.parse_args()

seq_length = 32 if parser_args.prefill_model else 1

# TODO: add more while keeping the precision
if parser_args.linear_param_encodings:
    if "7B" in str(parser_args.model):
        quant_list = ["att.key", "att.value", "att.receptance", "att.gate", "att.output", "ffn", "head"]
    else:
        quant_list = ["att.output", "ffn"]
    with open(parser_args.linear_param_encodings, "r") as f:
        encodings = json.load(f)
    linear_encodings_new = copy.deepcopy(encodings)
    for k, v in encodings['param_encodings'].items():
        if not any([x in k for x in quant_list]):
            del linear_encodings_new['param_encodings'][k]
    del encodings

USE_QNN_QUANT = parser_args.use_qnn_quant
ACT_BITWIDTH = parser_args.act_bitwidth

# do not set this to 4 directly; it will quantize all linear weights to 4 bits naively
# instead, set it to 8 and use linear_param_encodings to quantize only the weights that need to be quantized to 4 bits
WEIGHTS_BITWIDTH = 8

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False
model_args.fp16 = False
model_args.wkv_customop = parser_args.wkv_customop
model_args.USE_EMBEDDING = False if parser_args.ext_embedding else True

model_args.MODEL_NAME = str(parser_args.model)

if 'ABC' in model_args.MODEL_NAME or 'MIDI' in model_args.MODEL_NAME or USE_QNN_QUANT == True or 'x070' in model_args.MODEL_NAME:
    model_args.RESCALE_LAYER = 0
else:
    model_args.RESCALE_LAYER = 6

model = make_chunks(parser_args.chunks, model_args) if parser_args.chunks > 1 else RWKV_RNN(model_args)

if parser_args.prefill_model:
    model_args.MODEL_NAME = model_args.MODEL_NAME + "_prefill"

qnn_sdk_root = os.environ["QNN_SDK_ROOT"]
if not qnn_sdk_root:
    print("Please set QNN_SDK_ROOT environment variable to the root of the Qualcomm Neural Processing SDK")
    exit(1)
os.path.exists("onnx") or os.mkdir("onnx")
if os.name == 'nt':
    qnn_tools_target = 'x86_64-windows-msvc'
else:
    qnn_tools_target = 'x86_64-linux-clang'

def quant_override(model):
    def calc_quant_override_v6(model, layer_begin):
        encodings_dict = {'activation_encodings': {}, 'param_encodings': {}}
        graph = model.graph
        float_override = [{"bitwidth": parser_args.qnn_float_width, "dtype": "float"}]
        act_int_override = [{"bitwidth": 16, "dtype": "int"}]
        for i in range(len(graph.node)):
            if "matmul_kv" in graph.node[i].name \
                or "mul_time_decay" in graph.node[i].name \
                or "add_time_decay1" in graph.node[i].name \
                or "ln" in graph.node[i].name \
                or "wkv" in graph.node[i].name:
                for j in graph.node[i].output:
                    encodings_dict['activation_encodings'][j] = float_override
                if "wkv" in graph.node[i].name:
                    for j in graph.node[i].input:
                        encodings_dict['activation_encodings'][j] = float_override
            if "add_time_first" in graph.node[i].name:
                for j in graph.node[i].input:
                    if "state" in j:
                        encodings_dict['activation_encodings'][j] = float_override
                for j in graph.node[i].output:
                    encodings_dict['activation_encodings'][j] = float_override

            # a16w8 head
            if "head" in graph.node[i].name:
                for j in graph.node[i].output:
                    encodings_dict['activation_encodings'][j] = act_int_override

            if "emb" in graph.node[i].name or "Gather" in graph.node[i].name:
                for j in graph.node[i].output:
                    encodings_dict['activation_encodings'][j] = act_int_override

        for i in range(len(graph.input)):
            if (graph.input[i].type.tensor_type.elem_type == 1):
                encodings_dict['activation_encodings'][graph.input[i].name] = float_override
        for i in range(len(graph.output)):
            if not graph.output[i].name in encodings_dict['activation_encodings']:
                encodings_dict['activation_encodings'][graph.output[i].name] = float_override

        if parser_args.linear_param_encodings:
            for k, v in linear_encodings_new['param_encodings'].items():
                if not "ln" in k and not "head" in k:
                    k = k.replace(".", "/").replace("/weight", "").replace("blocks/", "/blocks.")
                    encoding_block_id = int(k.split(".")[-1].split("/")[0])
                    if encoding_block_id >= layer_begin:
                        for i in range(len(graph.node)):
                            if k.replace(f"blocks.{encoding_block_id}", f"blocks.{encoding_block_id - layer_begin}") in graph.node[i].name:
                                print("Setting encoding for", k)
                                print("onnx weight name:", graph.node[i].input[1])
                                encodings_dict["param_encodings"][graph.node[i].input[1]] = v
                elif "head" in k:
                    k = k.replace(".", "/").replace("/weight", "").replace("blocks/", "/blocks.")
                    for i in range(len(graph.node)):
                        if "head" in graph.node[i].name:
                            print("Setting encoding for", k)
                            print("onnx weight name:", graph.node[i].input[1])
                            encodings_dict["param_encodings"][graph.node[i].input[1]] = v

        return encodings_dict

    def calc_quant_override_v7(model, layer_begin):
        encodings_dict = {'activation_encodings': {}, 'param_encodings': {}}
        graph = model.graph
        float_override = [{"bitwidth": parser_args.qnn_float_width, "dtype": "float"}]
        act_int_override = [{"bitwidth": 16, "dtype": "int"}]
        for i in range(len(graph.node)):
            if "matmul_kv" in graph.node[i].name \
                or "mul_time_decay" in graph.node[i].name \
                or "matmul_ab" in graph.node[i].name \
                or ("ln" in graph.node[i].name and not "ln_x" in graph.node[i].name):
                for j in graph.node[i].output:
                    encodings_dict['activation_encodings'][j] = float_override
                if "ln" in graph.node[i].name and not "ln_x" in graph.node[i].name:
                    for j in graph.node[i].input:
                        encodings_dict['activation_encodings'][j] = float_override

            # a16w8 head
            if "head" in graph.node[i].name:
                for j in graph.node[i].output:
                    encodings_dict['activation_encodings'][j] = act_int_override

        for i in range(len(graph.input)):
            if (graph.input[i].type.tensor_type.elem_type == 1):
                encodings_dict['activation_encodings'][graph.input[i].name] = float_override
        for i in range(len(graph.output)):
            if not graph.output[i].name in encodings_dict['activation_encodings']:
                encodings_dict['activation_encodings'][graph.output[i].name] = float_override

        if parser_args.linear_param_encodings:
            for k, v in linear_encodings_new['param_encodings'].items():
                if not "ln" in k and not "head" in k:
                    k = k.replace(".", "/").replace("/weight", "").replace("blocks/", "/blocks.")
                    encoding_block_id = int(k.split(".")[-1].split("/")[0])
                    if encoding_block_id >= layer_begin:
                        for i in range(len(graph.node)):
                            if k.replace(f"blocks.{encoding_block_id}", f"blocks.{encoding_block_id - layer_begin}") in graph.node[i].name:
                                print("Setting encoding for", k)
                                print("onnx weight name:", graph.node[i].input[1])
                                encodings_dict["param_encodings"][graph.node[i].input[1]] = v
                elif "head" in k:
                    k = k.replace(".", "/").replace("/weight", "").replace("blocks/", "/blocks.")
                    for i in range(len(graph.node)):
                        if "head" in graph.node[i].name:
                            print("Setting encoding for", k)
                            print("onnx weight name:", graph.node[i].input[1])
                            encodings_dict["param_encodings"][graph.node[i].input[1]] = v

        return encodings_dict

    args = model[0].args if type(model) == list else model.args
    import onnx
    import json
    if type(model) == list:
        for i in range(len(model)):
            onnx_model = onnx.load("onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}.onnx")
            if args.version == 7:
                encodings_dict = calc_quant_override_v7(onnx_model, model[i].layer_begin)
            else:
                encodings_dict = calc_quant_override_v6(onnx_model, model[i].layer_begin)
            with open("onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}/" + "quant_override.json", 'w') as encoding_json:
                json.dump(encodings_dict, encoding_json, sort_keys=True, indent=4)
    else:
        onnx_model = onnx.load("onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx")
        if args.version == 7:
            encodings_dict = calc_quant_override_v7(onnx_model, 0)
        else:
            encodings_dict = calc_quant_override_v6(onnx_model, 0)
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
        if i == 0 and args.USE_EMBEDDING:
            in0 = torch.LongTensor([[1]*seq_length])
        else:
            in0 = torch.zeros(1, seq_length, args.n_embd, dtype=torch.float16 if fp16 else torch.float32)

        if model[0].device is not torch.device('cpu'):
            in0 = in0.to(model[0].device)
        inputs = {'in0': in0, 'state': [states[j] for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]}
        input_names = ['in'] + [f'state{j}_in' for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
        output_names = ['out'] + [f'state{j}_out' for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]

        if model[0].args.version == 7:
            if i != len(model) - 1:
                output_names += ['v_first_out']

            if i != 0:
                input_names += ['v_first_in']
                inputs['v_first'] = torch.zeros(seq_length, args.n_embd, dtype=torch.float16 if fp16 else torch.float32)

        if args.wkv_customop:
            from torch.onnx.symbolic_helper import _get_tensor_sizes
            from torch.onnx import register_custom_op_symbolic
            op_name = "rwkv::wkv_chunk" if parser_args.prefill_model else "rwkv::wkv"
            def onnx_custom_wkv(g, k, v, r, state2, time_first, time_decay):
                out1, out2 = g.op(op_name, k, v, r, state2, time_first, time_decay, outputs=2)
                return out1.setType(k.type().with_dtype(torch.float32).with_sizes([seq_length, _get_tensor_sizes(k)[0], 1, args.head_size])),\
                 out2.setType(k.type().with_dtype(torch.float32).with_sizes([1, _get_tensor_sizes(k)[0], args.head_size, args.head_size]))
            register_custom_op_symbolic(op_name, onnx_custom_wkv, 9)

        torch.onnx.export(model[i], inputs, dirname + "/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}.onnx", input_names=input_names, output_names=output_names, opset_version=17)
        shape_inference.infer_shapes_path(dirname + "/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}.onnx")
        onnx_model = onnx.load(dirname + "/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}.onnx")

        # To make qnn-onnx-converter happy when using custom op
        for initializer in onnx_model.graph.initializer:
            shape = list(initializer.dims)
            value_info = onnx.helper.make_tensor_value_info(initializer.name, initializer.data_type, shape)
            onnx_model.graph.value_info.append(value_info)
        onnx.save_model(onnx_model, dirname + "/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}.onnx", save_as_external_data=True, all_tensors_to_one_file=True)
        print(f"onnx model chunk{i} saved to {dirname}" + "/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}.onnx")

    quant_override(model)

    print("Converting and compiling QNN models...")
    for i in range(len(model)):
        dirname = "onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i+1}of{len(model)}"
        os.path.exists(dirname) or os.mkdir(dirname)
        if os.name == 'nt':
            states_layout = "NONTRIVIAL"
        else:
            states_layout = "NFC"
        converter_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qnn-onnx-converter -i {dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i+1}of{len(model)}.onnx --float_bw {parser_args.qnn_float_width} " + " ".join([f'--input_layout "state{3*j+1}_in" "{states_layout}"' for j in range(model[i].layer_begin, model[i].layer_end)])
        converter_cmd += ' --input_layout "in" "NFC"'
        if USE_QNN_QUANT:
            converter_cmd += f" --use_per_row_quantization --use_per_channel_quantization --act_bitwidth {ACT_BITWIDTH} --weights_bitwidth {WEIGHTS_BITWIDTH} --quantization_overrides {dirname}/quant_override.json --input_list {parser_args.calib_data_path}/input_list_chunk{i}.txt"
        if model_args.wkv_customop:
            converter_cmd += " --op_package_config hexagon/RwkvWkvOpPackageCPU.xml --op_package_lib hexagon/CPU/RwkvWkvOpPackage/libs/x86_64-linux-clang/libRwkvWkvOpPackage.so:RwkvWkvOpPackageInterfaceProvider"
        print(converter_cmd)
        if os.name == 'nt':
            converter_cmd = "python " + converter_cmd
        os.system(converter_cmd)
        # Set state{id}_in to have the same encoding as state{id}_out
        # with open(f"{dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i+1}of{len(model)}.cpp", "r") as f:
        #     cpp_lines = f.readlines()

        # for state_id in range(3*model[i].layer_begin, 3*model[i].layer_end):
        #     for j in range(len(cpp_lines)):
        #         if f'.name= "state{state_id}_out",' in cpp_lines[j]:
        #             addTensor_line_idx = j
        #             break
        #     for j in range(addTensor_line_idx, addTensor_line_idx + 100):
        #         if 'scaleOffsetEncoding' in cpp_lines[j]:
        #             state_out_encoding = cpp_lines[j]
        #             break

        #     for j in range(len(cpp_lines)):
        #         if f'"state{state_id}_in"' in cpp_lines[j] and 'model.addTensor' in cpp_lines[j]:
        #             addTensor_line_idx = j
        #             break
        #     for j in range(addTensor_line_idx, addTensor_line_idx + 100):
        #         if 'scaleOffsetEncoding' in cpp_lines[j]:
        #             cpp_lines[j] = state_out_encoding
        #             break

        # with open(f"{dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i+1}of{len(model)}.cpp", "w") as f:
        #     f.writelines(cpp_lines)

        print("Compiling QNN model library...")
        compiling_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qnn-model-lib-generator -c {os.getcwd()}/{dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i+1}of{len(model)}.cpp -b {os.getcwd()}/{dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i+1}of{len(model)}.bin"
        if os.name == 'nt':
            compiling_cmd = "python " + compiling_cmd
        os.system(compiling_cmd)
else:
    args = model.args
    if not args.USE_EMBEDDING:
        model.emb_weight.cpu().numpy().astype(np.float32).tofile("onnx/" + args.MODEL_NAME.split("/")[-1] + ".emb")
    args = model.args
    fp16 = args.fp16
    in0 = torch.LongTensor([[1]*seq_length]) if args.USE_EMBEDDING else [torch.zeros(1, seq_length, args.n_embd, dtype=torch.float16 if fp16 else torch.float32)]
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
    shape_inference.infer_shapes_path("onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx")
    onnx_model = onnx.load("onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx")

    # To make qnn-onnx-converter happy when using custom op
    for initializer in onnx_model.graph.initializer:
        shape = list(initializer.dims)
        value_info = onnx.helper.make_tensor_value_info(initializer.name, initializer.data_type, shape)
        onnx_model.graph.value_info.append(value_info)
    onnx.save_model(onnx_model, "onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx", save_as_external_data=True, all_tensors_to_one_file=True)
    print(f"onnx model saved to onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx")

    quant_override(model)

    print("Converting to QNN model...")
    if os.name == 'nt':
        states_layout = "NONTRIVIAL"
    else:
        states_layout = "NFC"
    converter_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qnn-onnx-converter -i onnx/{args.MODEL_NAME.split('/')[-1]}.onnx --float_bw {parser_args.qnn_float_width} " + " ".join([f'--input_layout "state{3*j+1}_in" "{states_layout}"' for j in range(model.layer_begin, model.layer_end)])
    converter_cmd += ' --input_layout "in" "NFC"'
    if USE_QNN_QUANT:
        converter_cmd += f" --use_per_row_quantization --use_per_channel_quantization --act_bitwidth {ACT_BITWIDTH} --weights_bitwidth {WEIGHTS_BITWIDTH} --quantization_overrides onnx/{args.MODEL_NAME.split('/')[-1]}_quant_override.json --input_list {parser_args.calib_data_path}/input_list.txt"
    if model_args.wkv_customop:
        converter_cmd += " --op_package_config hexagon/RwkvWkvOpPackageCPU.xml --op_package_lib hexagon/CPU/RwkvWkvOpPackage/libs/x86_64-linux-clang/libRwkvWkvOpPackage.so:RwkvWkvOpPackageInterfaceProvider"
    if os.name == 'nt':
        converter_cmd = "python " + converter_cmd
    print(converter_cmd)
    os.system(converter_cmd)
    print("Compiling QNN model library...")
    compiling_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qnn-model-lib-generator -c {os.getcwd()}/onnx/{args.MODEL_NAME.split('/')[-1]}.cpp -b {os.getcwd()}/onnx/{args.MODEL_NAME.split('/')[-1]}.bin"
    if os.name == 'nt':
        compiling_cmd = "python " + compiling_cmd
    print(compiling_cmd)
    os.system(compiling_cmd)
