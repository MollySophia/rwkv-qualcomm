from rwkv_src.rwkv_tokenizer import RWKV_TOKENIZER, ABCTokenizer
from rwkv_src.rwkv_model import RWKV_RNN, sample_logits, make_chunks, run_prompt
import types
import os, sys
import torch
import numpy as np

USE_SNPE_DLC = False
USE_QNN_QUANT = False
ACT_BITWIDTH = 16
WEIGHTS_BITWIDTH = 8

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False
model_args.USE_XPU = False
model_args.fp16 = False

model_args.USE_EMBEDDING = True
model_dir = '/home/molly/workspace/models/'
# model_args.MODEL_NAME = model_dir + 'RWKV-6-ABC-85M-v1-20240217-ctx1024'
# model_args.MODEL_NAME = model_dir + 'RWKV-6-MIDI-120M-20240220-ctx4096'
# model_args.MODEL_NAME = model_dir + 'RWKV-x060-World-3B-v2.1-20240417-ctx4096'
model_args.MODEL_NAME = model_dir + 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096'
# model_args.MODEL_NAME = model_dir + 'RWKV-x060-World-7B-v2.1-20240507-ctx4096'
# model_args.MODEL_NAME = model_dir + 'RWKV-5-ABC-82M-v1-20230901-ctx1024'
# model_args.MODEL_NAME = model_dir + 'RWKV-5-MIDI-120M-v1-20230728-ctx4096'
# model_args.MODEL_NAME = model_dir + 'RWKV-5-World-0.4B-v2-20231113-ctx4096'
# model_args.MODEL_NAME = model_dir + 'RWKV-5-World-1B5-v2-20231025-ctx4096'
# model_args.MODEL_NAME = model_dir + 'RWKV-5-World-3B-v2-20231118-ctx16k'

if 'ABC' in model_args.MODEL_NAME:
    model_args.RESCALE_LAYER = 0
    tokenizer = ABCTokenizer()
    prompt = """S:3
B:9
E:4
B:9
E:4
E:4
B:9
L:1/8
M:3/4
K:D
 Bc | d2 cB A2 FE | F2 B4 F^G |
"""
    prompt = chr(tokenizer.bos_token_id) + prompt
elif 'MIDI' in model_args.MODEL_NAME:
    model_args.RESCALE_LAYER = 0
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file("./tokenizer-midi.json")
    prompt = "<pad>"
else:
    if USE_QNN_QUANT == True:
        model_args.RESCALE_LAYER = 0
    else:
        model_args.RESCALE_LAYER = 6
    tokenizer = RWKV_TOKENIZER("./rwkv_vocab_v20230424.txt")
    prompt = "\n我们发现"

# model = RWKV_RNN(model_args)
# model = make_chunks(2, model_args)
model = make_chunks(4, model_args)

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
                if ("Reshape" in graph.node[i].input[0] and ("Add" in graph.node[i].input[1])):
                    for j in graph.node[i].input:
                        if not "Split" in j:
                            if "Constant" in j:
                                encodings_dict['param_encodings'][j] = [{"bitwidth": WEIGHTS_BITWIDTH, "dtype": "int"}]
                            else:
                                encodings_dict['activation_encodings'][j] = [{"bitwidth": 32, "dtype": "float"}]
                    for j in graph.node[i].output:
                        encodings_dict['activation_encodings'][j] = [{"bitwidth": 32, "dtype": "float"}]
                # else:
                #     for j in graph.node[i].input:
                #         if "Constant" in j:
                #             encodings_dict['param_encodings'][j] = [{"bitwidth": 4, "dtype": "int"}]
            if "Add" == graph.node[i].op_type:
                if "Mul" in graph.node[i].input[1] and "Add" in graph.node[i].input[0]:
                    # for j in graph.node[i].input:
                    encodings_dict['activation_encodings'][graph.node[i].input[0]] = [{"bitwidth": 32, "dtype": "float"}]
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
        for idx in range(len(model)):
            onnx_model = onnx.load("onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{idx}/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{idx}.onnx")
            encodings_dict = calc_quant_override(onnx_model, args)
            with open("onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{idx}/" + "quant_override.json", 'w') as encoding_json:
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
        dirname = "onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i}"
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

        torch.onnx.export(model[i], tuple(inputs), dirname + "/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i}.onnx", input_names=input_names, output_names=output_names, opset_version=15)
        print(f"onnx model chunk{i} saved to {dirname}" + "/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i}.onnx")
    
    quant_override(model)

    config = f"""version: {args.version}
head_size: {args.head_size}
n_layer: {args.n_layer}
n_embd: {args.n_embd}
n_att: {args.n_att}
n_ffn: {args.n_ffn}
vocab_size: {args.vocab_size}
"""
    with open("onnx/" + args.MODEL_NAME.split("/")[-1] + ".config", "w") as f:
        f.write(config)
    
    print("Converting and compiling QNN models...")
    for i in range(len(model)):
        dirname = "onnx/" + args.MODEL_NAME.split("/")[-1] + f"_chunk{i}"
        os.path.exists(dirname) or os.mkdir(dirname)
        if USE_SNPE_DLC:
            converter_cmd = f"{qnn_sdk_root}/bin/x86_64-linux-clang/snpe-onnx-to-dlc -i {dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i}.onnx --no_simplification " + " ".join([f'--input_layout "state{3*j+1}_in" NONTRIVIAL' for j in range(model[i].layer_begin, model[i].layer_end)])
            if USE_QNN_QUANT:
                converter_cmd += f" --quantization_override {dirname}/quant_override.json"
            print(converter_cmd)
            os.system(converter_cmd)

        else:
            converter_cmd = f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-onnx-converter -i {dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i}.onnx --float_bw 32 --no_simplification " + " ".join([f'--input_layout "state{3*j+1}_in" NONTRIVIAL' for j in range(model[i].layer_begin, model[i].layer_end)])
            if USE_QNN_QUANT:
                converter_cmd += f" --use_per_row_quantization --act_bitwidth {ACT_BITWIDTH} --weights_bitwidth {WEIGHTS_BITWIDTH} --bias_bitwidth {WEIGHTS_BITWIDTH} --quantization_overrides {dirname}/quant_override.json --input_list input_list_chunk{i}.txt"
                if WEIGHTS_BITWIDTH == 4:
                    converter_cmd += " --pack_4_bit_weights"
            print(converter_cmd)
            os.system(converter_cmd)
            print("Compiling QNN model library...")
            os.system(f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-model-lib-generator -c {dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i}.cpp -b {dirname}/{args.MODEL_NAME.split('/')[-1]}_chunk{i}.bin")
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
    # torch.jit.trace(model, tuple(inputs)).save("onnx/" + args.MODEL_NAME.split("/")[-1] + ".pt")
    torch.onnx.export(model, tuple(inputs), "onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx", input_names=input_names, output_names=output_names, opset_version=15)
    print(f"onnx model saved to onnx/" + args.MODEL_NAME.split("/")[-1] + ".onnx")
    config = f"""version: {args.version}
head_size: {args.head_size}
n_layer: {args.n_layer}
n_embd: {args.n_embd}
n_att: {args.n_att}
n_ffn: {args.n_ffn}
vocab_size: {args.vocab_size}
"""
    with open("onnx/" + args.MODEL_NAME.split("/")[-1] + ".config", "w") as f:
        f.write(config)

    quant_override(model)

    print("Converting to QNN model...")
    if USE_SNPE_DLC:
        converter_cmd = f"{qnn_sdk_root}/bin/x86_64-linux-clang/snpe-onnx-to-dlc -i onnx/{args.MODEL_NAME.split('/')[-1]}.onnx --no_simplification " + " ".join([f'--input_layout "state{3*j+1}_in" NONTRIVIAL' for j in range(model.args.n_layer)])
        # converter_cmd = f"{qnn_sdk_root}/bin/x86_64-linux-clang/qairt-converter -i onnx/{args.MODEL_NAME.split('/')[-1]}.onnx --onnx_no_simplification " + " ".join([f'--source_model_input_layout "state{3*j+1}_in" NONTRIVIAL' for j in range(model.args.n_layer)])
        if USE_QNN_QUANT:
            converter_cmd += f" --quantization_overrides onnx/{args.MODEL_NAME.split('/')[-1]}_quant_override.json"
        print(converter_cmd)
        os.system(converter_cmd)
    else:
        converter_cmd = f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-onnx-converter -i onnx/{args.MODEL_NAME.split('/')[-1]}.onnx --float_bw 32 " + " ".join([f'--input_layout "state{3*j+1}_in" NONTRIVIAL' for j in range(model.args.n_layer)])
        if USE_QNN_QUANT:
            converter_cmd += f" --use_per_row_quantization --act_bitwidth {ACT_BITWIDTH} --weights_bitwidth {WEIGHTS_BITWIDTH} --quantization_overrides onnx/{args.MODEL_NAME.split('/')[-1]}_quant_override.json --input_list input_list.txt"
            if WEIGHTS_BITWIDTH == 4:
                converter_cmd += " --pack_4_bit_weights"
        print(converter_cmd)
        os.system(converter_cmd)
        print("Compiling QNN model library...")
        os.system(f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-model-lib-generator -c onnx/{args.MODEL_NAME.split('/')[-1]}.cpp -b onnx/{args.MODEL_NAME.split('/')[-1]}.bin")
