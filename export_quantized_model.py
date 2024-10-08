# from rwkv_src.modeling_rwkv6 import Rwkv6ForCausalLM
from rwkv_src.rwkv_model import RWKV_RNN, sample_logits
from transformers import (
    AutoConfig, AutoModelForCausalLM,
    AutoTokenizer,
    modeling_utils
)
from transformers.tokenization_utils_base import BatchEncoding
import types
import torch
import torch.nn as nn
import numpy as np
import onnx
from aimet_torch.model_preparer import _prepare_traced_model

from utils.model_utils import get_dummy_input_for_rwkv_causal_llm, get_input_output_names, split_onnx, get_dummy_state_kvcache
from quantizers.base_quantizer import LLMQuantizer
from utils.dataset_builder import DatasetBuilder

import argparse
import os
import re
import subprocess
import json
from pathlib import Path

parser = argparse.ArgumentParser(description='Convert model')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser.add_argument('--linear_param_encodings', type=Path, default=None, help='Path to linear param encodings')
parser.add_argument('--calib_data_path', type=Path, help='Path to calibration data')
parser.add_argument('--weights_bitwidth', type=int, default=8, help='Weights bitwidth')
parser.add_argument('--use_cuda', action='store_true', default=True, help='Use CUDA')
parser.add_argument('--test_generate', action='store_true', default=False, help='Test generate')
parser.add_argument('--num_chunks', type=int, default=2, help='Number of chunks')
args_parser = parser.parse_args()

device = torch.device("cuda") if args_parser.use_cuda and torch.cuda.is_available() else torch.device("cpu")

args = types.SimpleNamespace()
##############################
args.quant_scheme = "tf"
args.activation_bit_width = 16
args.parameter_bit_width = args_parser.weights_bitwidth
args.in_place_quantsim = False
args.config_file = "quantizers/configs/qsim_config_per_channel_with_exceptions.json"
args.num_cands = 20
args.export_dir = "quant_export"
args.output_dir = "quant_export"
args.model_name = str(args_parser.model).replace(".pth", "").split("/")[-1]
args.input_symmetry = "symqt"
args.exceptions_file = "quantizers/configs/rwkv_activation_exceptions.json"
args.act_mse_loss_type = "mse"
args.parameter_encoding_file = str(args_parser.linear_param_encodings) if args_parser.linear_param_encodings else None
args.encoding_path = None
args.do_actmse = False
args.disable_act_quantizers = False
args.fp16 = True
args.do_train = False
args.clip_activation = None
args.load_sim_checkpoint = False
args.save_sim_checkpoint = False
##############################
args.calib_dataset_name = "wikitext"
args.calib_dataset_config_name = "wikitext-2-raw-v1"
args.dataset_cache_dir = "./dataset_cache"
args.calib_dataset_split = None
args.calib_dataset_preprocessor = "gpt2"
args.eval_dataset_name = "wikitext"
args.eval_dataset_config_name = "wikitext-103-raw-v1"
args.eval_dataset_split = "test"
args.eval_dataset_preprocessor = "gptq"
args.num_calibration_batches = 20
args.per_device_calib_batch_size = 1
args.per_device_eval_batch_size = 1
args.block_size = 1024
args.seed = 1234
##############################

qnn_sdk_root = os.environ["QNN_SDK_ROOT"]
if not qnn_sdk_root:
    print("Please set QNN_SDK_ROOT environment variable to the root of the Qualcomm Neural Processing SDK")
    exit(1)

device = torch.device("cuda") if args_parser.use_cuda and torch.cuda.is_available() else torch.device("cpu")
args.device = device

model_args = types.SimpleNamespace()
model_args.USE_CUDA = args_parser.use_cuda
model_args.fp16 = False
model_args.wkv_customop = False
model_args.USE_EMBEDDING = True
model_args.MODEL_NAME = str(args_parser.model)
model_args.RESCALE_LAYER = 0
model_args.eos_token_id = 0
model = RWKV_RNN(model_args)

tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-5-world-1b5", trust_remote_code=True)
tokenizer.model_max_length = 1024

dummy_input = get_dummy_input_for_rwkv_causal_llm(1, 1, device, model_cfg=model.args)

dataset_builder = DatasetBuilder(args)
dataset_builder.make_dataset(tokenizer=tokenizer, args=args, column_name="text", shuffle=True)

quantizer = LLMQuantizer(model, args, model.args)
quantizer.orig_model = model
quantizer.prepare_quantsim(dummy_input, args, dataset_builder.train_dataloader, tokenizer)

def test_generate(model, tokenizer,device='cuda'):
    print("Generating inference using QuantSim model")
    prompt = "\n我们发现，"
    print(prompt, end='')
    input_ids = tokenizer(prompt, return_tensors='pt')
    model = model.to(device)
    state = get_dummy_state_kvcache(1, model.args, device)
    output = None
    for id in input_ids['input_ids'][0]:
        input = [torch.LongTensor([[id]]).to(device)] + state
        output = model(*input)
        state = output[1:]
    for i in range(100):
        token = sample_logits(output[0].flatten().cpu())
        if token == 0:
            break
        print(tokenizer.decode(token), end='', flush=True)
        input = [torch.LongTensor([[token]]).to(device)] + state
        output = model(*input)
        state = output[1:]


if args_parser.test_generate:
    test_generate(quantizer.quant_sim.model, tokenizer=tokenizer,device=args.device)
else:
    input_names, output_names = get_input_output_names(model.args)
    dummy_input = get_dummy_input_for_rwkv_causal_llm(1, 1, device, model_cfg=model.args)
    quantizer.export_quantsim(dummy_input=dummy_input, input_names=input_names, output_names=output_names, opset_version=17)

    model_args = model.args

    print("Post processing ONNX model")
    onnx_path = os.path.join(args.export_dir, "onnx", f"{args.model_name}.onnx")
    model = onnx.load(onnx_path, load_external_data=False)
    graph = model.graph
    nodes  = graph.node
    pattern = r"blocks\.(\d+)"

    for i in range(1, len(graph.input), 3):
        graph.input[i].name = "layer" + str((i-1)//3) + "_state0_in"
        graph.input[i+1].name = "layer" + str((i-1)//3) + "_state1_in"
        graph.input[i+2].name = "layer" + str((i-1)//3) + "_state2_in"

    # state_in_dims = {}
    # for input in graph.input:
    #     if "state0" in input.name:
    #         state_in_dims["state0"] = input.type.tensor_type.shape
    #     elif "state1" in input.name:
    #         state_in_dims["state1"] = input.type.tensor_type.shape
    #     elif "state2" in input.name:
    #         state_in_dims["state2"] = input.type.tensor_type.shape
    # for idx, output in enumerate(graph.output):
    #     if "state0" in output.name:
    #         for i, dim in enumerate(state_in_dims["state0"].dim):
    #             graph.output[idx].type.tensor_type.shape.dim[i].dim_value = dim.dim_value
    #     elif "state1" in output.name:
    #         for i, dim in enumerate(state_in_dims["state1"].dim):
    #             graph.output[idx].type.tensor_type.shape.dim[i].dim_value = dim.dim_value
    #     elif "state2" in output.name:
    #         for i, dim in enumerate(state_in_dims["state2"].dim):
    #             graph.output[idx].type.tensor_type.shape.dim[i].dim_value = dim.dim_value
    #     elif "logits" in output.name:
    #         graph.output[idx].type.tensor_type.shape.dim[0].dim_value = 1
    #         graph.output[idx].type.tensor_type.shape.dim[1].dim_value = 1
    #         graph.output[idx].type.tensor_type.shape.dim[2].dim_value = 65536

    onnx.save_model(model, onnx_path, save_as_external_data=True, all_tensors_to_one_file=True, size_threshold=1024, convert_attribute=False)

    print("Post processing encodings")
    encodings = None
    with open(onnx_path.replace('.onnx', '.encodings'), "+r") as f:
        encodings = json.load(f)

    graph = model.graph
    for i in range(len(graph.node)):
        if "matmul_kv" in graph.node[i].name \
            or "mul_time_decay" in graph.node[i].name \
            or "add_time_first" in graph.node[i].name:
            for j in graph.node[i].input:
                if not ("Constant" in j or "Split" in j):
                    encodings['activation_encodings'][j] = [{"bitwidth": 32, "dtype": "float"}]
            for j in graph.node[i].output:
                encodings['activation_encodings'][j] = [{"bitwidth": 32, "dtype": "float"}]

    for i in range(len(graph.input)):
        if i == 0 and graph.input[i].type.tensor_type.elem_type != 1:
            continue
        encodings['activation_encodings'][graph.input[i].name] = [{"bitwidth": 32, "dtype": "float"}]
    for i in range(len(graph.output)):
        encodings['activation_encodings'][graph.output[i].name] = [{"bitwidth": 32, "dtype": "float"}]

    with open(onnx_path.replace('.onnx', '.encodings'), "w") as f:
        json.dump(encodings, f, indent=4)

    split_onnx(onnx_path, args.model_name, args_parser.num_chunks, args.export_dir, False)

    layers_per_chunk = len(quantizer.quant_sim.model.blocks) // args_parser.num_chunks
    os.path.exists(os.path.join(args.export_dir, f"sample_inputs")) or os.mkdir(os.path.join(args.export_dir, f"sample_inputs"))
    sample_input_path = os.path.join(args.export_dir, f"sample_inputs", args.model_name)
    os.path.exists(sample_input_path) or os.mkdir(sample_input_path)
    # assume the layers are evenly distributed
    for i in range(args_parser.num_chunks):
        input_list_line = " ".join([f"{args.export_dir}/sample_inputs/{args.model_name}/chunk_{i}/input_{j}.bin" for j in range(3*layers_per_chunk+1)])
        os.path.exists(os.path.join(sample_input_path, f"chunk_{i}")) or os.mkdir(os.path.join(sample_input_path, f"chunk_{i}"))
        with open(os.path.join(sample_input_path, f"input_list_chunk_{i}.txt"), 'w') as f:
            f.write(input_list_line)
        if i == 0:
            np.zeros((1, 1), dtype=np.int32).tofile(os.path.join(sample_input_path, f"chunk_{i}", "input_0.bin"))
        else:
            np.zeros((1, 1, model_args.n_embd), dtype=np.float32).tofile(os.path.join(sample_input_path, f"chunk_{i}", "input_0.bin"))
        for j in range(layers_per_chunk):
            np.zeros((1, 1, model_args.n_embd), dtype=np.float32).tofile(os.path.join(sample_input_path, f"chunk_{i}", f"input_{3*j+1}.bin"))
            np.zeros((model_args.n_head, model_args.head_size, model_args.head_size), dtype=np.float32).tofile(os.path.join(sample_input_path, f"chunk_{i}", f"input_{3*j+2}.bin"))
            np.zeros((1, 1, model_args.n_embd), dtype=np.float32).tofile(os.path.join(sample_input_path, f"chunk_{i}", f"input_{3*j+3}.bin"))
        
    for i in range(args_parser.num_chunks):
        onnx_file = os.path.join(args.export_dir, "split_onnx", f"{args.model_name}_chunk{i+1}of{args_parser.num_chunks}.onnx")
        cmd = [f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-onnx-converter"]
        cmd += ["-i", onnx_file]
        cmd += ["--act_bitwidth", "16"]
        cmd += ["--bias_bitwidth", "32"]
        cmd += ["--float_bitwidth", "32"]
        cmd += ["--quantization_overrides", onnx_path.replace('.onnx', '.encodings')]
        cmd += ["--input_list", os.path.join(args_parser.calib_data_path, f"input_list_chunk{i}.txt")]

        for j in range(i*layers_per_chunk, (i+1)*layers_per_chunk):
            for k in range(3):
                cmd += ["--input_layout", f"layer{j}_state{k}_in", "NONTRIVIAL"] 

        print(" ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = proc.communicate()
        print(error.decode())

        cmd = [f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-model-lib-generator"]
        cmd += ["-c", onnx_file.replace('.onnx', '.cpp')]
        cmd += ["-b", onnx_file.replace('.onnx', '.bin')]
        print(" ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = proc.communicate()
        print(error.decode())

