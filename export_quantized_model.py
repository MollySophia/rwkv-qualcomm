from rwkv_src.modeling_rwkv6 import Rwkv6ForCausalLM
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

from utils.model_utils import get_dummy_input_for_rwkv_causal_llm, get_input_output_names, split_onnx
from quantizers.base_quantizer import LLMQuantizer
from utils.dataset_builder import DatasetBuilder

import argparse
import os
import re
from pathlib import Path

parser = argparse.ArgumentParser(description='Convert model')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser.add_argument('model_name', type=str, help='Model name')
parser.add_argument('linear_param_encodings', type=Path, help='Path to linear param encodings')
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
args.config_file = "quantizers/configs/default_per_channel_config.json"
args.num_cands = 20
args.export_dir = "quant_export"
args.output_dir = "quant_export"
args.model_name = args_parser.model_name
args.input_symmetry = "symqt"
args.exceptions_file = "quantizers/configs/rwkv_activation_exceptions.json"
args.act_mse_loss_type = "mse"
args.parameter_encoding_file = str(args_parser.linear_param_encodings)
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
args.num_calibration_batches = 1
args.per_device_calib_batch_size = 2
args.per_device_eval_batch_size = 4
args.block_size = 1024
args.seed = 1234
##############################

qnn_sdk_root = os.environ["QNN_SDK_ROOT"]
if not qnn_sdk_root:
    print("Please set QNN_SDK_ROOT environment variable to the root of the Qualcomm Neural Processing SDK")
    exit(1)

device = torch.device("cuda") if args_parser.use_cuda and torch.cuda.is_available() else torch.device("cpu")
args.device = device

config = AutoConfig.from_pretrained(str(args_parser.model), trust_remote_code=True)
config.vocab_size = 65536
config.model_max_length = 1024
config.return_top_k = 0
config.use_position_embedding_input = False
config.use_combined_mask_input = False
config.num_logits_to_return = 0
config.shift_cache = False

model = Rwkv6ForCausalLM.from_pretrained(str(args_parser.model), config=config)
model = model.to(device)
model.rwkv.embeddings.weight = \
    nn.parameter.Parameter(data=nn.functional.layer_norm(model.rwkv.embeddings.weight, [model.config.hidden_size], \
                                weight=model.rwkv.blocks[0].pre_ln.weight, bias=model.rwkv.blocks[0].pre_ln.bias, eps=1e-5))
del model.rwkv.blocks[0].pre_ln
tokenizer = AutoTokenizer.from_pretrained(str(args_parser.model), trust_remote_code=True)
tokenizer.model_max_length = 1024

dummy_input = get_dummy_input_for_rwkv_causal_llm(1, 1, device, model_cfg=model.config)

dataset_builder = DatasetBuilder(args)
dataset_builder.make_dataset(tokenizer=tokenizer, args=args, column_name="text", shuffle=True)

quantizer = LLMQuantizer(model, args, config)
quantizer.orig_model = model.rwkv
quantizer.prepare_quantsim(dummy_input, args, dataset_builder.train_dataloader, tokenizer)

def test_generate(model, tokenizer,device='cuda'):
    config = model.config
    print("Generating inference using QuantSim model")
    prompt = "\n我们发现，"
    print(prompt, end='')
    input_ids = tokenizer (prompt, return_tensors='pt')
    input_ids.to(device)
    model.to(device)
    if isinstance(input_ids, BatchEncoding):
        attention_mask = input_ids['attention_mask']
        input_ids = input_ids['input_ids']
    output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=800, do_sample=True, repetition_penalty=1.1, top_p=1, top_k=128, temperature=1)
    print (tokenizer.batch_decode(output, skip_special_tokens=True)[0].split(prompt)[-1])

if args_parser.test_generate:
    test_generate(quantizer.quant_sim.model, tokenizer=tokenizer,device=args.device)
else:
    input_names, output_names = get_input_output_names(model.config)
    quantizer.export_quantsim(dummy_input=dummy_input, input_names=input_names, output_names=output_names)

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

    for node_id, node in enumerate(nodes):
        #print(node)
        if "ln1/Add_1_output_0" in node.output[0]:
            match = re.search(pattern, node.output[0])
            number = match.group(1)
            node.output[0] = "layer" + str(number) + "_state0_out"
        elif "ln2/Add_1_output_0" in node.output[0]:
            match = re.search(pattern, node.output[0])
            number = match.group(1)
            node.output[0] = "layer" + str(number) + "_state2_out"
        elif len(node.input) > 0:
            for id, _input in enumerate(node.input):
                if "ln1/Add_1_output_0" in _input:
                    match = re.search(pattern, node.input[id])
                    number = match.group(1)
                    node.input[id] = "layer" + str(number) + "_state0_out"
                elif "ln2/Add_1_output_0" in _input:
                    match = re.search(pattern, node.input[id])
                    number = match.group(1)
                    node.input[id] = "layer" + str(number) + "_state2_out"

        if "attention/sub_shift/Sub_output_0" in node.output[0]:
            match = re.search(pattern, node.output[0])
            number = match.group(1)
            node.input[0] = "layer" + str(number) + "_state0_in"
        elif "feed_forward/sub_shifted/Sub_output_0" in node.output[0]:
            match = re.search(pattern, node.output[0])
            number = match.group(1)
            node.input[0] = "layer" + str(number) + "_state2_in"
        elif "add_time_first/Add_output_0" in node.output[0]:
            match = re.search(pattern, node.output[0])
            number = match.group(1)
            node.input[1] = "layer" + str(number) + "_state1_in"
        elif "mul_time_decay/Mul_output_0" in node.output[0]:
            match = re.search(pattern, node.output[0])
            number = match.group(1)
            node.input[1] = "layer" + str(number) + "_state1_in"

    state_in_dims = {}
    for input in graph.input:
        if "state0" in input.name:
            state_in_dims["state0"] = input.type.tensor_type.shape
        elif "state1" in input.name:
            state_in_dims["state1"] = input.type.tensor_type.shape
        elif "state2" in input.name:
            state_in_dims["state2"] = input.type.tensor_type.shape
    for idx, output in enumerate(graph.output):
        if "state0" in output.name:
            for i, dim in enumerate(state_in_dims["state0"].dim):
                graph.output[idx].type.tensor_type.shape.dim[i].dim_value = dim.dim_value
        elif "state1" in output.name:
            for i, dim in enumerate(state_in_dims["state1"].dim):
                graph.output[idx].type.tensor_type.shape.dim[i].dim_value = dim.dim_value
        elif "state2" in output.name:
            for i, dim in enumerate(state_in_dims["state2"].dim):
                graph.output[idx].type.tensor_type.shape.dim[i].dim_value = dim.dim_value
        elif "logits" in output.name:
            graph.output[idx].type.tensor_type.shape.dim[0].dim_value = 1
            graph.output[idx].type.tensor_type.shape.dim[1].dim_value = 1
            graph.output[idx].type.tensor_type.shape.dim[2].dim_value = 65536

    model.ir_version = 8
    onnx.save_model(model, os.path.join(args.export_dir, "onnx", f"{args.model_name}.onnx"), save_as_external_data=True, all_tensors_to_one_file=True, size_threshold=1024, convert_attribute=False)

    split_onnx(os.path.join(args.export_dir, "onnx", f"{args.model_name}.onnx"), args.model_name, args_parser.num_chunks, args.export_dir, False)

    layers_per_chunk = len(quantizer.quant_sim.model.rwkv.blocks) // args_parser.num_chunks
    os.path.exists(os.path.join(args.export_dir, f"sample_inputs")) or os.mkdir(os.path.join(args.export_dir, f"sample_inputs"))
    sample_input_path = os.path.join(args.export_dir, f"sample_inputs", args.model_name)
    os.path.exists(sample_input_path) or os.mkdir(sample_input_path)
    # assume the layers are evenly distributed
    for i in range(args_parser.num_chunks):
        input_list_line = " ".join([f"{args.export_dir}/sample_inputs/chunk_{i}/input_{j}.bin" for j in range(3*layers_per_chunk+1)])
        os.path.exists(os.path.join(sample_input_path, f"chunk_{i}")) or os.mkdir(os.path.join(sample_input_path, f"chunk_{i}"))
        with open(os.path.join(sample_input_path, f"input_list_chunk_{i}.txt"), 'w') as f:
            f.write(input_list_line)
        if i == 0:
            np.zeros((1, 1), dtype=np.int32).tofile(os.path.join(sample_input_path, f"chunk_{i}", "input_0.bin"))
        else:
            np.zeros((1, 1, config.hidden_size), dtype=np.float32).tofile(os.path.join(sample_input_path, f"chunk_{i}", "input_0.bin"))
        for j in range(layers_per_chunk):
            np.zeros((1, 1, config.hidden_size), dtype=np.float32).tofile(os.path.join(sample_input_path, f"chunk_{i}", f"input_{3*j+1}.bin"))
            np.zeros((1, config.hidden_size//config.head_size, config.head_size, config.head_size), dtype=np.float32).tofile(os.path.join(sample_input_path, f"chunk_{i}", f"input_{3*j+2}.bin"))
            np.zeros((1, 1, config.hidden_size), dtype=np.float32).tofile(os.path.join(sample_input_path, f"chunk_{i}", f"input_{3*j+3}.bin"))
