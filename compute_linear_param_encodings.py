# from rwkv_src.modeling_rwkv6 import Rwkv6ForCausalLM
from rwkv_src.rwkv_model import RWKV_RNN
from transformers import AutoConfig, AutoTokenizer
import types
import torch
import torch.nn as nn
from transformers.tokenization_utils_base import BatchEncoding

from utils.model_utils import get_dummy_input_for_rwkv_causal_llm
from quantizers.advanced_ptq.actmse_quantizer import ActMSEQuantizer
from utils.dataset_builder import DatasetBuilder

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Compute param encodings for linear modules')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser.add_argument('--weights_bitwidth', type=int, default=4, help='Weights bitwidth')
parser.add_argument('--use_cuda', action='store_true', default=True, help='Use CUDA')
parser.add_argument('--test_generate', action='store_true', default=False, help='Test generate')
args_parser = parser.parse_args()

args = types.SimpleNamespace()
##############################
args.quant_scheme = "tf"
args.activation_bit_width = 32
args.parameter_bit_width = args_parser.weights_bitwidth
args.in_place_quantsim = False
args.config_file = "quantizers/configs/default_per_channel_config.json"
args.num_cands = 20
args.export_dir = "quant_export"
args.output_dir = "quant_export"
args.model_name = str(args_parser.model).replace(".pth", "").split("/")[-1]
args.input_symmetry = "symqt"
args.exceptions_file = "quantizers/configs/rwkv_gptq_exceptions.json"
args.act_mse_loss_type = "mse"
args.parameter_encoding_file = None
args.encoding_path = None
args.do_actmse = True
args.disable_act_quantizers = True
args.fp16 = False
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

quantizer = ActMSEQuantizer(model, args, model.args)
quantizer.orig_model = model
quantizer.prepare_quantsim(dummy_input, args, dataset_builder.train_dataloader, tokenizer)

def test_generate(model, tokenizer,device='cuda'):
    config = model.config
    print("Generating inference using QuantSim model")
    prompt = "User: 请为我写一首诗\n\nAssistant:"
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