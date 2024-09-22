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
from aimet_torch.model_preparer import _prepare_traced_model

from utils.model_utils import get_dummy_input_for_rwkv_causal_llm
from quantizers.advanced_ptq.actmse_quantizer import ActMSEQuantizer
from quantizers.base_quantizer import LLMQuantizer
from utils.dataset_builder import DatasetBuilder

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Convert model')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser.add_argument('model_name', type=str, help='Model name')
parser.add_argument('--weights_bitwidth', type=int, default=4, help='Weights bitwidth')
parser.add_argument('--use_cuda', action='store_true', default=True, help='Use CUDA')
args_parser = parser.parse_args()

config = AutoConfig.from_pretrained(str(args_parser.model), trust_remote_code=True)
config.vocab_size = 65536
config.model_max_length = 1024
config.return_top_k = 0
config.use_position_embedding_input = False
config.use_combined_mask_input = False
config.num_logits_to_return = 0
config.shift_cache = False

model = Rwkv6ForCausalLM.from_pretrained(str(args_parser.model), config=config)
device = torch.device("cuda") if args_parser.use_cuda and torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
model.rwkv.embeddings.weight = \
    nn.parameter.Parameter(data=nn.functional.layer_norm(model.rwkv.embeddings.weight, [model.config.hidden_size], \
                                weight=model.rwkv.blocks[0].pre_ln.weight, bias=model.rwkv.blocks[0].pre_ln.bias, eps=1e-5))
tokenizer = AutoTokenizer.from_pretrained(str(args_parser.model), trust_remote_code=True)
tokenizer.model_max_length = 1024

dummy_input = get_dummy_input_for_rwkv_causal_llm(1, 1, device, model_cfg=model.config)

args = types.SimpleNamespace()
##############################
args.quant_scheme = "tf"
args.activation_bit_width = 32
args.parameter_bit_width = args_parser.weights_bitwidth
args.in_place_quantsim = False
args.config_file = "quantizers/configs/default_per_channel_config.json"
args.num_cands = 30
args.export_dir = "quant_export"
args.output_dir = "quant_export"
args.model_name = args_parser.model_name
args.input_symmetry = "symqt"
args.exceptions_file = "quantizers/configs/rwkv_gptq_exceptions.json"
args.act_mse_loss_type = "mse"
args.parameter_encoding_file = None
args.encoding_path = None
args.do_actmse = True
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
args.num_calibration_batches = 30
args.per_device_calib_batch_size = 2
args.per_device_eval_batch_size = 4
args.block_size = 1024
args.seed = 1234
##############################

dataset_builder = DatasetBuilder(args)
dataset_builder.make_dataset(tokenizer=tokenizer, args=args, column_name="text", shuffle=True)

quantizer = ActMSEQuantizer(model, args, config)
quantizer.orig_model = model.rwkv
quantizer.prepare_quantsim(dummy_input, args, dataset_builder.train_dataloader, tokenizer)
