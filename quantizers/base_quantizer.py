import os
import math
import json
import time
import pickle
import inspect
from collections.abc import Iterable
import itertools
import copy

import torch
import numpy as np
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from transformers import TopKLogitsWarper, set_seed

import aimet_common.libpymo as libpymo
from aimet_common.defs import QuantScheme
from aimet_common.utils import save_json_yaml
# from aimet_torch.pro.quantsim import QuantizationSimModel
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.quantsim import load_encodings_to_sim, _get_encoding_by_quantizer
from aimet_torch.onnx_utils import OnnxExportApiArgs
from aimet_torch.qc_quantize_op import QcQuantizeWrapper, QcQuantizeOpMode
from aimet_torch import utils as aimet_utils

from .exceptions import ExceptionConfigurator
import utils.model_utils as utils

from importlib.metadata import version as impLib_version
from packaging import version


# Exception handling in case fastforward is not enabled in AIMET build
try:
    from aimet_torch.quantized_layers import fastforward, revert_fastforward, quantized_modules

    def has_fastforward_nodes(model):
        """
        Check if any fastforward nodes are present in the model
        """
        for module in model.modules():
            if isinstance(module, quantized_modules.SimulatedIntLinear):
                return True
        return False

except (ImportError, NotImplementedError):
    print("Failed to import fastforward build")
    def has_fastforward_nodes(_):
        """
        Always returns false if fastforward import failed
        """
        return False


def get_lambada_acc(pred, label, pad_token_id):
    # (batch, len_sequence)
    # find the last token
    output = []
    for b in range(label.shape[0]):
        for last_token_idx in range(label.shape[1]):
            if label[b][last_token_idx] == pad_token_id:
                break

        last_token = label[b][last_token_idx - 1]
        pred_token = pred[b][last_token_idx - 2]
        output.append((last_token == pred_token).float().item() * 100)
    return output

def flatten_tensors(tup):
    if not isinstance(tup, (tuple,list)):
        yield tup
        return
    for x in tup:
        yield from flatten_tensors(x)


def is_nested_tuple(tup):
    if not isinstance(tup, tuple):
        raise RuntimeError(f'{tup} is not a tuple')

    flat_tup = tuple(flatten_tensors(tup))

    if len(flat_tup) == len(tup) and all((x1 is x2) for x1, x2 in zip(flat_tup, tup)):
        # Flattened tuple is equal to the original tuple.
        # In other words, `tup` is already a flat tuple.
        return False

    return True


def is_flat_tuple(tup):
    return not is_nested_tuple(tup)


def disallow_fastforward_nodes(func):
    """
    Decorator to make sure model does not have fastforward nodes when entering a function
    """
    def wrapper(*args, **kwargs):
        quantizer = args[0]
        assert isinstance(quantizer, LLMQuantizer)
        if has_fastforward_nodes(quantizer.quant_sim.model):
            raise TypeError(f"Cannot call {func} with fastforward nodes present")
        return func(*args, **kwargs)

    return wrapper

def get_tokens(logits, get_max_token_mode='default'):
    if get_max_token_mode == 'argmax':
        return torch.argmax(logits, dim=-1)
    elif get_max_token_mode == 'top50':
        scores, indices = torch.topk(logits, 50)
        probs = torch.nn.functional.softmax(scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        return [indices[0][next_tokens[0]]]
    elif get_max_token_mode == 'default':
        logits_wraper = TopKLogitsWarper(50, min_tokens_to_keep=2)
        logits = logits_wraper(None, logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_tokens
    else:
        raise ValueError("get_max_token_mode should be one of (argmax, top50, default)")


# @register_quantizer("llm_quantizer")
class LLMQuantizer:
    def __init__(self, model:torch.nn.Module, args, model_config):
        self.model = model
        self.quant_scheme = args.quant_scheme
        self.act_bw = args.activation_bit_width
        self.param_bw = args.parameter_bit_width
        self.in_place = args.in_place_quantsim
        self.config_file = args.config_file or 'config/default_per_channel_config.json'
        self.num_calibration_batches = args.num_calibration_batches
        self.export_path = args.export_dir
        self.output_dir = args.output_dir
        self.model_config = model_config
        self.model_name = args.model_name
        self.encoding_file = getattr(args, 'encoding_file', None)
        self.load_sim_checkpoint = args.load_sim_checkpoint
        self.save_sim_checkpoint = args.save_sim_checkpoint

    def parse_quant_scheme(self, quant_scheme):
        if quant_scheme == "tf":
            return QuantScheme.post_training_tf
        elif quant_scheme == "tf_enhanced":
            return QuantScheme.post_training_tf_enhanced
        elif quant_scheme == "tf_range_learning":
            return QuantScheme.training_range_learning_with_tf_init
        elif quant_scheme == "tfe_range_learning":
            return QuantScheme.training_range_learning_with_tf_enhanced_init
        else:
            raise ValueError("select appropriate quantization scheme in [tf, tf_enhanced, tf_range_learning]")

    def rwkv_generate(self, model, tokenizer, prompt):
        with torch.no_grad():
            encoder_prompt = tokenizer(prompt, return_tensors='pt')
            model = model
            device = model.device

            input_ids = encoder_prompt["input_ids"].to(device=device)
            token_size = input_ids.shape[1]
            batch_size = input_ids.shape[0]
            num_input_tokens = 1

            past = utils.get_dummy_state_kvcache(batch_size, model.config, model.device)
            model_inputs = {'input_ids':None, 'state':None}
            print("Prompt stage...")
            for index in range(token_size):
                ids = input_ids[:,index:index+num_input_tokens]
                model_inputs['input_ids'] = ids
                model_inputs['state'] = past
                outputs = model(**model_inputs)
                past = outputs[1]
                ids = torch.argmax(outputs[0], dim=-1)
                print(ids)
            print("Generate stage...")
            # 2st inference(for new token)
            ids = torch.argmax(outputs[0], dim=-1)
            output_ids = None
            for _ in range(300):
                model_inputs['input_ids'] = ids
                model_inputs['state'] = past
                outputs = model(**model_inputs)
                past = outputs[1]
                ids = torch.argmax(outputs[0], dim=-1)
                print(ids)
                if output_ids is None:
                    output_ids = ids.clone()
                else:
                    output_ids = torch.cat((output_ids, ids), 1)
                # Determine whether it is finished and save the results
                if tokenizer.eos_token_id in ids:
                    break
            outputs_token = tokenizer.decode(output_ids.flatten(), skip_special_tokens=True)
            return prompt + outputs_token

    def create_quantsim(self, dummy_input):
        model_inputs = tuple([dummy_input[input] for input in dummy_input.keys()])
        #We need to flatten the tuples into a flat list of tensors when working with a prepared model.
        if "position_ids_cos" in inspect.signature(self.model.forward).parameters or \
           "past_key_0_h0_in" in inspect.signature(self.model.forward).parameters:
            model_inputs = tuple(flatten_tensors(model_inputs))
        if self.in_place:
            self.quant_sim = QuantizationSimModel(
                model=self.model,
                quant_scheme=self.parse_quant_scheme(self.quant_scheme),
                dummy_input=model_inputs,
                default_output_bw=self.act_bw,
                default_param_bw=self.param_bw,
                in_place=self.in_place,
                config_file=self.config_file,
            )
        else:
            device = self.model.device
            self.model = self.model.to(torch.device("cpu"))
            fp_model = copy.deepcopy(self.model)
            self.quant_sim = QuantizationSimModel(
                model=fp_model.to(device),
                quant_scheme=self.parse_quant_scheme(self.quant_scheme),
                dummy_input=model_inputs,
                default_output_bw=self.act_bw,
                default_param_bw=self.param_bw,
                in_place=True,
                config_file=self.config_file,
            )
            torch.cuda.empty_cache()
        #print(self.quant_sim)

    def prepare_quantsim(self, dummy_input, args, train_dataloader, tokenizer):
        print("Creating QuantSim model")
        tic = time.time()
        self.create_quantsim(dummy_input=dummy_input)
        print(f"Created quantsim in {time.time() - tic} seconds")

        #Apply exceptions to QuantSim before computing encodings
        exception_configurator = ExceptionConfigurator(args)
        print("Applying pre-calibration exceptions")
        tic = time.time()
        exception_configurator.apply_pre_calibration_exceptions(self.quant_sim)
        
        if args.parameter_encoding_file:
            loaded_quantizers = self.load_param_encodings(args.parameter_encoding_file)
        else:
            loaded_quantizers = []

        if args.encoding_path:
            self.load_encodings(args.encoding_path)
        else:
            with utils.freeze_quantizers(loaded_quantizers):
                # Temporarily freezed the loaded quantizers so that compute_encodings() doesn't
                # overwrite the loaded encodings.

                param_quantizers, input_quantizers, output_quantizers = aimet_utils.get_all_quantizers(self.quant_sim.model)
                all_quantizers = param_quantizers + input_quantizers + output_quantizers

                if any(q.enabled and not q.is_encoding_frozen for q in all_quantizers):
                    print("Computing encodings")
                    tic = time.time()
                    self.compute_encodings(calib_dataloader=train_dataloader, tokenizer=tokenizer)
                    print(f"Computed encodings in {time.time() - tic} seconds")
                else:
                    print("All the quantizers are disabled or frozen. Skipping compute_encodings.")

        if not args.encoding_path:
            # Apply post-calibration exceptions
            print("Applying post-calibration exceptions")
            tic = time.time()
            exception_configurator.apply_post_calibration_exceptions(self.quant_sim)
            if args.clip_activation:
                self.clip_activation_quantizers(-args.clip_activation, args.clip_activation)
            print(f"Applied post-cal exceptions in {time.time() - tic} seconds")

        if self.save_sim_checkpoint:
            self.save_checkpoint(self.quant_sim, self.save_sim_checkpoint)

        # quant_sim_str = self.quant_sim.__str__()
        # with open('quant_sim.txt', 'w') as f:
        #     f.write(quant_sim_str)
        
        # state_dict will be loaded nectar.registry.make_model while make_trainer is being called
        self.load_encodings(self.encoding_file)

    def load_param_encodings(self, path, freeze=True):
        """
        onnx/*_torch.encodings has {activation_encodings: {...}, param_encodings: {...}}
        but some encodings only have param_encodings within the encoding file,
        this loader is to load whaterver format they have and only use param encodings in there
        """
        loaded_quantizers = []
        with open(path) as json_file:
            param_encodings = json.load(json_file)

        if "param_encodings" in param_encodings:
            param_encodings = param_encodings["param_encodings"]

        for module_name, quant_module in self.quant_sim.model.named_modules():
            if isinstance(quant_module, QcQuantizeWrapper):

                # set_param_encoding
                skipped = True
                for orig_param_name, param_quantizer in quant_module.param_quantizers.items():
                    param_name = module_name + '.' + orig_param_name
                    if param_name not in param_encodings or param_encodings[param_name][0] is None:
                        continue
                    if param_name in param_encodings:
                        encodings = []
                        for encoding_dict in param_encodings[param_name]:
                            encoding = aimet_utils.create_encoding_from_dict(encoding_dict)
                            is_symmetric = encoding_dict.get('is_symmetric')
                            encodings.append(encoding)
                        param_quantizer.bitwidth = encodings[0].bw
                        param_quantizer.use_symmetric_encodings = is_symmetric
                        param_quantizer.encoding = encodings
                        print(f"Setting quantization encodings for parameter: {param_name}")
                        skipped = False


                if not skipped:
                    loaded_quantizers.append(param_quantizer)

                if freeze and not skipped:
                    quant_module.freeze_param_encoding(module_name, param_encodings)
        return loaded_quantizers


    def clip_activation_quantizers(self, clamp_min, clamp_max):

        if "range_learning" in self.quant_scheme:
            for name, param in self.quant_sim.model.named_parameters():
                if name.endswith("encoding_min") or name.endswith("encoding_max"):
                    _old_val = param.data.min().item(), param.data.max().item()
                    param.data = torch.clamp(param.data, clamp_min, clamp_max)
                    _new_val = param.data.min().item(), param.data.max().item()

                    if _old_val != _new_val:
                        print(name)
                        if clamp_min == _new_val[0]:
                            print(f"Activation clamping... before: {_old_val[0]} | after: {clamp_min}")
                        else:
                            print(f"Activation clamping... before: {_old_val[1]} | after: {clamp_max}")

        elif self.quant_scheme == "tf":
            def clip_and_recompute_encodings(quantizer, name):
                if (not quantizer.encoding) or (not quantizer.enabled):
                    return
                qmin = quantizer.encoding.min
                qmax = quantizer.encoding.max
                if qmin < clamp_min or qmax > clamp_max:
                    tensor = torch.clamp(
                        torch.tensor([qmin, qmax]), clamp_min, clamp_max
                    )
                    quantizer.reset_encoding_stats()
                    quantizer.update_encoding_stats(tensor)
                    quantizer.compute_encoding()

                    print(name)
                    print(f"Activation clamping... before: {qmin}, {qmax} | after: {quantizer.encoding.min}, {quantizer.encoding.max}")

            for name, module in self.quant_sim.model.named_modules():
                if isinstance(module, QcQuantizeWrapper):
                    for quantizer in module.output_quantizers:
                        clip_and_recompute_encodings(quantizer, name + " | output quantizer")
                    for quantizer in module.input_quantizers:
                        clip_and_recompute_encodings(quantizer, name + " | input quantizer")


    def enable_fastforward(self):
        fastforward(self.quant_sim.model)

    def disable_fastforward(self):
        revert_fastforward(self.quant_sim)

    @disallow_fastforward_nodes
    def load_encodings(self, encoding_file=None):
        if encoding_file:
            print("Loading encodings from path: ", encoding_file)
            load_encodings_to_sim(self.quant_sim, encoding_file)

    @disallow_fastforward_nodes
    def save_param_encodings(self, output_path=None):
        """
        generate model_torch.encodings without onnx
        """
        print(" -- only save the parameter encodings...")
        start_time = time.time()
        param_encodings = {}

        for layer_name, layer in tqdm(QuantizationSimModel._get_qc_quantized_layers(self.quant_sim.model)):
            for orig_param_name, param_quantizer in layer.param_quantizers.items():
                param_name = layer_name + '.' + orig_param_name
                if not param_quantizer.enabled:
                    continue
                elif isinstance(param_quantizer.encoding, Iterable):
                    param_encodings[param_name] = []
                    quantizer_encoding = _get_encoding_by_quantizer(param_quantizer)
                    for encoding in quantizer_encoding:
                        enc_dict = QuantizationSimModel._create_encoding_dict(encoding,
                            param_quantizer, propagate_encodings=False)
                        param_encodings[param_name].append(enc_dict)
                else:
                    quantizer_encoding = _get_encoding_by_quantizer(param_quantizer)
                    enc_dict = QuantizationSimModel._create_encoding_dict(quantizer_encoding, param_quantizer,
                        propagate_encodings=False)
                    param_encodings[param_name] = [enc_dict]


        encodings_dict_onnx = {
            "version": "0.6.1",
            "activation_encodings": {},
            "param_encodings": param_encodings,
            "excluded_layers": []
        }
        if output_path is None:
            onnx_path = os.path.join(self.output_dir, "onnx")
            if not os.path.exists(onnx_path):
                os.makedirs(onnx_path)
            output_path = os.path.join(onnx_path, f"{self._prefix}_torch.encodings")
        save_json_yaml(output_path, encodings_dict_onnx)
        print(f"encodings saved in {output_path} ... {time.time() - start_time:0.1f}s")

    @torch.no_grad()
    def add_to_test_vectors(self, test_vectors, inputs, outputs, hooked):

        def extract_test_data_by_batch(data, batch_index):
            def _pop(t):
                if isinstance(t, tuple):
                    return tuple(_pop(i) for i in t)
                if isinstance(t, list):
                    return list(_pop(i) for i in t)
                if isinstance(t, dict):
                    return {k:_pop(v) for k,v in t.items()}
                assert isinstance(t, torch.Tensor), f'Unexpected type({type(t)}) in data'
                assert t.shape[0] >= batch_index, f'Invalid batch_index:{batch_index} t.shape:{t.shape}'
                return utils.to_cpu(t[batch_index:batch_index+1])
            return _pop(data)

        def _cut_by(iterable, n):
            def _impl(it, n):
                cut = cls(itertools.islice(it, n))
                while cut:
                    yield cut
                    cut = cls(itertools.islice(it, n))
            cls = type(iterable)
            return cls(_impl(iter(iterable), n))

        data = inputs

        # Add outputs
        output_index = 0
        data['logits'] = outputs[output_index]
        output_index += 1
        if self.model_config.return_top_k > 0:
            data['indices'] = outputs[output_index]
            output_index += 1
        output_remained = len(outputs[output_index:])
        if output_remained == 1:
            assert isinstance(outputs[output_index], tuple), f'Unexpected output type:{type(outputs[output_index])}'
            data['output_key_values'] = outputs[output_index]
        else:
            # Tuple un-tupled output
            num_layers, num_heads, *_ = utils.extract_info_from_model_cfg(self.model_config)
            assert output_remained==num_layers*2*num_heads, f'Unexpected number of output remained:{output_remained}, total {len(outputs)} outputs'
            cache = _cut_by(tuple(outputs[output_index:]), num_heads)
            cache = _cut_by(cache, 2)
            cache = _cut_by(cache, num_layers)
            data['output_key_values'] = cache[0]
    @torch.no_grad()
    def rwkv_add_to_test_vectors(self, test_vectors, inputs, outputs, hooked, converted_flag):

        def extract_test_data_by_batch(data, batch_index):
            def _pop(t):
                if isinstance(t, tuple):
                    return tuple(_pop(i) for i in t)
                if isinstance(t, list):
                    return list(_pop(i) for i in t)
                if isinstance(t, dict):
                    return {k:_pop(v) for k,v in t.items()}
                if isinstance(t, str):
                    return t
                assert isinstance(t, torch.Tensor), f'Unexpected type({type(t)}) in data'
                assert t.shape[0] >= batch_index, f'Invalid batch_index:{batch_index} t.shape:{t.shape}'
                return utils.to_cpu(t[batch_index:batch_index+1])
            return _pop(data)

        data = inputs
        # Add outputs
        data['logits'] = outputs[0]
        if converted_flag == 1:
            data['state_out'] = outputs[1:]
        else:
            data['state_out'] = outputs[1]

        # Hooked data
        if hooked is not None:
            for name, tensors in hooked.items():
                data[name] = {key: tensors[key] for key in tensors.keys()}

        # Save it by batch
        sample_index = len(test_vectors)
        this_batch_size = inputs['input_ids'].shape[0]
        for batch_index in range(this_batch_size):
            test_vectors[str(sample_index)] = extract_test_data_by_batch(data, batch_index)
            sample_index += 1

    def _prepare_past_key_value(self, new_key_value, prev_key_value, shift_size=1):
        # past_key_value: [num_layers][2][key_value], where key_value can be a tensor or tuple of heads

        def _concat(a, b, dim):
            if isinstance(a, tuple):
                assert len(a) == len(b), 'Unexpected key/value pair'
                return tuple(_concat(ai, bi, dim) for ai, bi in zip(a, b))
            return torch.cat((a, b), dim=dim)

        def _do_concat(a, b, key_dim, value_dim):
            return tuple((_concat(ak, bk, key_dim), _concat(av, bv, value_dim)) for (ak, av), (bk, bv) in zip(a, b))

        def _shift(a, dim):
            if isinstance(a, tuple):
                return tuple(_shift(ai, dim) for ai in a)
            assert dim in (2,3), 'Unexpected shift axis'
            return a[:, :, shift_size:, :] if dim==2 else a[:, :, :, shift_size:]

        def _do_shift(a, key_dim, value_dim):
            return tuple((_shift(k, key_dim), _shift(v, value_dim)) for k, v in a)

        need_cache_shift = not self.model_config.shift_cache
        need_accumulate_cache = self.model_config.return_new_key_value_only
        value_dim = 2
        key_dim = 3 if self.model_config.transposed_key_cache else 2
        next_key_value = _do_concat(prev_key_value, new_key_value, key_dim, value_dim) if need_accumulate_cache else new_key_value
        next_key_value = _do_shift(next_key_value, key_dim, value_dim) if need_cache_shift else next_key_value
        return next_key_value

    @torch.no_grad()
    def evaluate(self, model, iterations, loader: Iterable, tokenizer, metric, get_test_vectors=False, hook_intermediate_tensor=False, hook_config_file=None, do_eval=False):
        def encode(
            *args,
            add_special_tokens=False,
            return_tensors='pt',
            max_length=tokenizer.model_max_length,
            **kwargs,
        ):
            """Redirect to the encode method on the tokenizer"""
            batch_encoding = tokenizer(*args, add_special_tokens=add_special_tokens, return_tensors=return_tensors,
                                            return_token_type_ids=False,
                                            padding='max_length', max_length=max_length, truncation=True, **kwargs)
            return batch_encoding

        model.eval()
        test_vectors = {}
        losses = []
        accs = []

        if hook_intermediate_tensor:
            assert hook_config_file is not None, "Please provide a config file to setup hooks"
            hook_handler, hook_recorder = self.register_forward_hooks_to_blocks(model, hook_config_file)

        #TODO move the input tensor generation to utils. This is repeated for dummy_input and inputs needed for calibration

        kvcache_mode = True
        max_length = tokenizer.model_max_length
        past_output_offset = 2 if self.model_config.return_top_k > 0 else 1

        rope = None
        if self.model_config.use_position_embedding_input:
            _, num_heads, _, embed_dim = utils.extract_info_from_model_cfg(self.model_config)
            rope = utils.RopeEmbedding(device=model.device, head_dim=embed_dim//num_heads, max_length=max_length)

        def _pad(ids, pad_value, target_length):
            assert len(ids.shape) == 2, "Unexpected input shape:{ids.shape} to _pad"
            batch_size, input_length = ids.shape
            pad_size = target_length - input_length
            if pad_size > 0:
                ids = torch.cat([torch.Tensor([[pad_value] * pad_size] * batch_size).to(ids.device), ids], dim=1).to(ids.dtype)
            return ids

        def _adjust_inputs(ids, mask, pos, rope=None, past_key_values_length=0):
            if self.model_config.use_combined_mask_input:
                mask = utils.prepare_combined_attention_mask(mask, ids.shape, model.device, past_key_values_length=past_key_values_length)
            if rope:
                pos = rope.get_embedding(pos)
            return ids, mask, pos

        for batch_id, batch in enumerate(tqdm(loader)):
            if batch_id < iterations:
                if "text" in batch:
                    encoded_tensor = encode(batch["text"], max_length=max_length)
                else:
                    encoded_tensor = batch
                if "labels" not in encoded_tensor:
                    encoded_tensor["labels"] = encoded_tensor["input_ids"].detach()
                labels = encoded_tensor.pop("labels").to(device=model.device)

                # Inputs
                value = torch.cumsum(encoded_tensor['attention_mask'], dim=1) - 1
                position_ids = value.clip(0, max_length - 1).to(device=model.device)
                input_ids, attention_mask = [encoded_tensor[i].to(device=model.device) for i in ['input_ids', 'attention_mask']]

                if not kvcache_mode:
                    # bertcache mode: Single inference with full input
                    #
                    # m:max_length, i:num_input_tokens, n:num_logits_to_return
                    #
                    #              │ inferences
                    # ─────────────┼───────────
                    #   inputs     │ ids[m]   (or [m-1]?)
                    #              │ pos[m]
                    #              │ mask[m]
                    # ─────────────┼───────────
                    #   outputs    │ logits[n]
                    #              │ past[m]

                    ids, mask, pos = _adjust_inputs(input_ids, attention_mask, position_ids, rope=rope)
                    if hasattr(model, 'rwkv'):
                        model_inputs = {'input_ids':ids}
                    else:
                        model_inputs = {'input_ids':ids, 'attention_mask':mask, 'position_ids':pos}
                    #We need to flatten the tuples into a flat list of tensors when working with a prepared model.
                    if "position_ids_cos" in inspect.signature(self.model.forward).parameters or \
                       "past_key_0_h0_in" in inspect.signature(self.model.forward).parameters:
                        outputs = model(*flatten_tensors(tuple(model_inputs.values())))
                    else:
                        outputs = model(**model_inputs)

                else:
                    if do_eval:
                        # kvcache ppl score evaluation
                        outputs = self.compute_kv_output_logits(model, max_length, past_output_offset, input_ids,
                                                                position_ids, rope, _adjust_inputs)
                    else:
                        # kvcache mode: Multiple inferences with sliced inputs,
                        # The goal is to
                        # - Use user tokens only, no generated tokens
                        # - Let the bert and kv$ models consume the same data eventually
                        #
                        # m:max_length, i:num_input_tokens, n:num_logits_to_return
                        #
                        #  inferences  │   1st       │  next
                        # ─────────────┼─────────────┼─────────────────────
                        #   inputs     │ ids[m-i]    │ ids[i]
                        #              │ pos[m-i]    │ pos[i]
                        #              │ mask[m]     │ mask[m]
                        #              │ past[i]     │ past[m-i]
                        # ─────────────┼─────────────┼─────────────────────
                        #   outputs    │ logits[n]   │ logits[n]
                        #              │ past[m-i]   │ past[i]

                        num_loops = 20#5
                        batch_size = input_ids.shape[0]
                        num_input_tokens = 1 if self.model_config.num_logits_to_return <= 1 else self.model_config.num_logits_to_return
                        assert not (self.model_config.shift_cache and need_accumulate_cache), "Unsupported configuration"

                        cur =  max_length - num_input_tokens * num_loops
                        ids = _pad(input_ids[:, :cur], self.model_config.eos_token_id, max_length - num_input_tokens)
                        pos = _pad(position_ids[:, :cur], 0, max_length - num_input_tokens)
                        mask = _pad(attention_mask[:, :cur], 0, max_length)

                        past = utils.get_dummy_state_kvcache(batch_size, self.model_config, model.device)

                        # 1st inference
                        if type(model.rwkv) == str:
                            model_inputs = utils.rwkv_get_converted_dict(ids, past)
                        else:
                            model_inputs = {'input_ids':ids, 'state': past}
                        
                        if 1:  # Firstly run with long sequence for once, and then run with sequence=1 in last num_loops
                            # We need to flatten the tuples into a flat list of tensors when working with a prepared model.
                            if "position_ids_cos" in inspect.signature(self.model.forward).parameters or \
                            "past_key_0_h0_in" in inspect.signature(self.model.forward).parameters:
                                # KV-cache prepared model can only handle input of length 1.
                                # For the initial bert-mode inference, we use the original unprepared fp32 model
                                # TODO (kyunggeu): This is a ad-hoc workaround. Need to be replaced with a real solution
                                orig_model = getattr(self, 'orig_model')
                                device = model.device
                                #Switch the HF model to CUDA and the prepared model to CPU (to prevent CUDA OOM error)
                                orig_model = orig_model.to(device=device)
                                #Now place the prepared model on CPU. If we did this first, the previous line would place the prepared model in CUDA
                                #since the prepared model is an attribute of the original HF model
                                model = model.to(device='cpu')
                                model.device = next(model.parameters()).device
                                outputs = orig_model(**model_inputs)
                                #Reinstate the models to the right devices
                                orig_model = orig_model.to(device='cpu')
                                model = model.to(device=device)
                                model.device = next(model.parameters()).device
                            else:
                                outputs = model(**model_inputs)

                            if hasattr(model, 'rwkv'):
                                if type(model.rwkv) == str:
                                    past = outputs[1:] #state
                                else:
                                    past = outputs[1] #state
                            else:
                                past = outputs[past_output_offset] if isinstance(outputs[past_output_offset], tuple) else outputs[past_output_offset:]
                        else: # Always run with sequence=1, it will be slower than first one
                            num_loops = max_length
                            print(f"max_length: {max_length}")
                            cur = 0

                        for _ in range(num_loops):
                            ids =  input_ids[:, cur:cur+num_input_tokens]
                            pos =  position_ids[:, cur:cur+num_input_tokens]
                            mask = _pad(attention_mask[:, :cur+num_input_tokens], 0, max_length)
                            cur += num_input_tokens

                            if hasattr(model, 'rwkv'):
                                if type(model.rwkv) == str:
                                    model_inputs = utils.rwkv_get_converted_dict(ids, past)
                                else:
                                    model_inputs = {'input_ids':ids, 'state': past}
                            else:
                                ids, mask, pos = _adjust_inputs(ids, mask, pos, rope=rope, past_key_values_length=max_length-num_input_tokens)
                                model_inputs = {'input_ids':ids, 'attention_mask':mask, 'position_ids':pos, 'past_key_values':past}

                            #We need to flatten the tuples into a flat list of tensors when working with a prepared model.
                            if "position_ids_cos" in inspect.signature(self.model.forward).parameters or \
                               "past_key_0_h0_in" in inspect.signature(self.model.forward).parameters:
                                outputs = model(*flatten_tensors(tuple(model_inputs.values())))
                            else:
                                outputs = model(**model_inputs)

                            if hasattr(model, 'rwkv'):
                                if type(model.rwkv) == str:
                                    past = outputs[1:] #state
                                else:
                                    past = outputs[1] #state
                            else:
                                new_past = outputs[past_output_offset] if isinstance(outputs[past_output_offset], tuple) else outputs[past_output_offset:]
    
                                if is_nested_tuple(past) and is_flat_tuple(new_past):
                                    # `past`, the output of the unprepared model, is a nested tuple of key-value tensors,
                                    # whereas `new_past`, the output of the prepared model, is a flat tuple of key-value tensors.
                                    # To concatenate `new_past` to `past`, we should first reform `new_past` into a nested tuple
                                    # of the same structure as `past`.
                                    num_layers, num_heads, _, _ = utils.extract_info_from_model_cfg(self.model_config)
                                    #Change the expectation about number of tensors depending on whether we are splitting KV$ tensors per head
                                    if not self.model_config.separate_kv_head:
                                        num_heads = 1
                                        assert len(new_past) == num_layers * num_heads * 2
                                    else:
                                        assert len(new_past) == num_layers * num_heads * 2
                                    new_past = tuple(
                                        (new_past[2*i*num_heads:2*i*num_heads+num_heads],
                                        new_past[2*i*num_heads+num_heads:2*(i+1)*num_heads]) for i in range(num_layers)
                                    )
                                    if num_heads == 1:
                                        new_past = tuple((k[0], v[0]) for k, v in new_past)
    
                                past = self._prepare_past_key_value(new_past, past, shift_size=num_input_tokens)

                        # Completed all iterations in kv_cache mode
                        # the last model_inputs will be included in the test vector

                if get_test_vectors:
                    if hasattr(model, 'rwkv'):
                        converted_flag = 0
                        if type(model.rwkv) == str:
                            converted_flag = 1
                        self.rwkv_add_to_test_vectors(test_vectors, model_inputs, outputs,
                                hook_recorder.data if hook_intermediate_tensor else None, converted_flag)
                    else:
                        self.add_to_test_vectors(test_vectors, model_inputs, outputs,
                            hook_recorder.data if hook_intermediate_tensor else None)

                # Need to figure out how to pass labels and compute loss
                if do_eval:
                    vocab_size = outputs[0].shape[-1]
                    batch_size = input_ids.shape[0]
                    lm_logits = torch.cat(outputs, dim=1) if do_eval and self.export_mode == 'kvcache' else outputs[0]
                    lm_logits = lm_logits.reshape(batch_size, -1, vocab_size)
                    # Shift so that tokens < n predict n
                    shift_logits = lm_logits if do_eval and self.export_mode == 'kvcache' else lm_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )

                    losses.append(loss.item())
                    if metric == "accuracy":
                        accs.extend(
                            get_lambada_acc(
                                outputs[0].argmax(-1),
                                labels,
                                self.model_config.eos_token_id,
                            )
                        )
            else:
                break

        if hook_intermediate_tensor:
            self.remove_forward_hooks(hook_handler)

        outputs = {}
        if do_eval:
            loss = np.mean(losses)
            outputs["loss"] = loss
            if metric == "perplexity":
                try:
                    ppl = math.exp(loss)
                except OverflowError:
                    ppl = float("inf")
                outputs["perplexity"] = ppl
            elif metric == "accuracy":
                outputs["accuracy"] = np.mean(accs)

        return outputs, test_vectors

    def eval_wrapper(self, model, args):
        iterations, loader, tokenizer, metric = args
        return self.evaluate(model, iterations=iterations, loader=loader, tokenizer=tokenizer, metric=metric)

    def compute_kv_output_logits(self, model, max_length, past_output_offset, input_ids, position_ids, rope, _adjust_inputs):
        output_logits = []
        batch_size = input_ids.shape[0]
        past_kv_length = max_length - 1
        past = utils.get_dummy_state_kvcache(batch_size, self.model_config, model.device)
        attention_mask = torch.zeros((batch_size, max_length), device=model.device, dtype=torch.long)
        attention_mask[:, -1] = 1
        print(max_length - 1)
        for n in range(max_length - 1):
            ids = input_ids[:, n:n + 1]
            pos = position_ids[:, n:n + 1]
            ids, mask, pos = _adjust_inputs(ids, attention_mask, pos, rope=rope, past_key_values_length=past_kv_length)
            if type(model.rwkv) == str:
                model_inputs = utils.rwkv_get_converted_dict(ids, past)
            else:
                model_inputs = {'input_ids':ids, 'state': past}
            # We need to flatten the tuples into a flat list of tensors when working with a prepared model.
            if "position_ids_cos" in inspect.signature(self.model.forward).parameters or \
                    "past_key_0_h0_in" in inspect.signature(self.model.forward).parameters:
                outputs = model(*flatten_tensors(tuple(model_inputs.values())))
            else:
                outputs = model(**model_inputs)
            output_logits.append(outputs[0].detach().clone())
            if len(output_logits) == (max_length - 1):
                break
            attention_mask[:, -n - 2] = 1
            if type(model.rwkv) == str:
                past = outputs[1:] #state
            else:
                past = outputs[1] #state

        return output_logits

    @disallow_fastforward_nodes
    def compute_encodings(self, calib_dataloader, tokenizer):
        if calib_dataloader is not None:
            self.quant_sim.compute_encodings(self.eval_wrapper,
                                             (self.num_calibration_batches, calib_dataloader, tokenizer, "None"))

    @disallow_fastforward_nodes
    def export_quantsim(self, dummy_input, input_names=None, output_names=None, opset_version=11):
        print("in export_quantsim")
        tic = time.time()
        cpu_dummy_input = utils.to_cpu(dummy_input)
        model_device = self.quant_sim.model.device
        self.quant_sim.model.to(device='cpu')

        onnx_api_args = OnnxExportApiArgs(
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
        )

        output_dir = f'{self.export_path}/onnx'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Changed model and tensor placement in {time.time() - tic} seconds")
        print("Calling export on quantsim object")
        #Store the original dtypes of params so that we can restore them later    
        orig_dtypes = {n: p.dtype for n, p in self.quant_sim.model.named_parameters()}
        self.quant_sim.model = self.quant_sim.model.float()
        
        tic = time.time()
        self.quant_sim.export(output_dir, self.model_name, cpu_dummy_input, onnx_export_args=onnx_api_args)
        print(f"Completed export in {time.time() - tic} seconds")
        
        #Restore the dtypes of parameters
        for n, p in self.quant_sim.model.named_parameters():
            p.data = p.data.to(orig_dtypes[n])        

        self.quant_sim.model.to(device=model_device)

    @disallow_fastforward_nodes
    def save_checkpoint(self, save_onnx=False):
        if version.parse(impLib_version('AimetTorch')) >= version.parse('1.29.0.0.181.0.4483+torch.gpu') and not save_onnx:
            print("Saving state dict and encoding")
            model_device = self.quant_sim.model.device
            self.quant_sim.model.to(device='cpu')

            output_dir = f'{self.export_path}/checkpoint'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            #Store the original dtypes of params so that we can restore them later
            orig_dtypes = {n: p.dtype for n, p in self.quant_sim.model.named_parameters()}
            self.quant_sim.model = self.quant_sim.model.float()

            tic = time.time()
            original_model = QuantizationSimModel.get_original_model(self.quant_sim.model)
            torch.save(original_model.state_dict(), os.path.join(output_dir, self.model_name + ".pth"))
            self.quant_sim.save_encodings_to_json(output_dir, self.model_name)
            print(f"Completed save state dict and encoding in {time.time() - tic} seconds")

            #Restore the dtypes of parameters
            for n, p in self.quant_sim.model.named_parameters():
                p.data = p.data.to(orig_dtypes[n])

            self.quant_sim.model.to(device=model_device)
        elif save_onnx:
            print("[WARNING] Checkpoint and encoding already exported with save_onnx enabled, skipping save_checkpoint")
        else:
            print("[WARNING] AimetTorch version does not support save_encodings_to_json, skipping save_checkpoint")

    def write_bert_encodings(self):
        for node in self.quant_sim.model.graph.nodes:
            if node.op == 'output':
                args = tuple(node.args)
                module_list = args[0][1]
        output_dir = os.path.join(self.output_dir, 'kv_encodings_tmp')

        file_index = 0
        for tuple_of_two in module_list:
            for offset, single_tuple in enumerate(tuple_of_two):
                file_name = single_tuple.name.split('_')[0] + str(file_index) + ".json"
                file_name = os.path.join(output_dir, file_name)
                try:
                    handle = open(file_name, 'r')
                except:
                    print(f'Not overriding encoding for module: {single_tuple.name}')
                    continue
                else:
                    with handle:
                        output_encodings = dict(json.load(handle))
                        output_quant_wrapper = getattr(self.quant_sim.model, single_tuple.name)
                    
                        encoding = libpymo.TfEncoding()
                        encoding.bw = output_encodings['bitwidth']
                        encoding.min = output_encodings['min']
                        encoding.max = output_encodings['max']
                        encoding.delta = output_encodings['scale']
                        encoding.offset = output_encodings['offset']
                        output_quant_wrapper.output_quantizers[0].encoding = encoding
                file_index += 1

    def write_bert_encodings_for_prepared_model(self):
        def _read_encodings(path):
            encodings_dict = dict(json.load(open(path, 'r')))
            encodings = libpymo.TfEncoding()
            encodings.bw = encodings_dict['bitwidth']
            encodings.min = encodings_dict['min']
            encodings.max = encodings_dict['max']
            encodings.delta = encodings_dict['scale']
            encodings.offset = encodings_dict['offset']
            return encodings, encodings_dict['symmetric']

        output_dir = os.path.join(self.output_dir, 'kv_encodings_tmp')

        num_layers = utils.extract_info_from_model_cfg(self.model_config)[0]
        num_heads = utils.extract_info_from_model_cfg(self.model_config)[1]

        for layer_num in range(num_layers):
            key_concat_index = 64
            value_concat_index = 96
            key_rope_idx = 32
            query_rope_idx = 0
            key_mul_idx = 128
            query_mul_idx = 0
            for head_num in range(num_heads):
                if head_num:
                    past_key_source_module = "model_layers_" + str(layer_num) + "_self_attn_MatMul_" + str(head_num)
                else:
                    past_key_source_module = "model_layers_" + str(layer_num) + "_self_attn_MatMul"
                past_value_source_module = "model_layers_" + str(layer_num) + "_self_attn_MatMul_" + str(head_num+32)

                past_key_file_name = os.path.join(output_dir, past_key_source_module + ".json")
                past_value_file_name = os.path.join(output_dir, past_value_source_module + ".json")

                key_source_encodings, is_key_symmetric = _read_encodings(past_key_file_name)
                value_source_encodings, is_value_symmetric = _read_encodings(past_value_file_name)

                # Overwrite encodings of MatMul from kvcache mode
                self._copy(past_key_source_module, "input", 1, key_source_encodings, is_key_symmetric)
                self._copy(past_value_source_module, "input", 1, value_source_encodings, is_value_symmetric)

                # Overwrite encodings from source to stack, Add and Sub in RoPE for key
                for op in ["Add", "Sub", "Concat"]:
                    self._copy(self._get_name(layer_num, op, key_rope_idx), "output", 0, key_source_encodings, is_key_symmetric)

                # Overwrite encodings of Mul in RoPE for key
                self._overwrite_k_rope_mul(layer_num, head_num, key_mul_idx)

                # Overwrite encodings from source to v_proj
                v_proj_module = "model_layers_" + str(layer_num) + "_self_attn_v_proj_sha_" + str(head_num) + "_Conv"
                self._copy(v_proj_module, "output", 0, value_source_encodings, is_value_symmetric, bitwidth=16)

                # Overwrite encodings in RoPE for query
                self._overwrite_q_rope(layer_num, head_num, query_mul_idx, query_rope_idx)

                key_concat_index += 1
                value_concat_index += 1
                key_rope_idx += 1
                query_rope_idx += 1
                key_mul_idx += 4
                query_mul_idx += 4

    def write_encodings_to_file(self, encoding, is_symmetric):
        encoding_dict = {}
        encoding_dict['bitwidth'] = encoding.bw
        encoding_dict['max'] = encoding.max
        encoding_dict['min'] = encoding.min
        encoding_dict['offset'] = encoding.offset
        encoding_dict['scale'] = encoding.delta
        encoding_dict['symmetric'] = is_symmetric
        return encoding_dict

    def _copy(self, name, type, idx, src_encodings, is_symmetric, bitwidth=None, factor=256):
        assert type in ["input", "output"]
        quant_wrapper = getattr(self.quant_sim.model, name)
        if type == "input":
            quantizers = quant_wrapper.input_quantizers if idx < 0 else [quant_wrapper.input_quantizers[idx]]
        else:
            quantizers = quant_wrapper.output_quantizers if idx < 0 else [quant_wrapper.output_quantizers[idx]]
        for quantizer in quantizers:
            quantizer.enabled = True
            quantizer.use_symmetric_encodings = is_symmetric
            if not quantizer.encoding:
                quantizer.encoding = libpymo.TfEncoding()
            quantizer.encoding.min = src_encodings.min
            quantizer.encoding.max = src_encodings.max
            if bitwidth and src_encodings.bw != bitwidth:
                quantizer.encoding.bw = bitwidth
                if bitwidth == 16:
                    quantizer.encoding.offset = src_encodings.offset * factor
                    quantizer.encoding.delta = src_encodings.delta / factor
                else:
                    quantizer.encoding.offset = src_encodings.offset / factor
                    quantizer.encoding.delta = src_encodings.delta * factor
            else:
                quantizer.encoding.bw = src_encodings.bw
                quantizer.encoding.offset = src_encodings.offset
                quantizer.encoding.delta = src_encodings.delta

    def _overwrite_q_rope(self, layer_num, head_num, query_mul_idx, query_rope_idx):
        # Overwrite encodings from q_proj to Muls in RoPE for query
        q_proj = "model_layers_" + str(layer_num) + "_self_attn_q_proj_sha_" + str(head_num) + "_Conv"
        q_proj_quant_wrapper = getattr(self.quant_sim.model, q_proj)
        q_proj_encodings = q_proj_quant_wrapper.output_quantizers[0].encoding
        is_q_proj_symmetric = q_proj_quant_wrapper.output_quantizers[0].use_symmetric_encodings
        # Overwrite mul_rr, mul_ii, mul_ri, mul_ir
        for i in range(4):
            self._copy(self._get_name(layer_num, "Mul", query_mul_idx + i), "input", 0, q_proj_encodings, is_q_proj_symmetric)

        # Overwrite encodings from 1st input of MatMul to output of Add, Sub, and Stack (Concat) in RoPE for key
        source_module_quant_wrapper = getattr(self.quant_sim.model, self._get_name(layer_num, "MatMul", query_rope_idx))
        source_encodings = source_module_quant_wrapper.input_quantizers[0].encoding
        is_source_symmetric = source_module_quant_wrapper.input_quantizers[0].use_symmetric_encodings
        for op in ["Add", "Sub", "Concat"]:
            self._copy(self._get_name(layer_num, op, query_rope_idx), "output", 0, source_encodings, is_source_symmetric)

    def _overwrite_k_rope_mul(self, layer_num, head_num, key_mul_idx):
        # Overwrite encodings from q_proj to Muls in RoPE for query
        k_proj = "model_layers_" + str(layer_num) + "_self_attn_k_proj_sha_" + str(head_num) + "_Conv"
        k_proj_quant_wrapper = getattr(self.quant_sim.model, k_proj)
        k_proj_encodings = k_proj_quant_wrapper.output_quantizers[0].encoding
        is_k_proj_symmetric = k_proj_quant_wrapper.output_quantizers[0].use_symmetric_encodings
        # Overwrite mul_rr, mul_ii, mul_ri, mul_ir
        for i in range(4):
            self._copy(self._get_name(layer_num, "Mul", key_mul_idx + i), "input", 0, k_proj_encodings, is_k_proj_symmetric)

    def _get_name(self, layer_num, op, idx):
        if idx:
            return "model_layers_" + str(layer_num) + "_self_attn_" + op + "_" + str(idx)
        else:
            return "model_layers_" + str(layer_num) + "_self_attn_" + op

    def write_kv_encodings_for_prepared_model(self):
        """
        Query:
            Source: 1st input of attn_MatMul_{0-31}
            Destination: output of attn_Add_{0-31}, output of attn_Concat_{0-31}, output of attn_Sub_{0-31}
        Muls in q_rope:
            Source: output of attn_q_proj_sha_{0-31}_Conv
            Destination: 1st input of attn_Mul_{0-127}
        Key:
            Source: 2nd input of attn_MatMul_{0-31}
            Destination: inputs and output of attn_Concat_{64-95}, output of attn_Add_{32-63}, output of attn_Concat_{32-63}, output of attn_Sub_{32-63}
        Muls in k_rope:
            Source: output of attn_k_proj_sha_{0-31}_Conv
            Destination: 1st input of attn_Mul_{128-255}
        Value:
            Source: 2nd input of attn_MatMul_{32-63}
            Destination: inputs and output of attn_Concat_{96-127}, output of v_proj_sha_{0-31}
        """

        output_dir = os.path.join(self.output_dir, 'kv_encodings_tmp')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            #Clean out old files
            with os.scandir(output_dir) as it:
                for entry in it:
                    os.remove(os.path.join(output_dir, entry.name))

        num_layers = utils.extract_info_from_model_cfg(self.model_config)[0]
        num_heads = utils.extract_info_from_model_cfg(self.model_config)[1]


        for layer_num in range(num_layers):
            key_concat_index = 64
            value_concat_index = 96
            key_rope_idx = 32
            query_rope_idx = 0
            key_mul_idx = 128
            query_mul_idx = 0
            for head_num in range(num_heads):
                # Identify the source of encodings
                past_key_source_module = self._get_name(layer_num, "MatMul", head_num)
                past_value_source_module = self._get_name(layer_num, "MatMul", head_num+32)

                key_source_quant_wrapper = getattr(self.quant_sim.model, past_key_source_module)
                key_source_encodings = key_source_quant_wrapper.input_quantizers[1].encoding
                is_key_symmetric = key_source_quant_wrapper.input_quantizers[1].use_symmetric_encodings
                value_source_quant_wrapper = getattr(self.quant_sim.model, past_value_source_module)
                value_source_encodings = value_source_quant_wrapper.input_quantizers[1].encoding
                is_value_symmetric = value_source_quant_wrapper.input_quantizers[1].use_symmetric_encodings

                # Overwrite encodings from source to inputs and output of Concat for past and present key/value
                key_concat_module = "model_layers_" + str(layer_num) + "_self_attn_Concat_" + str(key_concat_index)
                self._copy(key_concat_module, "input", -1, key_source_encodings, is_key_symmetric)
                self._copy(key_concat_module, "output", 0, key_source_encodings, is_key_symmetric)

                value_concat_module = "model_layers_" + str(layer_num) + "_self_attn_Concat_" + str(value_concat_index)
                self._copy(value_concat_module, "input", 0, value_source_encodings, is_value_symmetric)
                self._copy(value_concat_module, "input", 1, value_source_encodings, is_value_symmetric, bitwidth=16)
                self._copy(value_concat_module, "output", 0, value_source_encodings, is_value_symmetric)

                # Overwrite encodings from source to Stack, Add and Sub in RoPE for key
                for op in ["Add", "Sub", "Concat"]:
                    self._copy(self._get_name(layer_num, op, key_rope_idx), "output", 0, key_source_encodings, is_key_symmetric)

                # Overwrite encodings of Mul in RoPE for key
                self._overwrite_k_rope_mul(layer_num, head_num, key_mul_idx)

                # Overwrite encodings from source to v_proj
                v_proj_module = "model_layers_" + str(layer_num) + "_self_attn_v_proj_sha_" + str(head_num) + "_Conv"
                self._copy(v_proj_module, "output", 0, value_source_encodings, is_value_symmetric, bitwidth=16)

                # Overwrite encodings in RoPE for query
                self._overwrite_q_rope(layer_num, head_num, query_mul_idx, query_rope_idx)

                # Align SHA Concat
                attn_output_wrapper = getattr(self.quant_sim.model, self._get_name(layer_num, "Concat", 128))
                attn_output_encodings = attn_output_wrapper.output_quantizers[0].encoding
                is_attn_output_symmetric = attn_output_wrapper.output_quantizers[0].use_symmetric_encodings
                self._copy(self._get_name(layer_num, "MatMul", head_num+32), "output", 0, attn_output_encodings, is_attn_output_symmetric)

                file_name = past_key_source_module + ".json"
                file_name = os.path.join(output_dir, file_name)
                with open(file_name, "w") as file_handle:
                    encoding_dict = self.write_encodings_to_file(key_source_encodings, is_key_symmetric)
                    json.dump(encoding_dict, file_handle)
                file_name = past_value_source_module + ".json"
                file_name = os.path.join(output_dir, file_name)
                with open(file_name, "w") as file_handle:
                    encoding_dict = self.write_encodings_to_file(value_source_encodings, is_value_symmetric)
                    json.dump(encoding_dict, file_handle)

                key_concat_index += 1
                value_concat_index += 1
                key_rope_idx += 1
                query_rope_idx += 1
                key_mul_idx += 4
                query_mul_idx += 4

    def write_kv_encodings(self):
        num_layers = utils.extract_info_from_model_cfg(self.model_config)[0]
        num_heads = utils.extract_info_from_model_cfg(self.model_config)[1]
        module_list = []
        # Identify the source of encodings. If we have return_new_key_value_only set to TRUE,
        # use the encodings from the input quantizer of past_key_values. Else use the encodings
        # from the output quantizer of the model output
        if self.model_config.return_new_key_value_only:
            for layer_num in range(num_layers):
                for offset in range(2): #tuple of (k,v)
                    index = (layer_num * num_heads * 2) + offset
                    module_name = 'module_select'
                    if index:
                        module_name = module_name + "_" + str(index)
                    module_list.append(module_name)
        else:
            for node in self.quant_sim.model.graph.nodes:
                if node.op == 'output':
                    args = tuple(node.args)
                    module_list = args[0][1]
        
        
        layer_num = 0
        output_dir = os.path.join(self.output_dir, 'kv_encodings_tmp')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            #Clean out old files
            with os.scandir(output_dir) as it:
                for entry in it:
                    os.remove(os.path.join(output_dir, entry.name))

        # Apply the encodings to the correct destination. If return_new_key_value_only is TRUE,
        # the destination is the output quantizer of the model output. If it is false, the destination
        # is the input quantizer of the model input
        file_index = 0
        if self.model_config.return_new_key_value_only:
            for node in self.quant_sim.model.graph.nodes:
                if node.op == 'output':
                    args = tuple(node.args)
                    dest_module_list = args[0][1]

            src_module_index = 0
            for tuple_of_two in dest_module_list:
                for single_tuple in tuple_of_two:
                    input_quant_wrapper = getattr(self.quant_sim.model, \
                                                  module_list[src_module_index])
                    input_encoding = input_quant_wrapper.input_quantizers[0].encoding
                    output_quant_wrapper = getattr(self.quant_sim.model, single_tuple.name)
                    file_name = module_list[src_module_index].split('_')[0] + \
                                str(file_index) + ".json"
                    file_name = os.path.join(output_dir, file_name)
                    with open(file_name, "w") as file_handle:
                        encoding_dict = self.write_encodings_to_file(input_encoding)
                        json.dump(encoding_dict, file_handle)
                    file_index += 1
                    output_quant_wrapper.output_quantizers[0].encoding = input_encoding
                    src_module_index += 1

        else:
            for tuple_of_two in module_list:
                for offset, single_tuple in enumerate(tuple_of_two):
                    input_module_index = (layer_num * num_heads * 2) + offset
                    output_quant_wrapper = getattr(self.quant_sim.model, single_tuple.name)
                    output_encoding = output_quant_wrapper.output_quantizers[0].encoding
                    # write to file
                    file_name = single_tuple.name.split('_')[0] + str(file_index) + ".json"
                    file_name = os.path.join(output_dir, file_name)
                    with open(file_name, "w") as file_handle:
                        encoding_dict = self.write_encodings_to_file(output_encoding)
                        json.dump(encoding_dict, file_handle)
                    file_index += 1

                    # write to input encodings
                    for i in range(num_heads):
                        module_name = 'module_select'
                        if input_module_index:
                            module_name = module_name + "_" + str(input_module_index)
                        input_quant_wrapper = getattr(self.quant_sim.model, module_name)
                        input_quant_wrapper.input_quantizers[0].encoding = output_encoding
                        input_module_index += 2
                layer_num += 1

    def write_test_vectors(self, model, dataloader, num_batches, tokenizer, file_prefix, hook_intermediate_tensor=False, hook_config_file=None):
        print(f"Generating test vectors using {file_prefix} model")
        tic = time.time()
        if file_prefix not in ('fp', 'qt'):
            raise ValueError(f"Expected file_prefix to be one of ('fp', 'qt')")

        _, test_vectors = self.evaluate(model=model, iterations=num_batches, loader=dataloader, \
                                        tokenizer=tokenizer, metric="perplexity", get_test_vectors=True,
                                        hook_intermediate_tensor=hook_intermediate_tensor, hook_config_file=hook_config_file)

        output_dir = f"{self.export_path}/test_vectors"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for key, value in test_vectors.items():
            filename = os.path.join(output_dir, file_prefix + f"_{key}.pkl")
            with open(filename, 'wb') as file:
                pickle.dump({key: value}, file)
        print(f"Completed writing test vectors in {time.time() - tic} seconds")

    @staticmethod
    def register_forward_hooks_to_blocks(model, config_file):
        target_dict = json.load(open(config_file))
        targets = list(target_dict.keys())
        hook_handler, hook_recorder, resitered = [], utils.ForwardHook(), []
        for n, m in model.named_modules():
            if n in targets:
                hook_handler.append(m.register_forward_hook(hook_recorder.get_activation(n, target_dict[n])))
                resitered.append(n)
        print('Registered hooks for', resitered)
        return hook_handler, hook_recorder

    @staticmethod
    def register_forward_hooks_to_blocks_all(model):
        hook_handler_quant, hook_recorder_quant, resitered_quant = [], utils.ForwardHook(), []
        for n, m in model.named_modules():
            hook_handler_quant.append(m.register_forward_hook(hook_recorder_quant.get_activation(n, ["output"])))
            resitered_quant.append(n)
        print(f'Registered hooks for {resitered_quant}')
        return hook_handler_quant, hook_recorder_quant

    @staticmethod
    def remove_forward_hooks(hook_handler):
        for h in hook_handler: h.remove()

    def run_autoregressive_inference(self, tokenizer, args):
        print(f"Generating output tokens using {args.export_mode} mode")
        if args.export_mode == 'bertcache':
            self.run_model_with_bert_cache_mode(self.quant_sim.model, tokenizer, args)
        elif args.export_mode == 'kvcache':
            self.run_model_with_kv_cache_mode(self.quant_sim.model, tokenizer, args)
        else:
            raise ValueError("Unable to run autoregressive inference, export_mode should be bertcache or kvcache")

    def run_model_with_bert_cache_mode(self, model, tokenizer, args):
        dummy_input = utils.get_dummy_input_for_causal_llm("Hello, my dog is cute", tokenizer, args.device, "", model.config, preprocess=False)
        input_ids, attention_mask, position_ids = dummy_input['input_ids'], dummy_input['attention_mask'], dummy_input['position_ids']

        set_seed(args.seed)
        pad_size = torch.numel(torch.where(input_ids == 0)[0])
        for i in trange(pad_size):
            # Preprocess the position_ids and attention mask if needed
            if model.config.use_position_embedding_input:
                position_ids_processed = utils.RopeEmbedding(device=args.device).get_embedding(position_ids)
            else:
                position_ids_processed = position_ids
            if model.config.use_combined_mask_input:
                attention_mask_processed = utils.prepare_combined_attention_mask(attention_mask, input_ids.shape, args.device)
            else:
                attention_mask_processed = attention_mask

            if "position_ids_cos" in inspect.signature(model.forward).parameters or \
                    "past_key_0_h0_in" in inspect.signature(model.forward).parameters:
                model_inputs = {'input_ids':input_ids, 'attention_mask':attention_mask_processed, 'position_ids':position_ids_processed}
                outputs = model(*flatten_tensors(tuple(model_inputs.values())))

            else:
                outputs = model(input_ids, attention_mask=attention_mask_processed, position_ids=position_ids_processed)

            last_logits = outputs[0][:, -1, :]
            last_tokens = get_tokens(last_logits).view(1, -1)
            input_ids = torch.cat((input_ids[:,1:], last_tokens), dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask[:, 1:], attention_mask[:, -1:]], dim=-1)
            position_ids = torch.cat([position_ids[:, 1:], torch.tensor([[position_ids[0][-1]+1]], device=args.device)], dim=-1)
            if last_tokens[-1] == tokenizer.eos_token_id:
                break

        decoded_string = repr(tokenizer.decode(input_ids[0]))
        print(f"{'BERT mode':40}", decoded_string)

        if args.output_dir:
            output_dir = f"{args.output_dir}/output_tokens"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            filename = os.path.join(output_dir, "bertcache.pkl")
            with open(filename, 'wb') as file:
                pickle.dump(input_ids[0].clone().detach().to(device='cpu'), file)

            txtfilename = os.path.join(output_dir, "bertcache_decoded.txt")
            with open(txtfilename, 'w') as file:
                file.write(decoded_string)

    def run_model_with_kv_cache_mode(self, model, tokenizer, args):
        model.config.return_past = True

        dummy_input = utils.get_dummy_input_for_causal_llm("Hello, my dog is cute", tokenizer, args.device, "", model.config, pad=False, preprocess=False)
        input_ids = dummy_input['input_ids']
        all_tokens = input_ids[0]
        input_token_length = input_ids.shape[-1]

        num_layers, num_heads, max_length, n_embd = utils.extract_info_from_model_cfg(model.config)

        input_ids = torch.tensor([[0]], device=args.device)
        position_ids = torch.tensor([[0]], device=args.device)
        attention_mask = torch.tensor([[0]*max_length], device=args.device)

        cur_pos = 0
        input_ids[0][0] = all_tokens[cur_pos]
        position_ids[0][0] = cur_pos
        attention_mask[0][-cur_pos-1] = 1

        if hasattr(model, 'rwkv'):
            past = utils.get_dummy_state_kvcache(1, model.config, args.device)
        past_key_values = utils.get_dummy_kvcache(1, max_length-1, model.config, args.device)
        past_output_offset = 1

        set_seed(args.seed)
        for i in trange(max_length-1):
            # Preprocess the position_ids and attention mask if needed
            if model.config.use_position_embedding_input:
                position_ids_processed = utils.RopeEmbedding(device=args.device).get_embedding(position_ids)
            else:
                position_ids_processed = position_ids
            if model.config.use_combined_mask_input:
                attention_mask_processed = utils.prepare_combined_attention_mask(attention_mask, input_ids.shape, args.device)
            else:
                attention_mask_processed = attention_mask
            if hasattr(model, 'rwkv'):
                if type(model.rwkv) == str:
                    model_inputs = utils.rwkv_get_converted_dict(ids, past)
                else:
                    model_inputs = {'input_ids':ids, 'state': past}
                outputs = model(**model_inputs)
                if type(model.rwkv) == str:
                    past = outputs[1:] #state
                else:
                    past = outputs[1] #state
                if cur_pos < input_token_length-1:
                    cur_pos += 1
                    input_ids[0][0] = all_tokens[cur_pos]
                else:
                    last_logits = outputs[0][:, -1, :]
                    last_tokens = get_tokens(last_logits)
                    input_ids = last_tokens.view(1, -1)
                    all_tokens = torch.cat([all_tokens, input_ids[0]])
                    if all_tokens[-1] == tokenizer.eos_token_id:
                        break
            else:
                if "position_ids_cos" in inspect.signature(model.forward).parameters or \
                        "past_key_0_h0_in" in inspect.signature(model.forward).parameters:
                    model_inputs = {'input_ids':input_ids, 'attention_mask':attention_mask_processed, 'position_ids':position_ids_processed, 'past_key_values':past_key_values}
                    outputs = model(*flatten_tensors(tuple(model_inputs.values())))
                else:
                    try:
                        outputs = model(input_ids, attention_mask=attention_mask_processed, position_ids=position_ids_processed, past_key_values=past_key_values)
                    except:
                        # If past_key_values is passed as a positional argument
                        outputs = model(input_ids, past_key_values, attention_mask=attention_mask_processed, position_ids=position_ids_processed)
    
                new_past = outputs[past_output_offset] if isinstance(outputs[past_output_offset], tuple) else outputs[past_output_offset:]
    
                if cur_pos < input_token_length-1:
                    cur_pos += 1
                    input_ids[0][0] = all_tokens[cur_pos]
                    position_ids[0][0] = cur_pos
                    attention_mask[0][-cur_pos-1] = 1
    
                else:
                    last_logits = outputs[0][:, -1, :]
                    last_tokens = get_tokens(last_logits)
                    input_ids = last_tokens.view(1, -1)
                    position_ids = position_ids[:, -1:] + 1
                    attention_mask = torch.cat([attention_mask[:, 1:], attention_mask[:, -1:]], dim=-1)
    
                    all_tokens = torch.cat([all_tokens, input_ids[0]])
                    if all_tokens[-1] == tokenizer.eos_token_id:
                        break
    
                if is_nested_tuple(past_key_values) and is_flat_tuple(new_past):
                    # `past`, the output of the unprepared model, is a nested tuple of key-value tensors,
                    # whereas `new_past`, the output of the prepared model, is a flat tuple of key-value tensors.
                    # To concatenate `new_past` to `past`, we should first reform `new_past` into a nested tuple
                    # of the same structure as `past`.
                    assert len(new_past) == num_layers * num_heads * 2
                    new_past = tuple(
                        (new_past[2 * i * num_heads:2 * i * num_heads + num_heads],
                        new_past[2 * i * num_heads + num_heads:2 * (i + 1) * num_heads]) for i in range(num_layers)
                    )
                past_key_values = self._prepare_past_key_value(new_past, past_key_values, shift_size=1)

        decoded_string = repr(tokenizer.decode(all_tokens))
        print(f"{'Kvcache mode':40}", decoded_string)

        if args.output_dir:
            output_dir = f"{args.output_dir}/output_tokens"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            filename = os.path.join(output_dir, "kvcache.pkl")
            with open(filename, 'wb') as file:
                pickle.dump(all_tokens.clone().detach().to(device='cpu'), file)

            txtfilename = os.path.join(output_dir, "kvcache_decoded.txt")
            with open(txtfilename, 'w') as file:
                file.write(decoded_string)
