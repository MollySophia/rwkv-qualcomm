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
try:
    from aimet_torch.pro.quantsim import QuantizationSimModel
except:
    from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.quantsim import load_encodings_to_sim, _get_encoding_by_quantizer
from aimet_torch.onnx_utils import OnnxExportApiArgs
from aimet_torch.qc_quantize_op import QcQuantizeWrapper, QcQuantizeOpMode
from aimet_torch import utils as aimet_utils

from .exceptions import ExceptionConfigurator
import utils.model_utils as utils

from importlib.metadata import version as impLib_version
from packaging import version

# from aimet_torch import onnx_utils
# onnx_utils.EXPORT_TO_ONNX_DIRECT = True

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

    def create_quantsim(self, dummy_input):
        if type(dummy_input) is dict:
            model_inputs = tuple([dummy_input[input] for input in dummy_input.keys()])
        else:
            model_inputs = tuple(dummy_input)
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

        # debug
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
        max_length = tokenizer.model_max_length
        past_output_offset = 1
        rope = None

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

                if do_eval:
                    # kvcache ppl score evaluation
                    outputs = self.compute_kv_output_logits(model, max_length, past_output_offset, input_ids,
                                                            position_ids, rope, _adjust_inputs)
                else:
                    num_loops = 20
                    batch_size = input_ids.shape[0]
                    num_input_tokens = 1

                    cur =  max_length - num_input_tokens * num_loops
                    ids = input_ids[:, :cur]

                    past = utils.get_dummy_state_kvcache(batch_size, self.model_config, model.device)

                    model_inputs = {'in0': ids, 'state': past}
                    _, past = model(**model_inputs)

                    for _ in range(num_loops):
                        ids =  input_ids[:, cur:cur+num_input_tokens]
                        cur += num_input_tokens

                        if ids.shape[1] == 0:
                            break

                        model_inputs = {'in0': ids, 'state': past}
                        _, past = model(**model_inputs)

                if get_test_vectors:
                    converted_flag = 0
                    if type(model.rwkv) == str:
                        converted_flag = 1
                    self.rwkv_add_to_test_vectors(test_vectors, model_inputs, outputs,
                            hook_recorder.data if hook_intermediate_tensor else None, converted_flag)

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
                outputs = model(*model_inputs)
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
        # cpu_dummy_input = tuple([i.cpu() for i in dummy_input])
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
