import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from aimet_torch.qc_quantize_op import QcQuantizeWrapper
import aimet_common.libpymo as libpymo

from quantizers.base_quantizer import LLMQuantizer
from utils.model_utils import get_dummy_state_kvcache

class ActMSEQuantizer(LLMQuantizer):
    def __init__(self, model, args, model_config):
        super().__init__(model, args, model_config)
        
        self.true_sequential = True
        self.save_param_enc = True
        self.num_cands = args.num_cands
        self.input_symmetry = args.input_symmetry
        self.exceptions_file = args.exceptions_file.split("/")[-1].replace(".json", "")
        self.act_mse_loss_type = args.act_mse_loss_type
        self.param_bw = args.parameter_bit_width

    def prepare_quantsim(self, dummy_input, args, train_dataloader, tokenizer):
        super().prepare_quantsim(dummy_input, args, train_dataloader, tokenizer)
        if args.do_actmse:
            self.compute_activation_mse(train_dataloader, args)
        if self.save_param_enc:
            self.export_param_encodings(args)

    def export_param_encodings(self, args):
        # make options
        if args.do_actmse:
            fname = f"{self.model_name}_{self.act_mse_loss_type}_{self.exceptions_file}_{self.input_symmetry}_torch_w{self.param_bw}.encodings"
        else:
            fname = f"{self.model_name}_{args.quant_scheme}_torch_w{self.param_bw}.encodings"
        outpath = os.path.join(self.export_path, fname)
        print("encodings saved at:")
        print(outpath)
        os.path.exists(self.export_path) or os.makedirs(self.export_path)
        self.save_param_encodings(outpath)

    @torch.no_grad()
    def compute_activation_mse(self, data_loader, args):
        # Both self.model (FP) & self.quant_sim.model (quantized) will be used for optimization
        assert not self.in_place
        fp_blocks, qt_blocks = self.get_blocks()
        batch_data = []
        for step, batch in enumerate(data_loader):
            for k in batch.keys():
                batch[k] = batch[k].to(self.quant_sim.model.device)
            if step == self.num_calibration_batches:
                break
            batch_data.append(batch)

        def get_block_inputs(blocks, model):
            cache = {"i":0, "state": None}
            block_inputs = torch.zeros(
                (
                    args.num_calibration_batches,
                    args.per_device_calib_batch_size,
                    args.block_size,
                    model.args.n_embd,
                ), device=model.device,
            )
            blocks[0] = InputCatcher(blocks[0], cache, block_inputs)
            for batch in batch_data:
                try:
                    state = get_dummy_state_kvcache(1, model.args, model.device)
                    input = {'in0': batch['input_ids'], 'state': state}
                    _, state = model(**input)
                except ValueError:
                    pass
            blocks[0] = blocks[0].module
            return cache, block_inputs

        device = self.quant_sim.model.device
        self.quant_sim.model = self.quant_sim.model.to(device)
        _, qt_inputs = get_block_inputs(qt_blocks, self.quant_sim.model)
        self.quant_sim.model= self.quant_sim.model.to("cpu")

        self.model = self.model.to(device)
        cache, fp_inputs = get_block_inputs(fp_blocks, self.model)
        self.model = self.model.to("cpu")

        del batch_data
        torch.cuda.empty_cache()

        state = cache["state"]
        fp_outputs, qt_outputs = torch.zeros_like(fp_inputs), torch.zeros_like(qt_inputs)

        for block_idx in range(len(qt_blocks)):
            t0 = time.time()
            print(f"Quantizing layer {block_idx+1}/{len(qt_blocks)}...")
            print('+------------------+--------------+------------+-----------+-------+')
            #print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
            #print('+==================+==============+============+===========+=======+')
            #print(torch.cuda.memory_summary(device=0))

            fp_modules = self.find_layers_from_block(fp_blocks[block_idx], from_quant_sim=False)
            qt_modules = self.find_layers_from_block(qt_blocks[block_idx], from_quant_sim=True)

            fp_blocks[block_idx] = fp_blocks[block_idx].to(device)
            qt_blocks[block_idx] = qt_blocks[block_idx].to(device)

            if self.true_sequential:
                sequence = self.get_sequential_layers()
            else:
                sequence = [list(qt_modules.keys())]

            # optimize each module
            for names in sequence:
                def get_layer_inputs(module, block, inputs):
                    hook_recorder = LinearHook()
                    handler = module.register_forward_hook(hook_recorder.forward_hook())

                    for i in range(self.num_calibration_batches):
                        try:
                            block(inputs[i], state)
                        except ValueError:
                            pass
                    layer_inputs = torch.stack(hook_recorder.data)
                    handler.remove()
                    return cache, layer_inputs

                for name in names:
                    print(f"optimizing layer block[{block_idx}].{name}")
                    _, xq = get_layer_inputs(qt_modules[name], qt_blocks[block_idx], qt_inputs)
                    _, x = get_layer_inputs(fp_modules[name], fp_blocks[block_idx], fp_inputs)

                    if self.input_symmetry == "asym":
                        self.optimize_module(qt_modules[name], x, xq)
                    elif self.input_symmetry == "symfp":
                        self.optimize_module(qt_modules[name], x, x)
                    elif self.input_symmetry == "symqt":
                        self.optimize_module(qt_modules[name], xq, xq)
                    else:
                        raise ValueError

                    """
                    # test
                    out_1 = F.linear(x, qt_modules[name].weight, qt_modules[name].bias)
                    out_2 = qt_modules[name](x)
                    qt_modules[name].param_quantizers["weight"].enabled = False
                    qt_modules[name].output_quantizers[0].enabled = False
                    out_3 = qt_modules[name](x)
                    out_4 = fp_modules[name](x)
                    """

                for i in range(self.num_calibration_batches):
                    qt_outputs[i] = qt_blocks[block_idx](qt_inputs[i], state=state)[0]

            for i in range(self.num_calibration_batches):
                fp_outputs[i] = fp_blocks[block_idx](fp_inputs[i], state=state)[0]

            qt_inputs, qt_outputs = qt_outputs, qt_inputs
            fp_inputs, fp_outputs = fp_outputs, fp_inputs

            fp_blocks[block_idx] = fp_blocks[block_idx].to("cpu")
            qt_blocks[block_idx] = qt_blocks[block_idx].to("cpu")
            torch.cuda.empty_cache()

            t_block = time.time() - t0
            print(f"block optimization took {t_block} seconds...")

        del self.model
        self.quant_sim.model.to(device)
        self.model = self.quant_sim.model


    def find_layers_from_block(self, block, from_quant_sim=True):
        modules = {}
        for name, module in block.named_modules():
            if from_quant_sim:
                if isinstance(module, QcQuantizeWrapper) and isinstance(module._module_to_wrap, torch.nn.Linear):
                    modules[name] =  module
            else:
                if isinstance(module, torch.nn.Linear):
                    modules[name] = module
        return modules

    def get_sequential_layers(self):
        # rwkv v6
        layers = [
            ["att.key"], ["att.value"], ["att.receptance"], ["att.gate"], 
            ["att.output"],
            ["ffn.key"], ["ffn.receptance"], ["ffn.value"],
        ]
        return layers

    def get_candidates(self, n, _max, _min=None):
        candidates = []
        if _min is not None:
            # asym case
            for i in range(n):
                # candidate pairs
                max_cand = torch.tensor(_max / n * (i+1))
                min_cand = torch.tensor(_min / n * (i+1))
                candidates.append((max_cand, min_cand))

                """
                # combination of asym cands
                max_cand = torch.tensor(_max /  n * (i+1))
                for j in range(n):
                    min_cand = torch.tensor(_min / n * (j+1))
                    candidates.append((max_cand, min_cand))
                """
        else:
            # symmetric case
            for i in range(n):
                max_cand = torch.tensor(_max / n * (i+1))
                min_cand = -max_cand
                candidates.append((max_cand, min_cand))
        return candidates


    @torch.no_grad()
    def optimize_module(self, qt_layer, x, xq):
        if qt_layer.param_quantizers["weight"].bitwidth >= 16:
            return

        weight = qt_layer.weight
        if qt_layer.param_quantizers["weight"].use_symmetric_encodings:
            per_channel_max = torch.max(weight.abs(), dim=1)[0].detach()
            per_channel_min = None
        else:
            per_channel_max = torch.max(weight, dim=1)[0].detach()
            per_channel_min = torch.min(weight, dim=1)[0].detach()

        candidates = self.get_candidates(self.num_cands, per_channel_max, per_channel_min)

        xw = torch.zeros(
            (
                x.shape[0], x.shape[1], x.shape[2], len(per_channel_max),
            ), device=weight.device
        )

        for batch_idx in range(self.num_calibration_batches):
            xw[batch_idx] = F.linear(x[batch_idx], weight, qt_layer.bias)

        if self.act_mse_loss_type == "mse":
            loss_fn = torch.nn.functional.mse_loss
        elif self.act_mse_loss_type == "l1":
            loss_fn = torch.nn.functional.l1_loss
        else:
            # minimizing negative SQNR is equivalent to maximizing SQNR
            loss_fn = neg_sqnr

        loss = []
        for max_cand, min_cand in candidates:
            self.set_quantizer_encodings(qt_layer.param_quantizers["weight"], min_cand, max_cand)
            #wq = qt_layer.param_quantizers["weight"].quantize_dequantize(weight, max_cand, min_cand)
            wq = qt_layer.param_quantizers["weight"].quantize_dequantize(weight, libpymo.RoundingMode.ROUND_NEAREST)
            _loss = torch.zeros(len(per_channel_max), device=weight.device)
            for batch_idx in range(self.num_calibration_batches):
                xqwq = F.linear(xq[batch_idx], wq, qt_layer.bias)
                _loss += loss_fn(xqwq, xw[batch_idx], reduction="none").sum((0, 1))
            loss.append(_loss)

        cand_args = torch.stack(loss).min(0, keepdim=True)[1]

        # print to debug
        # print(cand_args.squeeze(0)[:20])
        max_sol = torch.stack([_max for _max, _ in candidates]).gather(0, cand_args)[0]
        min_sol = torch.stack([_min for _, _min in candidates]).gather(0, cand_args)[0]

        # set per channel encodings
        self.set_quantizer_encodings(qt_layer.param_quantizers["weight"], min_sol, max_sol)

    def set_quantizer_encodings(self, quantizer, min_cand, max_cand):
        tensor = torch.stack([min_cand, max_cand], dim=-1)
        quantizer.reset_encoding_stats()
        quantizer.update_encoding_stats(tensor)
        quantizer.compute_encoding()

    def _set_module_param_encodings(self, module, min_cand, max_cand):
        tensor = torch.stack([min_cand, max_cand], dim=-1).to(module.weight.device)
        if self.quant_scheme == "tf":
            module.param_quantizers["weight"].reset_encoding_stats()
            module.param_quantizers["weight"].update_encoding_stats(tensor)
            module.param_quantizers["weight"].compute_encoding()


    @torch.no_grad()
    def optimize_output(self, qt_layer, x, xq):
        """
        grid search for output quantizer
        """
        output_quantizer = qt_layer.output_quantizers[0]
        weight = qt_layer.weight
        per_channel_max = torch.max(weight, dim=1)[0].detach()
        if output_quantizer.encoding is None:
            return

        candidates = self.get_candidates(self.num_cands,
            output_quantizer.encoding.max, output_quantizer.encoding.min)

        xw = torch.zeros(
            (
                x.shape[0], x.shape[1], x.shape[2], len(per_channel_max),
            ), device=weight.device
        )

        for batch_idx in range(self.num_calibration_batches):
            xw[batch_idx] = F.linear(x[batch_idx], weight, qt_layer.bias)

        if self.act_mse_loss_type == "mse":
            loss_fn = torch.nn.functional.mse_loss
        elif self.act_mse_loss_type == "l1":
            loss_fn = torch.nn.functional.l1_loss

        loss = []
        for max_cand, min_cand in candidates:
            self.set_quantizer_encodings(output_quantizer, min_cand, max_cand)

            _loss = torch.zeros(1, device=weight.device)
            for batch_idx in range(self.num_calibration_batches):
                xqwq = qt_layer(xq[batch_idx])
                _loss += loss_fn(xqwq, xw[batch_idx], reduction="none").sum()
            loss.append(_loss)

        idx = torch.tensor(loss).argmin()
        min_sol = candidates[idx][1]
        max_sol = candidates[idx][0]

        # logger.debug
        print(f"act sol: {min_sol}, {max_sol} | {idx} / {self.num_cands}")
        self.set_quantizer_encodings(output_quantizer, min_sol, max_sol)


    def get_blocks(self):
        fp_blocks = self.model.blocks
        qt_blocks = self.quant_sim.model.blocks

        return fp_blocks, qt_blocks


class InputCatcher(nn.Module):
    def __init__(self, module, cache, block_inputs):
        super().__init__()
        self.module = module
        self.cache = cache
        self.block_inputs = block_inputs

    def forward(self, inp, state):
        self.block_inputs[self.cache["i"]] = inp
        self.cache["i"] += 1
        self.cache["state"] = state
        # only pass the inputs until here
        raise ValueError

class LinearHook:
    def __init__(self):
        self.data = []

    def forward_hook(self):
        def hook(model, inputs, outputs):
            if isinstance(inputs, tuple):
                self.data.append(inputs[0])
            raise ValueError
        return hook

def neg_sqnr(pred, target, eps=1e-10, reduction="none"):
    quant_error = target - pred
    exp_noise = torch.mean(quant_error ** 2, (0, 1), keepdim=True) + eps
    exp_signal = torch.mean(target ** 2, (0, 1), keepdim=True)
    sqnr = exp_signal / exp_noise
    sqnr_db = 10 * torch.log10(sqnr)
    return -sqnr_db
