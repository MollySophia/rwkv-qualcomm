# coding=utf-8
# Copyright 2024 The RWKV team and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RWKV6 World model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_ninja_available,
    is_torch_cuda_available,
    logging,
)

import aimet_torch.elementwise_ops as op
from .configuration_rwkv6 import Rwkv6Config

logger = logging.get_logger(__name__)

class CustomTanh(torch.nn.Module):
    """ Add module for a functional add"""
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass routine for add op
        """
        out = torch.tanh(x)
        return out

class Rwkv6SelfAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        attention_hidden_size = config.attention_hidden_size
        self.attention_hidden_size = attention_hidden_size
        head_size = config.head_size
        num_heads = attention_hidden_size // head_size

        self.time_maa_x = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_w = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_k = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_v = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_r = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_g = nn.Parameter(torch.empty(1, 1, hidden_size))


        TIME_MIX_EXTRA_DIM = 32 # generate TIME_MIX for w,k,v,r,g
        if hidden_size == 4096: #7b
            TIME_MIX_EXTRA_DIM = 64
        self.time_mix_extra_dim = TIME_MIX_EXTRA_DIM

        self.time_maa_w1 = nn.Parameter(torch.empty(hidden_size, TIME_MIX_EXTRA_DIM*5))
        self.time_maa_w2 = nn.Parameter(torch.empty(5, TIME_MIX_EXTRA_DIM, hidden_size))

        self.time_decay = nn.Parameter(torch.empty(1, 1, attention_hidden_size))

        TIME_DECAY_EXTRA_DIM = 64
        if hidden_size == 4096: #7b
            TIME_DECAY_EXTRA_DIM = 128

        self.time_decay_w1 = nn.Parameter(torch.empty(hidden_size, TIME_DECAY_EXTRA_DIM))
        self.time_decay_w2 = nn.Parameter(torch.empty(TIME_DECAY_EXTRA_DIM, attention_hidden_size))

        self.time_faaaa = nn.Parameter(torch.empty(num_heads, config.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.output = nn.Linear(attention_hidden_size, hidden_size, bias=False)
        self.ln_x = nn.GroupNorm(num_heads, hidden_size, eps=(1e-5)*(config.head_size_divisor**2))

        self.scale_flag = 1
        ##Added new definition
        self.sub_shift              = op.Subtract()
        self.mul_time_maa           = op.Multiply()
        self.add_time_maa           = op.Add()
        self.matmul_time_maa_w1     = nn.Linear(hidden_size, TIME_MIX_EXTRA_DIM*5, bias=False)#op.MatMul()
        self.matmul_time_maa_w2_0   = nn.Linear(TIME_MIX_EXTRA_DIM, hidden_size, bias=False)#op.MatMul()
        self.matmul_time_maa_w2_1   = nn.Linear(TIME_MIX_EXTRA_DIM, hidden_size, bias=False)#op.MatMul()
        self.matmul_time_maa_w2_2   = nn.Linear(TIME_MIX_EXTRA_DIM, hidden_size, bias=False)#op.MatMul()
        self.matmul_time_maa_w2_3   = nn.Linear(TIME_MIX_EXTRA_DIM, hidden_size, bias=False)#op.MatMul()
        self.matmul_time_maa_w2_4   = nn.Linear(TIME_MIX_EXTRA_DIM, hidden_size, bias=False)#op.MatMul()
        self.add_time_maa_w         = op.Add()
        self.add_time_maa_k         = op.Add()
        self.add_time_maa_v         = op.Add()
        self.add_time_maa_r         = op.Add()
        self.add_time_maa_g         = op.Add()
        self.mul_time_maa_w         = op.Multiply()
        self.mul_time_maa_k         = op.Multiply()
        self.mul_time_maa_v         = op.Multiply()
        self.mul_time_maa_r         = op.Multiply()
        self.mul_time_maa_g         = op.Multiply()

        self.add_w_state0           = op.Add()
        self.add_k_state0           = op.Add()
        self.add_v_state0           = op.Add()
        self.add_r_state0           = op.Add()
        self.add_g_state0           = op.Add()


        self.ln_x_2                 = nn.LayerNorm(config.head_size, eps=(1e-5)*(config.head_size_divisor**2))
        #self.ln_x_1                = nn.InstanceNorm2d(hidden_size // config.head_size)
        self.ln_x_mul               = op.Multiply()
        self.ln_x_add               = op.Add()
        self.matmul_time_decay_w1   = nn.Linear(hidden_size, TIME_DECAY_EXTRA_DIM, bias=False)#op.MatMul()
        self.matmul_time_decay_w2   = nn.Linear(TIME_DECAY_EXTRA_DIM, hidden_size, bias=False)#op.MatMul()
        self.add_time_decay0        = op.Add()

        self.matmul_kv              = op.MatMul()
        self.mul_time_first         = op.Multiply()
        self.add_time_first         = op.Add()
        self.mul_scale_kv           = op.Multiply()
        self.matmul_rkv             = op.MatMul()
        self.mul_time_decay         = op.Multiply()
        self.add_time_decay1        = op.Add()
        self.mul_attention          = op.Multiply()

        self.tanh0                  = CustomTanh()
        self.tanh1                  = CustomTanh()
        self.silu0                  = op.CustomSiLU()
        self.split0                 = op.Split()
        self.exp0                   = op.Exponential()
        self.exp1                   = op.Exponential()
        self.neg                    = op.Neg()
        self.time_first             = nn.Parameter(torch.zeros((num_heads, config.head_size, 1)))
        self.reshape0               = op.Reshape()
        self.reshape1               = op.Reshape()
        self.reshape2               = op.Reshape()
        self.reshape3               = op.Reshape()
        self.reshape4               = op.Reshape()


    
    def rwkv6_linear_attention_cpu(self, receptance, key, value, time_decay, time_first, state):
        # For CPU fallback. Will be slower and probably take more memory than the custom CUDA kernel if not executed
        # within a torch.no_grad.
        batch, seq_length, _ = receptance.shape
        num_heads, head_size = time_first.shape
        key = key.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2).transpose(-2, -1)
        value = value.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2)
        receptance = receptance.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2).transpose(-2, -1)
        time_decay = torch.exp(-torch.exp(time_decay.float())).view(batch, seq_length, num_heads, head_size).permute(0, 2, 3, 1)
        time_first = time_first.float().reshape(-1, 1, 1).reshape(num_heads, -1, 1)
        self.time_first.data.copy_(time_first)
        out = torch.zeros_like(key).reshape(batch, seq_length, num_heads, head_size)
    
        for current_index in range(seq_length):
            current_receptance = receptance[:, :, :, current_index:current_index+1]
            current_key = key[:, :, :, current_index:current_index+1]
            current_value = value[:, :, current_index:current_index+1, :]
            current_time_decay = time_decay[:, :, :, current_index:current_index+1]
            attention_output = self.matmul_kv(current_key, current_value)
            time_first = self.mul_time_first(self.time_first, attention_output)
            time_first = self.add_time_first(time_first, state).permute(0, 1, 3, 2)
            out[:, current_index] = self.matmul_rkv(time_first, current_receptance).squeeze(3)
            state_time_decay = self.mul_time_decay(current_time_decay, state)
            state = self.add_time_decay1(attention_output, state_time_decay)
        return out, state
    
    def rwkv6_linear_attention_kv_1(self, receptance, key, value, time_decay, time_first, state):
        # For CPU fallback. Will be slower and probably take more memory than the custom CUDA kernel if not executed
        # within a torch.no_grad.
        batch, seq_length, _ = receptance.shape
        num_heads, head_size = time_first.shape
        key = self.reshape0(key.float(), (batch, num_heads, head_size, seq_length))
        value = self.reshape1(value.float(), (batch, num_heads, seq_length, head_size))
        receptance = self.reshape2(receptance.float(), (batch, num_heads, head_size, seq_length))
        time_decay = self.reshape3(self.exp1(self.neg(self.exp0(time_decay.float()))), (batch, num_heads, head_size, seq_length))
        time_first = time_first.float().reshape(-1, 1, 1).reshape(num_heads, -1, 1)
        self.time_first.data.copy_(time_first)

        attention_output = self.matmul_kv(key, value)
        #attention_output = self.mul_scale_kv(attention_output, self.scale_kv)
        time_first = self.mul_time_first(self.time_first, attention_output)

        time_first = self.add_time_first(time_first, state).permute(0, 1, 3, 2)
        out = self.matmul_rkv(time_first, receptance)
        state_time_decay = self.mul_time_decay(time_decay, state)
        state = self.add_time_decay1(attention_output, state_time_decay)
        return out, state
    
    def rwkv6_linear_attention(self,
        training,
        receptance,
        key,
        value,
        time_decay,
        time_first,
        state,
    ):
        no_cuda = any(t.device.type != "cuda" for t in [time_decay, time_first, receptance, key, value])
        # Launching the CUDA kernel for just one token will actually be slower (there is no for loop in the CPU version
        # in this case).
        one_token = key.size(1) == 1
        if not training or no_cuda or one_token:
            if one_token:
                return self.rwkv6_linear_attention_kv_1(
                    receptance, key, value, time_decay, time_first, state
                )            
            else:
                return self.rwkv6_linear_attention_cpu(
                    receptance, key, value, time_decay, time_first, state
                )
        else:
            batch, seq_length, _ = receptance.shape
            num_heads, head_size = time_first.shape
            key = key.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2) # B, T, H, K -> B, H, T, K
            value = value.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2) # B, T, H, K - > B, H, T, V
            receptance = receptance.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2) # B, H, T, K
            time_decay = -torch.exp(time_decay.float()).view(batch, seq_length, num_heads, head_size).permute(0, 2, 1, 3) # B, T, H, K -> B, H, T, K
            time_first = time_first.float().reshape(num_heads, head_size) # H, K
            out, state = fused_recurrent_rwkv6(receptance, key, value, time_decay, time_first, scale=1.0, initial_state=state, output_final_state=True)
            return out.transpose(1, 2), state

    def extract_key_value(self, hidden, state=None):
        # Mix hidden with the previous timestep to produce key, value, receptance
        if hidden.size(1) == 1 and state is not None:
            shifted = state[self.layer_id*3]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[self.layer_id*3][:, :]
        if len(shifted.size()) == 2:
            shifted = shifted.unsqueeze(1)

        x = hidden
        B, T, C = hidden.shape

        xx  = self.sub_shift(shifted, x)
        xxx = self.add_time_maa(x, self.mul_time_maa(xx, self.time_maa_x))
        self.matmul_time_maa_w1.weight.data.copy_(self.time_maa_w1.transpose(0, 1))
        xxx = self.matmul_time_maa_w1(xxx)
        xxx = self.tanh0(xxx)
        mw, mk, mv, mr, mg = self.split0(xxx, split_size_or_sections=self.time_mix_extra_dim, dim=-1)
        self.matmul_time_maa_w2_0.weight.data.copy_(self.time_maa_w2[0,:,:].transpose(0, 1))
        self.matmul_time_maa_w2_1.weight.data.copy_(self.time_maa_w2[1,:,:].transpose(0, 1))
        self.matmul_time_maa_w2_2.weight.data.copy_(self.time_maa_w2[2,:,:].transpose(0, 1))
        self.matmul_time_maa_w2_3.weight.data.copy_(self.time_maa_w2[3,:,:].transpose(0, 1))
        self.matmul_time_maa_w2_4.weight.data.copy_(self.time_maa_w2[4,:,:].transpose(0, 1))
        mw = self.add_time_maa_w(self.matmul_time_maa_w2_0(mw), self.time_maa_w)
        mk = self.add_time_maa_k(self.matmul_time_maa_w2_1(mk), self.time_maa_k)
        mv = self.add_time_maa_v(self.matmul_time_maa_w2_2(mv), self.time_maa_v)
        mr = self.add_time_maa_r(self.matmul_time_maa_w2_3(mr), self.time_maa_r)
        mg = self.add_time_maa_g(self.matmul_time_maa_w2_4(mg), self.time_maa_g)

        mw = self.add_w_state0(x, self.mul_time_maa_w(xx, mw))
        mk = self.add_k_state0(x, self.mul_time_maa_k(xx, mk))
        mv = self.add_v_state0(x, self.mul_time_maa_v(xx, mv))
        mr = self.add_r_state0(x, self.mul_time_maa_r(xx, mr))
        mg = self.add_g_state0(x, self.mul_time_maa_g(xx, mg))

        if self.scale_flag == 1:
            self.value.weight.data.copy_(self.value.weight.data/4)
            self.key.weight.data.copy_(self.key.weight.data/2)
            self.scale_flag = 0
            #print(torch.flatten(self.key.weight)[0:4])
        receptance = self.receptance(mr)
        key = self.key(mk)
        value = self.value(mv)
        gate = self.silu0(self.gate(mg))

        self.matmul_time_decay_w1.weight.data.copy_(self.time_decay_w1.transpose(0, 1))
        self.matmul_time_decay_w2.weight.data.copy_(self.time_decay_w2.transpose(0, 1))
        mw = self.tanh1(self.matmul_time_decay_w1(mw))
        time_decay = self.matmul_time_decay_w2(mw)
        time_decay = self.add_time_decay0(self.time_decay, time_decay)

        if state is not None:
            if hidden.size(1) == 1:
                state[self.layer_id*3] = hidden
            else:
                state[self.layer_id*3] = hidden[:,-1]

        return receptance, key, value, gate, time_decay, state

    def forward(self, hidden, state=None, use_cache=False, seq_mode=True):
        with torch.no_grad():
            receptance, key, value, gate, time_decay, state = self.extract_key_value(hidden, state=state)
    
            B,T,C = receptance.shape
            H, S = self.time_faaaa.shape
    
            layer_state = state[self.layer_id*3+1][:, :, :, :] if state is not None else None
            out, layer_state = self.rwkv6_linear_attention(
                self.training, receptance, key, value, time_decay, self.time_faaaa, layer_state,
            )
    
            if layer_state is not None:
                state[self.layer_id*3+1] = layer_state
    
            if hidden.size(1) == 1:
                #out = out.view(B, H, T*S) # 1x64x64
                out = self.reshape4(out, (B, H, T*S))
                self.ln_x_2.weight.fill_(1)
                self.ln_x_2.bias.fill_(0)
                out = self.ln_x_2(out).reshape(B, T, H*S)
                #out = self.ln_x_1(out).reshape(B, T, H*S)
                out = self.ln_x_mul(out, self.ln_x.weight)
                out = self.ln_x_add(out, self.ln_x.bias)
            else:
                out = out.reshape(B * T, H * S)
                out = F.group_norm(out, num_groups=H, weight=self.ln_x.weight.to(out.dtype), bias=self.ln_x.bias.to(out.dtype), eps=self.ln_x.eps).reshape(B, T, H * S)
            out = self.mul_attention(out.to(dtype=hidden.dtype), gate)
            out = self.output(out)
            return out, state


class Rwkv6FeedForward(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        # https://github.com/BlinkDL/RWKV-LM/blob/3db37a72356b736966ddd377268f02b80963af3f/RWKV-v4neo/train.py#L168
        intermediate_size = (
            config.intermediate_size
            if config.intermediate_size is not None
            else int((config.hidden_size * 3.5) // 32 * 32)
        )

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_maa_k = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_r = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

        #Add new Op def
        self.sub_shifted            = op.Subtract()
        self.mul_time_maa_k         = op.Multiply()
        self.add_time_maa_k         = op.Add()
        self.mul_time_maa_r         = op.Multiply()
        self.add_time_maa_r         = op.Add()
        self.relu                   = nn.ReLU()
        self.pow                    = op.Pow()
        self.sigmoid                = nn.Sigmoid()
        self.mul_rv                 = op.Multiply()

    def forward(self, hidden, state=None):
        if hidden.size(1) == 1 and state is not None:
            shifted = state[self.layer_id*3+2]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[self.layer_id*3+2][:, :]
        if len(shifted.size()) == 2:
            shifted = shifted.unsqueeze(1)

        delta_hidden_to_shifted = self.sub_shifted(shifted, hidden)
        key = self.add_time_maa_k(hidden, self.mul_time_maa_k(delta_hidden_to_shifted, self.time_maa_k))
        receptance = self.add_time_maa_r(hidden, self.mul_time_maa_r(delta_hidden_to_shifted, self.time_maa_r))

        key = self.pow(self.relu(self.key(key)), 2)
        value = self.value(key)
        receptance = self.sigmoid(self.receptance(receptance))

        if state is not None:
            if hidden.size(1) == 1:
                state[self.layer_id*3+2] = hidden
            else:
                state[self.layer_id*3+2] = hidden[:,-1]

        return self.mul_rv(receptance, value), state


class Rwkv6Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if layer_id == 0:
            self.pre_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.attention = Rwkv6SelfAttention(config, layer_id)
        self.feed_forward = Rwkv6FeedForward(config, layer_id)
        self.add_attention = op.Add()
        self.add_feed_forward = op.Add()

    def forward(self, hidden, state=None, use_cache=False, output_attentions=False, seq_mode=True):
        # if self.layer_id == 0:
        #     hidden = self.pre_ln(hidden)
        attention, state = self.attention(self.ln1(hidden), state=state, use_cache=use_cache, seq_mode=seq_mode)
        hidden = self.add_attention(hidden, attention)

        feed_forward, state = self.feed_forward(self.ln2(hidden), state=state)
        hidden = self.add_feed_forward(hidden, feed_forward)

        outputs = (hidden, state)
        return outputs

class Rwkv6PreTrainedModel(PreTrainedModel):
    config_class = Rwkv6Config
    base_model_prefix = "rwkv6"
    _no_split_modules = ["Rwkv6Block"]
    _keep_in_fp32_modules = ["time_decay", "time_first"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, Rwkv6SelfAttention):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size
            attention_hidden_size = module.attention_hidden_size
            head_size = module.config.head_size
            num_heads = attention_hidden_size // head_size

            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.time_maa_k.dtype,
                device=module.time_maa_k.device,
            )
            time_weight = time_weight[None, None, :]

            decay_speed = [
                -6.0 + 5.0 * (h / (attention_hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                for h in range(attention_hidden_size)
            ]
            decay_speed = torch.tensor(decay_speed, dtype=module.time_decay.dtype, device=module.time_decay.device)
            tmp = torch.tensor(
                [
                    (1.0 - (i / (attention_hidden_size - 1.0))) * ratio_0_to_1 + 0.1 * ((i + 1) % 3 - 1)
                    for i in range(attention_hidden_size)
                ],
                dtype=module.time_faaaa.dtype,
                device=module.time_faaaa.device,
            )

            with torch.no_grad():
                module.time_maa_x.data = 1.0 - torch.pow(time_weight, ratio_1_to_almost0)
                module.time_maa_w.data = 1.0 - torch.pow(time_weight, ratio_1_to_almost0)
                module.time_maa_k.data = 1.0 - torch.pow(time_weight, ratio_1_to_almost0)
                module.time_maa_v.data = 1.0 - (torch.pow(time_weight, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                module.time_maa_r.data = 1.0 - torch.pow(time_weight, 0.5 * ratio_1_to_almost0)
                module.time_maa_g.data = 1.0 - torch.pow(time_weight, 0.5 * ratio_1_to_almost0)

                TIME_MIX_EXTRA_DIM = 32 # generate TIME_MIX for w,k,v,r,g
                if hidden_size == 4096: #7b
                    TIME_MIX_EXTRA_DIM = 64
                module.time_maa_w1.data = torch.zeros(hidden_size, TIME_MIX_EXTRA_DIM*5, dtype=module.time_maa_w1.dtype, device=module.time_maa_w1.device).uniform_(-1e-4, 1e-4)
                module.time_maa_w2.data = torch.zeros(5, TIME_MIX_EXTRA_DIM, hidden_size, dtype=module.time_maa_w2.dtype, device=module.time_maa_w2.device).uniform_(-1e-4, 1e-4)

                TIME_DECAY_EXTRA_DIM = 64
                if hidden_size == 4096: #7b
                    TIME_DECAY_EXTRA_DIM = 128
                module.time_decay_w1.data = torch.zeros(hidden_size, TIME_DECAY_EXTRA_DIM, dtype=module.time_decay_w1.dtype, device=module.time_decay_w1.device).uniform_(-1e-4, 1e-4)
                module.time_decay_w2.data = torch.zeros(TIME_DECAY_EXTRA_DIM, attention_hidden_size, dtype=module.time_decay_w2.dtype, device=module.time_decay_w2.device).uniform_(-1e-4, 1e-4)

                module.time_decay.data = decay_speed.reshape(1, 1, attention_hidden_size)#decay_speed.reshape(num_heads, head_size)
                module.time_faaaa.data = tmp.reshape(num_heads, head_size)

        elif isinstance(module, Rwkv6FeedForward):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size

            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.time_maa_k.dtype,
                device=module.time_maa_k.device,
            )
            time_weight = time_weight[None, None, :]

            with torch.no_grad():
                module.time_maa_k.data = 1.0 - torch.pow(time_weight, ratio_1_to_almost0)
                module.time_maa_r.data = 1.0 - torch.pow(time_weight, ratio_1_to_almost0)


@dataclass
class Rwkv6Output(ModelOutput):
    """
    Class for the RWKV model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    state: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Rwkv6CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    state: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Rwkv6Model(Rwkv6PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Rwkv6Block(config, layer_id=idx) for idx in range(config.num_hidden_layers)])
        self.ln_out = nn.LayerNorm(config.hidden_size)

        self.layers_are_rescaled = False
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, Rwkv6Output]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # return_dict = False

        if self.training == self.layers_are_rescaled and (
            self.embeddings.weight.dtype == torch.float16 or self.embeddings.weight.dtype == torch.bfloat16
        ):
            self._rescale_layers()

        # To fix aimet jit trace issue,  use inputs_embeds as placeholder, so need to do embeddings always
        inputs_embeds = self.embeddings(input_ids)
        '''
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        '''

        if state is None:
            def _cache(shape):
                return torch.zeros(shape).to(device=inputs_embeds.device)
            batch_size = inputs_embeds.size(0)
            head_size = self.config.head_size
            num_heads = self.config.attention_hidden_size // head_size
            if batch_size == 1:
                state_0 = (batch_size, 1, self.config.hidden_size)  # (1, 2560)
                state_1 = (batch_size, num_heads, head_size, head_size)  # (1, 40, 64, 64)
                state_2 = (batch_size, 1, self.config.hidden_size) # (1, 2560)
            else:
                state_0 = (batch_size, self.config.hidden_size)  # (1, 2560)
                state_1 = (batch_size, num_heads, head_size, head_size)  # (1, 40, 64, 64)
                state_2 = (batch_size, self.config.hidden_size) # (1, 2560)
 
            state = []
            for i in range(0, self.config.num_hidden_layers):
                state += [_cache(state_0), _cache(state_1), _cache(state_2)]
        '''
        if state is None:
            state = []
            head_size = self.config.head_size
            num_heads = self.config.attention_hidden_size // head_size
            state_attn_x = torch.zeros(
                    (inputs_embeds.size(0), self.config.hidden_size, self.config.num_hidden_layers),
                    dtype=inputs_embeds.dtype,
                    requires_grad=False,
                    device=inputs_embeds.device,
                ).contiguous()
            state_attn_kv = torch.zeros(
                    (
                        inputs_embeds.size(0),
                        num_heads,
                        head_size,
                        head_size,
                        self.config.num_hidden_layers,
                    ),
                    dtype=torch.float32,
                    requires_grad=False,
                    device=inputs_embeds.device,
                ).contiguous()
            state_ffn_x = torch.zeros(
                    (inputs_embeds.size(0), self.config.hidden_size, self.config.num_hidden_layers),
                    dtype=inputs_embeds.dtype,
                    requires_grad=False,
                    device=inputs_embeds.device,
                ).contiguous()
            state.append(state_attn_x)
            state.append(state_attn_kv)
            state.append(state_ffn_x)
        '''

        seq_mode = inputs_embeds.shape[1] > 1
        hidden_states = inputs_embeds

        for idx, block in enumerate(self.blocks):
            hidden_states, state = block(
                hidden_states, state=state, use_cache=use_cache, output_attentions=output_attentions, seq_mode=seq_mode
            )
            if (
                self.layers_are_rescaled
                and self.config.rescale_every > 0
                and (idx + 1) % self.config.rescale_every == 0
            ):
                hidden_states = hidden_states / 2

        hidden_states = self.ln_out(hidden_states)

        if not return_dict:
            return(hidden_states, state)

        return Rwkv6Output(
            last_hidden_state=hidden_states,
            state=state,
        )

    def _rescale_layers(self):
        # Layers should be rescaled for inference only.
        if self.layers_are_rescaled == (not self.training):
            return
        if self.config.rescale_every > 0:
            with torch.no_grad():
                for block_id, block in enumerate(self.blocks):
                    block.attention.output.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                    block.feed_forward.value.weight.mul_(2 ** int(block_id // self.config.rescale_every))

        self.layers_are_rescaled = not self.training

class Rwkv6ForCausalLM(Rwkv6PreTrainedModel):
    _tied_weights_keys = ["head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.rwkv = Rwkv6Model(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, state=None, inputs_embeds=None, **kwargs):
        # only last token for inputs_ids if the state is passed along.
        if state is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and state is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["state"] = state
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, Rwkv6CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        with torch.no_grad():
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            # return_dict = False
    
            outputs = self.rwkv(
                input_ids,
                inputs_embeds=inputs_embeds,
                state=state,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]

            logits = self.head(hidden_states)
            #print(torch.argmax(logits, dim=-1))
    
            loss = None
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(logits.device)
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
            if not return_dict:
                output = (logits,) + outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return Rwkv6CausalLMOutput(
                loss=loss,
                logits=logits,
                state=outputs.state,
                #hidden_states=outputs.hidden_states,
                #attentions=outputs.attentions,
            )