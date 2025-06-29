import torch
import torch.nn as nn
import torch.nn.functional as F
from aimet_torch.v2.nn.modules.custom import *
from aimet_torch.v2.quantization.tensor import DequantizedTensor

custom_norm_wrapper_src = """
#include <torch/extension.h>
#include <torch/script.h>

torch::Tensor l2norm(torch::Tensor x) {
    return x / (x.norm(2, -1, true) + 1e-6);
}

TORCH_LIBRARY(customop, m) {
    m.def("l2norm", &l2norm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
"""

_ = torch.utils.cpp_extension.load_inline(
    name='extension', cpp_sources=[custom_norm_wrapper_src])

class L2Norm(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.ops.customop.l2norm(x)

class Wkv7Op(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, r, w, k, v, a, b, state2):
        return torch.ops.rwkv.wkv7(r, w, k, v, a, b, state2)

class Wkv7OutputX(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.ops.rwkv.wkv7_output_x(input)

class Wkv7OutputState(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.ops.rwkv.wkv7_output_state(input)

class Wkv7(nn.Module):
    def __init__(self, num_heads, head_size, custom_wkv=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.custom_wkv = custom_wkv

        if not self.custom_wkv:
            self.apply_time_decay = Multiply()
            self.matmul_sa = MatMul()
            self.matmul_sab = MatMul()
            self.matmul_kv = MatMul()
            self.matmul_r = MatMul()
            self.add_kv = Add()
            self.add_sab = Add()

        self.reshape_r = Reshape()
        self.reshape_w = Reshape()
        self.reshape_k = Reshape()
        self.reshape_v = Reshape()
        self.reshape_a = Reshape()
        self.reshape_b = Reshape()
        self.reshape_x = Reshape()

        # self.gather_state = CustomGather()

        if custom_wkv:
            # for adding the onnx nodes
            from rwkv_src.wkv_custom import wkv_c_impl_src
            module = torch.utils.cpp_extension.load_inline(
                    name='extension', cpp_sources=[wkv_c_impl_src])
            # self.wkv_state_func = torch.ops.rwkv.wkv7_state
            # self.wkv_output_func = torch.ops.rwkv.wkv7_output
            self.wkv = Wkv7Op()
            self.wkv_output_x = Wkv7OutputX()
            self.wkv_output_state = Wkv7OutputState()

    def forward(self, seq_length, r, w, k, v, a, b, state2):
        if self.custom_wkv:
            r = self.reshape_r(r, [seq_length, self.num_heads, 1, self.head_size])
            w = self.reshape_w(w, [seq_length, self.num_heads, 1, self.head_size])
            k = self.reshape_k(k, [seq_length, self.num_heads, 1, self.head_size])
            v = self.reshape_v(v, [seq_length, self.num_heads, 1, self.head_size])
            a = self.reshape_a(a, [seq_length, self.num_heads, 1, self.head_size])
            b = self.reshape_b(b, [seq_length, self.num_heads, 1, self.head_size])

            # state2_out = self.wkv_state(w, k, v, a, b, state2)
            # x = self.wkv_output(r, state2_out)
            output = self.wkv(r, w, k, v, a, b, state2)
            x = self.wkv_output_x(output)
            state2_out = self.wkv_output_state(output)
            # if seq_length != 1:
            #     state2_out = self.gather_state(state2_out, torch.LongTensor([-1]).to(state2_out.device), 0)
            x = self.reshape_x(x, [seq_length, self.num_heads, 1, self.head_size])
        else:
            if seq_length == 1:
                r = r.view(self.num_heads, self.head_size, 1)
                v = v.view(self.num_heads, self.head_size, 1)
                k = k.view(self.num_heads, 1, self.head_size)
                w = w.view(self.num_heads, 1, self.head_size)
                b = b.view(self.num_heads, 1, self.head_size)
                a = a.view(self.num_heads, self.head_size, 1)

                kv = self.matmul_kv(v, k)
                state2_out = self.add_kv(self.add_sab(self.apply_time_decay(state2, w), self.matmul_sab(self.matmul_sa(state2, a), b)), kv)
                x = self.matmul_r(state2_out, r).view(seq_length, self.num_heads, 1, self.head_size)
            else:
                r = r.view(seq_length, self.num_heads, self.head_size, 1)
                v = v.view(seq_length, self.num_heads, self.head_size, 1)
                k = k.view(seq_length, self.num_heads, 1, self.head_size)
                w = w.view(seq_length, self.num_heads, 1, self.head_size)
                b = b.view(seq_length, self.num_heads, 1, self.head_size)
                a = a.view(seq_length, self.num_heads, self.head_size, 1)
                kv = self.matmul_kv(v, k)
                x = torch.zeros(seq_length, self.num_heads, self.head_size, 1, device=k.device, dtype=kv.dtype)
                for i in range(seq_length):
                    state2 = self.apply_time_decay(state2, w[i, :, :, :]) + (state2 @ a[i, :, :, :] @ b[i, :, :, :]) + kv[i, :, :, :]
                    x[i, :, :, :] = state2 @ r[i, :, :, :]
                state2_out = state2
                x = x.view(seq_length, self.num_heads, 1, self.head_size)

        return x, state2_out


class Rwkv7SelfAttention(nn.Module):
    def __init__(self, state_dict, hidden_size, head_size, layer_id=0, rescale_layer=0, custom_wkv=False):
        super().__init__()
        prefix = f'blocks.{layer_id}.att.'
        self.layer_id = layer_id
        self.num_heads = hidden_size // head_size
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.custom_wkv = custom_wkv

        self.D_DECAY_LORA = state_dict[prefix + 'w1'].shape[-1]
        self.D_AAA_LORA = state_dict[prefix + 'a1'].shape[-1]
        self.D_GATE_LORA = state_dict[prefix + 'g1'].shape[-1]

        self.x_r = nn.Parameter(state_dict[prefix + 'x_r'])
        self.x_w = nn.Parameter(state_dict[prefix + 'x_w'])
        self.x_k = nn.Parameter(state_dict[prefix + 'x_k'])
        self.x_v = nn.Parameter(state_dict[prefix + 'x_v'])
        self.x_a = nn.Parameter(state_dict[prefix + 'x_a'])
        self.x_g = nn.Parameter(state_dict[prefix + 'x_g'])

        self.receptance = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.receptance.weight = nn.Parameter(state_dict[prefix + 'receptance.weight'].view(hidden_size, hidden_size, 1, 1))
        self.key = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.key.weight = nn.Parameter(state_dict[prefix + 'key.weight'].view(hidden_size, hidden_size, 1, 1))
        self.value = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.value.weight = nn.Parameter(state_dict[prefix + 'value.weight'].view(hidden_size, hidden_size, 1, 1))

        self.matmul_time_decay_w1 = nn.Conv2d(hidden_size, self.D_DECAY_LORA, kernel_size=1, bias=False)
        self.matmul_time_decay_w1.weight = nn.Parameter(state_dict[prefix + 'w1'].t().view(self.D_DECAY_LORA, hidden_size, 1, 1))
        self.matmul_time_decay_w2 = nn.Conv2d(self.D_DECAY_LORA, hidden_size, kernel_size=1)
        self.matmul_time_decay_w2.weight = nn.Parameter(state_dict[prefix + 'w2'].t().view(hidden_size, self.D_DECAY_LORA, 1, 1))
        self.matmul_time_decay_w2.bias = nn.Parameter(state_dict[prefix + 'w0'].view(-1))

        self.matmul_a1 = nn.Conv2d(hidden_size, self.D_AAA_LORA, kernel_size=1, bias=False)
        self.matmul_a1.weight = nn.Parameter(state_dict[prefix + 'a1'].t().view(self.D_AAA_LORA, hidden_size, 1, 1))
        self.matmul_a2 = nn.Conv2d(self.D_AAA_LORA, hidden_size, kernel_size=1)
        self.matmul_a2.weight = nn.Parameter(state_dict[prefix + 'a2'].t().view(hidden_size, self.D_AAA_LORA, 1, 1))
        self.matmul_a2.bias = nn.Parameter(state_dict[prefix + 'a0'].view(-1))

        if layer_id != 0:
            self.D_MV_LORA = state_dict[prefix + 'v1'].shape[-1]
            self.matmul_v1 = nn.Conv2d(hidden_size, self.D_MV_LORA, kernel_size=1, bias=False)
            self.matmul_v1.weight = nn.Parameter(state_dict[prefix + 'v1'].t().view(self.D_MV_LORA, hidden_size, 1, 1))
            self.matmul_v2 = nn.Conv2d(self.D_MV_LORA, hidden_size, kernel_size=1)
            self.matmul_v2.weight = nn.Parameter(state_dict[prefix + 'v2'].t().view(hidden_size, self.D_MV_LORA, 1, 1))
            self.matmul_v2.bias = nn.Parameter(state_dict[prefix + 'v0'].view(-1))

        self.matmul_g1 = nn.Conv2d(hidden_size, self.D_GATE_LORA, kernel_size=1, bias=False)
        self.matmul_g1.weight = nn.Parameter(state_dict[prefix + 'g1'].t().view(self.D_GATE_LORA, hidden_size, 1, 1))
        self.matmul_g2 = nn.Conv2d(self.D_GATE_LORA, hidden_size, kernel_size=1, bias=False)
        self.matmul_g2.weight = nn.Parameter(state_dict[prefix + 'g2'].t().view(hidden_size, self.D_GATE_LORA, 1, 1))

        self.k_k = nn.Parameter(state_dict[prefix + 'k_k'].view(self.num_heads, 1, self.head_size))
        self.k_a = nn.Parameter(state_dict[prefix + 'k_a'].view(self.num_heads, 1, self.head_size))
        self.r_k = nn.Parameter(state_dict[prefix + 'r_k'].view(self.num_heads, 1, self.head_size))

        self.output = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.output.weight = nn.Parameter(state_dict[prefix + 'output.weight'].view(hidden_size, hidden_size, 1, 1))
        self.ln_x = nn.LayerNorm(self.head_size, eps=64e-5)
        self.ln_x_w = nn.Parameter(state_dict[prefix + 'ln_x.weight'])
        self.ln_x_b = nn.Parameter(state_dict[prefix + 'ln_x.bias'])
        self.mul_ln_x = Multiply()
        self.add_ln_x = Add()

        self.ln_1                   = nn.LayerNorm(hidden_size, eps=1e-5)
        self.ln_1.weight            = nn.Parameter(state_dict[f'blocks.{layer_id}.ln1.weight'])
        self.ln_1.bias              = nn.Parameter(state_dict[f'blocks.{layer_id}.ln1.bias'])
        self.add_attention          = Add()
        self.mul_gate               = Multiply()
        self.add_x_residual         = Add()
        self.sub_shifted            = Subtract()

        self.tanh_w                 = nn.Tanh()
        self.exp_w                  = Exponential()
        self.scale_w                = Multiply()
        self.scale_w_param          = nn.Parameter(torch.as_tensor([-0.606531] * self.head_size, dtype=state_dict[f'blocks.{layer_id}.ln1.bias'].dtype, device=state_dict[f'blocks.{layer_id}.ln1.bias'].device))
        self.ones_0                 = nn.Parameter(torch.ones(self.head_size, dtype=state_dict[f'blocks.{layer_id}.ln1.bias'].dtype, device=state_dict[f'blocks.{layer_id}.ln1.bias'].device))
        self.ones_1                 = nn.Parameter(torch.ones(self.head_size, dtype=state_dict[f'blocks.{layer_id}.ln1.bias'].dtype, device=state_dict[f'blocks.{layer_id}.ln1.bias'].device))
        self.sigmoid_a              = nn.Sigmoid()
        self.sigmoid_g              = nn.Sigmoid()
        self.sigmoid_v              = nn.Sigmoid()
        self.sigmoid_w              = nn.Sigmoid()

        self.lerp_mul_r             = Multiply()
        self.lerp_add_r             = Add()
        self.lerp_mul_w             = Multiply()
        self.lerp_add_w             = Add()
        self.lerp_mul_k             = Multiply()
        self.lerp_add_k             = Add()
        self.lerp_mul_v             = Multiply()
        self.lerp_add_v             = Add()
        self.lerp_mul_a             = Multiply()
        self.lerp_add_a             = Add()
        self.lerp_mul_g             = Multiply()
        self.lerp_add_g             = Add()

        self.sub_value              = Subtract()
        self.mul_value              = Multiply()
        self.add_value_residual     = Add()

        self.mix_kk                 = Multiply()
        self.mix_ka_mul_key         = Multiply()
        self.mix_ka_add             = Add()
        self.mix_ka_sub             = Subtract()
        self.mix_ka_mul_a           = Multiply()

        self.mul_r_k                = Multiply()
        self.mix_rk                 = Multiply()
        self.mix_rkv                = Multiply()
        self.reduce_sum             = Sum()
        
        self.l2norm                 = L2Norm()

        self.get_a                  = Neg()
        self.get_b                  = Multiply()

        self.concat_shift = Concat(1)
        self.shift_gather1 = CustomGather()
        self.shift_gather2 = CustomGather()
        self.state_reshape = Reshape()
        self.wkv7 = Wkv7(self.num_heads, self.head_size, custom_wkv=self.custom_wkv)

        self.pre_reshape_r = Reshape()
        self.pre_reshape_w = Reshape()
        self.pre_reshape_k = Reshape()
        self.pre_reshape_v = Reshape()
        self.pre_reshape_a = Reshape()
        self.pre_reshape_g = Reshape()
        self.pre_permute_r = Permute()
        self.pre_permute_w = Permute()
        self.pre_permute_k = Permute()
        self.pre_permute_v = Permute()
        self.pre_permute_a = Permute()
        self.pre_permute_g = Permute()

        self.post_reshape_r = Reshape()
        self.post_reshape_w = Reshape()
        self.post_reshape_k = Reshape()
        self.post_reshape_v = Reshape()
        self.post_reshape_a = Reshape()
        self.post_reshape_g = Reshape()
        self.post_reshape_v1 = Reshape()
        self.post_permute_r = Permute()
        self.post_permute_w = Permute()
        self.post_permute_k = Permute()
        self.post_permute_v = Permute()
        self.post_permute_a = Permute()
        self.post_permute_g = Permute()
        self.post_permute_v1 = Permute()

        self.pre_output_reshape = Reshape()
        self.pre_output_transpose = Permute()
        self.post_output_transpose = Permute()
        self.post_output_reshape = Reshape()

    def forward(self, x, state1, state2, v_first):
        last_x = x
        x = self.ln_1(x)
        batch_size, seq_length, _ = x.size()
        assert batch_size == 1
        if seq_length == 1:
            state1_out = x
            sx = self.sub_shifted(state1, x)
        else:
            state1_out = self.shift_gather1(x, torch.LongTensor([-1]).to(x.device), 1)
            past = self.shift_gather2(x, torch.LongTensor([i for i in range(seq_length-1)]).to(x.device), 1)
            past = self.concat_shift(self.state_reshape(state1, [1, 1, -1]), past)
            sx = self.sub_shifted(past, x)

        xr = self.lerp_add_r(x, self.lerp_mul_r(sx, self.x_r))
        xw = self.lerp_add_w(x, self.lerp_mul_w(sx, self.x_w))
        xk = self.lerp_add_k(x, self.lerp_mul_k(sx, self.x_k))
        xv = self.lerp_add_v(x, self.lerp_mul_v(sx, self.x_v))
        xa = self.lerp_add_a(x, self.lerp_mul_a(sx, self.x_a))
        xg = self.lerp_add_g(x, self.lerp_mul_g(sx, self.x_g))

        xr = self.pre_reshape_r(xr, [batch_size, seq_length, 1, self.hidden_size])
        xw = self.pre_reshape_w(xw, [batch_size, seq_length, 1, self.hidden_size])
        xk = self.pre_reshape_k(xk, [batch_size, seq_length, 1, self.hidden_size])
        xv = self.pre_reshape_v(xv, [batch_size, seq_length, 1, self.hidden_size])
        xa = self.pre_reshape_a(xa, [batch_size, seq_length, 1, self.hidden_size])
        xg = self.pre_reshape_g(xg, [batch_size, seq_length, 1, self.hidden_size])

        xr = self.pre_permute_r(xr, [0, 3, 2, 1])
        xw = self.pre_permute_w(xw, [0, 3, 2, 1])
        xk = self.pre_permute_k(xk, [0, 3, 2, 1])
        xv_premute = self.pre_permute_v(xv, [0, 3, 2, 1])
        xa = self.pre_permute_a(xa, [0, 3, 2, 1])
        xg = self.pre_permute_g(xg, [0, 3, 2, 1])

        receptance = self.receptance(xr)
        key = self.key(xk)
        value = self.value(xv_premute)
        gate = self.matmul_g2(self.sigmoid_g(self.matmul_g1(xg)))
        a = self.sigmoid_a(self.matmul_a2(self.matmul_a1(xa)))
        time_decay = self.matmul_time_decay_w2(self.tanh_w(self.matmul_time_decay_w1(xw)))

        receptance = self.post_permute_r(receptance, [0, 3, 2, 1])
        key = self.post_permute_k(key, [0, 3, 2, 1])
        value = self.post_permute_v(value, [0, 3, 2, 1])
        gate = self.post_permute_g(gate, [0, 3, 2, 1])
        a = self.post_permute_a(a, [0, 3, 2, 1])
        time_decay = self.post_permute_w(time_decay, [0, 3, 2, 1])

        receptance = self.post_reshape_r(receptance, [seq_length, self.num_heads, 1, self.head_size])
        key = self.post_reshape_k(key, [seq_length, self.num_heads, 1, self.head_size])
        value = self.post_reshape_v(value, [seq_length, self.num_heads, 1, self.head_size])
        gate = self.post_reshape_g(gate, [batch_size, seq_length, self.hidden_size])
        a = self.post_reshape_a(a, [seq_length, self.num_heads, 1, self.head_size])
        time_decay = self.post_reshape_w(time_decay, [seq_length, self.num_heads, 1, self.head_size])
        time_decay = self.exp_w(self.scale_w(self.sigmoid_w(time_decay), self.scale_w_param))

        kk = self.mix_kk(key, self.k_k)
        kk = self.l2norm(kk)
        key = self.mix_ka_mul_key(key, self.mix_ka_add(self.ones_0, self.mix_ka_mul_a(self.mix_ka_sub(a, self.ones_1), self.k_a)))

        if self.layer_id == 0:
            v_first = value
        else:
            tmp = self.sigmoid_v(self.matmul_v2(self.matmul_v1(xv_premute)))
            tmp = self.post_permute_v1(tmp, [0, 3, 2, 1])
            tmp = self.post_reshape_v1(tmp, [seq_length, self.num_heads, 1, self.head_size])
            value = self.add_value_residual(value, self.mul_value(self.sub_value(v_first, value), tmp))

        b = self.get_b(kk, a)
        a = self.get_a(kk)
        x, state2_out = self.wkv7(seq_length, receptance, time_decay, key, value, a, b, state2)

        # group_norm
        x = self.ln_x(x).view(batch_size, seq_length, self.hidden_size)
        x = self.mul_ln_x(x, self.ln_x_w)
        x = self.add_ln_x(x, self.ln_x_b)

        rkv = self.mix_rkv(self.reduce_sum(self.mul_r_k(self.mix_rk(receptance, key), self.r_k), dim=-1, keepdim=True), value).view(seq_length, self.hidden_size)
        x = self.add_x_residual(x , rkv)
        x = self.mul_gate(x, gate)

        x = self.pre_output_reshape(x, [batch_size, seq_length, 1, self.hidden_size])
        x = self.pre_output_transpose(x, [0, 3, 2, 1])
        x = self.output(x)
        x = self.post_output_transpose(x, [0, 3, 2, 1])
        x = self.post_output_reshape(x, [batch_size, seq_length, self.hidden_size])

        # if self.layer_id == 0:
        return self.add_attention(last_x, x), state1_out, state2_out, v_first
        # else:
        #     return self.add_attention(last_x, x), state1_out, state2_out

class Rwkv7FeedForward(nn.Module):
    def __init__(self, state_dict, hidden_size, intermediate_size, layer_id=0, layer_total=0, output_last=False):
        super().__init__()
        prefix = f'blocks.{layer_id}.ffn.'
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.layer_total = layer_total
        self.output_last = output_last
        self.x_k = nn.Parameter(state_dict[prefix + 'x_k'])

        # self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        # self.key.weight = nn.Parameter(state_dict[prefix + 'key.weight'])
        # self.value = nn.Linear(intermediate_size, hidden_size, bias=False)
        # self.value.weight = nn.Parameter(state_dict[prefix + 'value.weight'])

        self.ln_2                   = nn.LayerNorm(hidden_size, eps=1e-5)
        self.ln_2.weight            = nn.Parameter(state_dict[f'blocks.{layer_id}.ln2.weight'])
        self.ln_2.bias              = nn.Parameter(state_dict[f'blocks.{layer_id}.ln2.bias'])

        self.sub_shifted            = Subtract()
        self.mul_x_k                = Multiply()
        self.add_x_k                = Add()
        self.relu                   = nn.ReLU()
        self.pow                    = Pow()
        self.add_feed_forward       = Add()

        self.concat_shift = Concat(1)
        self.shift_gather1 = CustomGather()
        self.shift_gather2 = CustomGather()
        if self.layer_id == self.layer_total - 1:
            self.shift_gather3 = CustomGather()
            self.shift_gather4 = CustomGather()
            self.shift_split = Split()
        self.state_reshape = Reshape()
        self.layer_total = layer_total

        self.key = nn.Conv2d(hidden_size, intermediate_size, 1, bias=False)
        self.key.weight = nn.Parameter(state_dict[prefix + 'key.weight'].view(intermediate_size, hidden_size, 1, 1))
        self.value = nn.Conv2d(intermediate_size, hidden_size, 1, bias=False)
        self.value.weight = nn.Parameter(state_dict[prefix + 'value.weight'].view(hidden_size, intermediate_size, 1, 1))
        self.pre_conv_reshape = Reshape()
        self.pre_conv_transpose = Permute()
        self.post_conv_transpose = Permute()
        self.post_conv_reshape = Reshape()
        self.pre_conv_reshape2 = Reshape()
        self.pre_conv_transpose2 = Permute()
        self.post_conv_transpose2 = Permute()
        self.post_conv_reshape2 = Reshape()

    def forward(self, x, state):
        batch_size, seq_length, _ = x.size()
        if self.output_last and self.layer_id == self.layer_total - 1:
            if seq_length == 1:
                last_x = x
                x = self.ln_2(x)
                state_out = x
                sx = self.sub_shifted(state, x)
            else:
                last_x = self.shift_gather4(x, torch.LongTensor([-1]).to(x.device), 1)
                x = self.shift_gather3(x, torch.LongTensor([-2, -1]).to(x.device), 1)
                x = self.ln_2(x)
                past, state_out = self.shift_split(x, 1, dim=1)
                sx = self.sub_shifted(past, state_out)
                x = state_out
        else:
            last_x = x
            x = self.ln_2(x)
            assert batch_size == 1
            if seq_length == 1:
                state_out = x
                sx = self.sub_shifted(state, x)
            else:
                state_out = self.shift_gather1(x, torch.LongTensor([-1]).to(x.device), 1)
                past = self.shift_gather2(x, torch.LongTensor([i for i in range(seq_length-1)]).to(x.device), 1)
                past = self.concat_shift(self.state_reshape(state, [1, 1, -1]), past)
                sx = self.sub_shifted(past, x)

        xk = self.add_x_k(x, self.mul_x_k(sx, self.x_k))

        xk = self.pre_conv_reshape(xk, [batch_size, -1, 1, self.hidden_size])
        xk = self.pre_conv_transpose(xk, [0, 3, 2, 1])
        key = self.key(xk)
        key = self.post_conv_transpose2(key, [0, 3, 2, 1])
        key = self.post_conv_reshape2(key, [batch_size, -1, self.intermediate_size])

        key = self.pow(self.relu(key), 2)

        key = self.pre_conv_reshape2(key, [batch_size, -1, 1, self.intermediate_size])
        key = self.pre_conv_transpose2(key, [0, 3, 2, 1])
        value = self.value(key)
        value = self.post_conv_transpose(value, [0, 3, 2, 1])
        value = self.post_conv_reshape(value, [batch_size, -1, self.hidden_size])

        return self.add_feed_forward(value, last_x), state_out