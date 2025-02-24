import torch
import torch.nn as nn
import torch.nn.functional as F
# try:
from aimet_torch.v2.nn.modules.custom import *
from aimet_torch.v2.quantization.tensor import DequantizedTensor

# except:
#     from rwkv_src.elemwise_ops import *

# try:
# from fla.ops.rwkv7 import fused_recurrent_rwkv7, chunk_rwkv7
# except:
    # pass

custom_norm_wrapper_src = """
#include <torch/extension.h>
#include <torch/script.h>

torch::Tensor l2norm(torch::Tensor x) {
    return x / (x.norm(2, -1, true) + 1e-12);
}

TORCH_LIBRARY(customop, m) {
    m.def("l2norm", &l2norm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
"""

_ = torch.utils.cpp_extension.load_inline(
    name='extension', cpp_sources=[custom_norm_wrapper_src])

class Wkv7(nn.Module):
    def __init__(self, num_heads, head_size, custom_wkv=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.custom_wkv = custom_wkv
        self.apply_time_decay = Multiply()
        self.matmul_sa = MatMul()
        self.matmul_sab = MatMul()
        self.matmul_kv = MatMul()
        self.matmul_r = MatMul()
        self.add_kv = Add()
        self.add_sab = Add()
        # self.split_r = Split()
        # self.split_w = Split()
        # self.split_k = Split()
        # self.split_v = Split()
        # self.split_a = Split()
        # self.split_b = Split()
        # self.split_state = Split()
        self.reshape_r = Reshape()
        self.reshape_w = Reshape()
        self.reshape_k = Reshape()
        self.reshape_v = Reshape()
        self.reshape_a = Reshape()
        self.reshape_b = Reshape()

        # self.concat_x = Concat(0)
        # self.concat_state = Concat(0)

        if custom_wkv:
            # for adding the onnx nodes
            from rwkv_src.wkv_custom import wkv_c_impl_src
            module = torch.utils.cpp_extension.load_inline(
                    name='extension', cpp_sources=[wkv_c_impl_src])
            self.wkv_func = torch.ops.rwkv.wkv7

    def forward(self, seq_length, r, w, k, v, a, b, state2):
        if self.custom_wkv:
            # b = b.reshape(seq_length*self.num_heads, self.head_size)
            # a = a.reshape(seq_length*self.num_heads, self.head_size)
            # w = w.reshape(seq_length*self.num_heads, self.head_size)
            # r = r.reshape(seq_length*self.num_heads, self.head_size)
            # k = k.reshape(seq_length*self.num_heads, self.head_size)
            # v = v.reshape(seq_length*self.num_heads, self.head_size)
            r = self.reshape_r(r, [seq_length*self.num_heads, self.head_size])
            w = self.reshape_w(w, [seq_length*self.num_heads, self.head_size])
            k = self.reshape_k(k, [seq_length*self.num_heads, self.head_size])
            v = self.reshape_v(v, [seq_length*self.num_heads, self.head_size])
            a = self.reshape_a(a, [seq_length*self.num_heads, self.head_size])
            b = self.reshape_b(b, [seq_length*self.num_heads, self.head_size])
            # if seq_length == 1:
            if False:
                # k_split = torch.split(k, self.num_heads//4, dim=0)
                # v_split = torch.split(v, self.num_heads//4, dim=0)
                # r_split = torch.split(r, self.num_heads//4, dim=0)
                # w_split = torch.split(w, self.num_heads//4, dim=0)
                # a_split = torch.split(a, self.num_heads//4, dim=0)
                # b_split = torch.split(b, self.num_heads//4, dim=0)
                k_split = self.split_k(k, self.num_heads//4, dim=0)
                v_split = self.split_v(v, self.num_heads//4, dim=0)
                r_split = self.split_r(r, self.num_heads//4, dim=0)
                w_split = self.split_w(w, self.num_heads//4, dim=0)
                a_split = self.split_a(a, self.num_heads//4, dim=0)
                b_split = self.split_b(b, self.num_heads//4, dim=0)
                if (len(state2.shape) == 3):
                    # state2_split = torch.split(state2, self.num_heads//4, dim=0)
                    state2_split = self.split_state(state2, self.num_heads//4, dim=0)
                else:
                    # state2_split = torch.split(state2, self.num_heads//4, dim=1)
                    state2_split = self.split_state(state2, self.num_heads//4, dim=1)
                x0, state2_out0 = self.wkv_func(r_split[0], w_split[0], k_split[0], v_split[0], a_split[0], b_split[0], state2_split[0])
                x1, state2_out1 = self.wkv_func(r_split[1], w_split[1], k_split[1], v_split[1], a_split[1], b_split[1], state2_split[1])
                x2, state2_out2 = self.wkv_func(r_split[2], w_split[2], k_split[2], v_split[2], a_split[2], b_split[2], state2_split[2])
                x3, state2_out3 = self.wkv_func(r_split[3], w_split[3], k_split[3], v_split[3], a_split[3], b_split[3], state2_split[3])

                # x = torch.cat([x0, x1, x2, x3], dim=0)
                # state2_out = torch.cat([state2_out0, state2_out1, state2_out2, state2_out3], dim=0)
                x = self.concat_x(x0, x1, x2, x3)
                state2_out = self.concat_state(state2_out0, state2_out1, state2_out2, state2_out3)
            else:
                x, state2_out = self.wkv_func(r, w, k, v, a, b, state2)
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
                # r = r.reshape(1, seq_length, self.num_heads, self.head_size)
                # v = v.reshape(1, seq_length, self.num_heads, self.head_size)
                # k = k.reshape(1, seq_length, self.num_heads, self.head_size)
                # w = w.reshape(1, seq_length, self.num_heads, self.head_size)
                # b = b.reshape(1, seq_length, self.num_heads, self.head_size)
                # a = a.reshape(1, seq_length, self.num_heads, self.head_size)
                # state2 = state2.reshape(1, self.num_heads, self.head_size, self.head_size).permute(0, 1, 3, 2).contiguous()

                # x, state2_out = chunk_rwkv7(
                #     r=r, w=w, k=k, v=v, a=a, b=b, scale=1.,
                #     initial_state=state2, output_final_state=True,
                #     cu_seqlens=None, head_first=False
                # )

                # x = x.view(seq_length, self.num_heads, 1, self.head_size)
                # state2_out = state2_out.permute(0, 1, 3, 2).contiguous().view(self.num_heads, self.head_size, self.head_size)

                r = r.view(seq_length, self.num_heads, self.head_size, 1)
                v = v.view(seq_length, self.num_heads, self.head_size, 1)
                k = k.view(seq_length, self.num_heads, 1, self.head_size)
                w = w.view(seq_length, self.num_heads, 1, self.head_size)
                b = b.view(seq_length, self.num_heads, 1, self.head_size)
                a = a.view(seq_length, self.num_heads, self.head_size, 1)
                kv = self.matmul_kv(v, k)
                x = torch.zeros(seq_length, self.num_heads, self.head_size, 1, device=k.device, dtype=kv.dtype)
                for i in range(seq_length):
                    state2 = self.apply_time_decay(state2, w[i, :, :, :].exp()) + (state2 @ a[i, :, :, :] @ b[i, :, :, :]) + kv[i, :, :, :]
                    x[i, :, :, :] = state2 @ r[i, :, :, :]
                state2_out = state2
                x = x.view(seq_length, self.num_heads, 1, self.head_size)

        return x, state2_out


class Rwkv7SelfAttention(nn.Module):
    def __init__(self, state_dict, hidden_size, head_size, layer_id=0, rescale_layer=0, custom_wkv=False, online_preparing=False):
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

        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance.weight = nn.Parameter(state_dict[prefix + 'receptance.weight'])
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key.weight = nn.Parameter(state_dict[prefix + 'key.weight'])
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value.weight = nn.Parameter(state_dict[prefix + 'value.weight'])

        self.matmul_time_decay_w1   = nn.Linear(hidden_size, self.D_DECAY_LORA, bias=False)
        self.matmul_time_decay_w1.weight = nn.Parameter(state_dict[prefix + 'w1'].t())
        self.matmul_time_decay_w2   = nn.Linear(self.D_DECAY_LORA, hidden_size)
        self.matmul_time_decay_w2.weight = nn.Parameter(state_dict[prefix + 'w2'].t())
        self.matmul_time_decay_w2.bias = nn.Parameter(state_dict[prefix + 'w0'].view(-1))

        self.matmul_a1 = nn.Linear(hidden_size, self.D_AAA_LORA, bias=False)
        self.matmul_a1.weight = nn.Parameter(state_dict[prefix + 'a1'].t())
        self.matmul_a2 = nn.Linear(self.D_AAA_LORA, hidden_size)
        self.matmul_a2.weight = nn.Parameter(state_dict[prefix + 'a2'].t())
        self.matmul_a2.bias = nn.Parameter(state_dict[prefix + 'a0'].view(-1))

        if layer_id != 0:
            self.D_MV_LORA = state_dict[prefix + 'v1'].shape[-1]
            self.matmul_v1 = nn.Linear(hidden_size, self.D_MV_LORA, bias=False)
            self.matmul_v1.weight = nn.Parameter(state_dict[prefix + 'v1'].t())
            self.matmul_v2 = nn.Linear(self.D_MV_LORA, hidden_size)
            self.matmul_v2.weight = nn.Parameter(state_dict[prefix + 'v2'].t())
            self.matmul_v2.bias = nn.Parameter(state_dict[prefix + 'v0'].view(-1))

        self.matmul_g1 = nn.Linear(hidden_size, self.D_GATE_LORA, bias=False)
        self.matmul_g1.weight = nn.Parameter(state_dict[prefix + 'g1'].t())
        self.matmul_g2 = nn.Linear(self.D_GATE_LORA, hidden_size, bias=False)
        self.matmul_g2.weight = nn.Parameter(state_dict[prefix + 'g2'].t())

        self.k_k = nn.Parameter(state_dict[prefix + 'k_k'].view(self.num_heads, self.head_size))
        self.k_a = nn.Parameter(state_dict[prefix + 'k_a'].view(self.num_heads, self.head_size))
        self.r_k = nn.Parameter(state_dict[prefix + 'r_k'].view(self.num_heads, self.head_size))

        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output.weight = nn.Parameter(state_dict[prefix + 'output.weight'])
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
        
        self.l2norm                 = Normalize()

        self.get_a                  = Neg()
        self.get_b                  = Multiply()

        self.wkv7 = Wkv7(self.num_heads, self.head_size, custom_wkv=self.custom_wkv)
    
    def forward(self, x, state1, state2, v_first):
        last_x = x
        x = self.ln_1(x)
        batch_size, seq_length, _ = x.size()
        assert batch_size == 1
        if seq_length == 1:
            state1_out = x
            sx = self.sub_shifted(state1, x)
        else:
            past = torch.cat([state1.unsqueeze(1), x[:, :-1, :]], dim=1)
            sx = self.sub_shifted(past, x)
            state1_out = x[:, -1, :]

        xr = self.lerp_add_r(x, self.lerp_mul_r(sx, self.x_r))
        xw = self.lerp_add_w(x, self.lerp_mul_w(sx, self.x_w))
        xk = self.lerp_add_k(x, self.lerp_mul_k(sx, self.x_k))
        xv = self.lerp_add_v(x, self.lerp_mul_v(sx, self.x_v))
        xa = self.lerp_add_a(x, self.lerp_mul_a(sx, self.x_a))
        xg = self.lerp_add_g(x, self.lerp_mul_g(sx, self.x_g))

        receptance = self.receptance(xr).view(seq_length, self.num_heads, self.head_size)
        key = self.key(xk).view(seq_length, self.num_heads, self.head_size)
        value = self.value(xv).view(seq_length, self.num_heads, self.head_size)
        gate = self.matmul_g2(self.sigmoid_g(self.matmul_g1(xg)))
        a = self.sigmoid_a(self.matmul_a2(self.matmul_a1(xa))).view(seq_length, self.num_heads, self.head_size)
        time_decay = self.matmul_time_decay_w2(self.tanh_w(self.matmul_time_decay_w1(xw)))
        if seq_length == 1 or self.custom_wkv:
            time_decay = self.exp_w(self.scale_w(-0.606531, self.sigmoid_w(time_decay)))
        else:
            time_decay = self.scale_w(-0.606531, self.sigmoid_w(time_decay))

        kk = self.mix_kk(key, self.k_k)
        # kk = torch.ops.customop.l2norm(kk)
        kk = self.l2norm(kk, p=2, dim=-1, eps=1e-12)
        key = self.mix_ka_mul_key(key, self.mix_ka_add(1, self.mix_ka_mul_a(self.mix_ka_sub(a, 1), self.k_a)))

        if self.layer_id == 0:
            v_first = value
        else:
            value = self.add_value_residual(value, self.mul_value(self.sub_value(v_first, value), self.sigmoid_v(self.matmul_v2(self.matmul_v1(xv)).view(seq_length, self.num_heads, self.head_size))))

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
        x = self.output(x)

        return self.add_attention(last_x, x), state1_out, state2_out, v_first

class Rwkv7FeedForward(nn.Module):
    def __init__(self, state_dict, hidden_size, intermediate_size, layer_id=0):
        super().__init__()
        prefix = f'blocks.{layer_id}.ffn.'
        self.layer_id = layer_id
        self.hidden_size = hidden_size

        self.x_k = nn.Parameter(state_dict[prefix + 'x_k'])

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.key.weight = nn.Parameter(state_dict[prefix + 'key.weight'])
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.value.weight = nn.Parameter(state_dict[prefix + 'value.weight'])

        self.ln_2                   = nn.LayerNorm(hidden_size, eps=1e-5)
        self.ln_2.weight            = nn.Parameter(state_dict[f'blocks.{layer_id}.ln2.weight'])
        self.ln_2.bias              = nn.Parameter(state_dict[f'blocks.{layer_id}.ln2.bias'])

        self.sub_shifted            = Subtract()
        self.mul_x_k                = Multiply()
        self.add_x_k                = Add()
        self.relu                   = nn.ReLU()
        self.pow                    = Pow()
        self.add_feed_forward       = Add()

    def forward(self, x, state):
        last_x = x
        x = self.ln_2(x)
        batch_size, seq_length, _ = x.size()
        assert batch_size == 1
        if seq_length == 1:
            state_out = x
            sx = self.sub_shifted(state, x)
        else:
            past = torch.cat([state.unsqueeze(1), x[:, :-1, :]], dim=1)
            sx = self.sub_shifted(past, x)
            state_out = x[:, -1, :]

        xk = self.add_x_k(x, self.mul_x_k(sx, self.x_k))

        key = self.pow(self.relu(self.key(xk)), 2)
        value = self.value(key)

        return self.add_feed_forward(value, last_x), state_out