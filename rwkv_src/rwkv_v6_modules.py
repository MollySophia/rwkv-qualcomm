import torch
import torch.nn as nn
import torch.nn.functional as F
import rwkv_src.elemwise_ops as op

class Rwkv6SelfAttention(nn.Module):
    def __init__(self, state_dict, hidden_size, head_size, layer_id=0, rescale_layer=0, custom_wkv=False):
        super().__init__()
        prefix = f'blocks.{layer_id}.att.'
        self.layer_id = layer_id
        self.num_heads = hidden_size // head_size
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.custom_wkv = custom_wkv

        self.TIME_MIX_EXTRA_DIM = 64 if hidden_size == 4096 else 32
        self.TIME_DECAY_EXTRA_DIM = 128 if hidden_size == 4096 else 64

        self.time_maa_x = nn.Parameter(state_dict[prefix + 'time_maa_x'])
        maa = torch.cat([state_dict[prefix + f'time_maa_{i}'].view(1, 1, -1) for i in ['w', 'k', 'v', 'r', 'g']], dim=0)
        self.time_maa = nn.Parameter(maa)

        self.time_decay = nn.Parameter(state_dict[prefix + 'time_decay'].view(self.num_heads, 1, self.head_size))
        self.time_first = nn.Parameter(state_dict[prefix + 'time_faaaa'].view(self.num_heads, 1, self.head_size))

        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance.weight = nn.Parameter(state_dict[prefix + 'receptance.weight'])
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key.weight = nn.Parameter(state_dict[prefix + 'key.weight'] / 2)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value.weight = nn.Parameter(state_dict[prefix + 'value.weight'] / 4)
        self.gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate.weight = nn.Parameter(state_dict[prefix + 'gate.weight'])

        self.matmul_time_maa_w1     = nn.Linear(hidden_size, self.TIME_MIX_EXTRA_DIM*5, bias=False)
        self.matmul_time_maa_w1.weight = nn.Parameter(state_dict[prefix + 'time_maa_w1'].t())
        self.matmul_time_decay_w1   = nn.Linear(hidden_size, self.TIME_DECAY_EXTRA_DIM, bias=False)
        self.matmul_time_decay_w1.weight = nn.Parameter(state_dict[prefix + 'time_decay_w1'].t())
        self.matmul_time_decay_w2   = nn.Linear(self.TIME_DECAY_EXTRA_DIM, hidden_size, bias=False)
        self.matmul_time_decay_w2.weight = nn.Parameter(state_dict[prefix + 'time_decay_w2'].t())

        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        if rescale_layer > 0:
            self.output.weight = nn.Parameter(state_dict[prefix + 'output.weight'] / (2 ** int(layer_id // rescale_layer)))
        else:
            self.output.weight = nn.Parameter(state_dict[prefix + 'output.weight'])
        # self.ln_x = nn.InstanceNorm2d(self.num_heads, eps=1e-5, affine=False)
        self.ln_x = nn.LayerNorm(self.head_size, eps=1e-5)
        self.ln_x.weight = nn.Parameter(torch.ones(self.head_size))
        self.ln_x.bias = nn.Parameter(torch.zeros(self.head_size))
        self.ln_x_w = nn.Parameter(state_dict[prefix + 'ln_x.weight'])
        self.ln_x_b = nn.Parameter(state_dict[prefix + 'ln_x.bias'])
        self.mul_ln_x = op.Multiply()
        self.add_ln_x = op.Add()

        self.sub_shift              = op.Subtract()
        self.mul_time_maa           = op.Multiply()
        self.add_time_maa           = op.Add()
        self.time_maa_w2            = nn.Parameter(state_dict[prefix + 'time_maa_w2'])
        self.matmul_time_maa_w2     = op.Bmm()
        self.add_time_maa           = op.Add()
        self.mul_time_maa           = op.Multiply()
        self.add_x                  = op.Add()

        self.ln_1                   = nn.LayerNorm(hidden_size, eps=1e-5)
        self.ln_1.weight            = nn.Parameter(state_dict[f'blocks.{layer_id}.ln1.weight'])
        self.ln_1.bias              = nn.Parameter(state_dict[f'blocks.{layer_id}.ln1.bias'])
        self.add_time_decay0        = op.Add()

        self.matmul_kv              = op.MatMul()
        self.mul_time_first         = op.Multiply()
        self.add_time_first         = op.Add()
        self.matmul_rkv             = op.MatMul()
        self.mul_time_decay         = op.Multiply()
        self.add_time_decay1        = op.Add()
        self.mul_attention          = op.Multiply()

        self.tanh0                  = op.Tanh()
        self.tanh1                  = op.Tanh()
        self.silu0                  = op.SiLU()
        self.split0                 = op.Split()
        self.exp0                   = op.Exponential()
        self.exp1                   = op.Exponential()
        self.neg                    = op.Neg()
        self.add_attention          = op.Add()

        if custom_wkv:
            # dummy customop for adding the onnx nodes
            from rwkv_src.wkv_custom import wkv_c_impl_src
            module = torch.utils.cpp_extension.load_inline(
                    name='extension', cpp_sources=[wkv_c_impl_src])
            self.wkv_func = torch.ops.rwkv.wkv
            self.wkv_chunk_func = torch.ops.rwkv.wkv_chunk
    
    def forward(self, x, state1, state2):
        last_x = x
        x = self.ln_1(x)
        batch_size, seq_length, _ = x.size()
        assert batch_size == 1
        if seq_length == 1:
            state1_out = x
            sx = self.sub_shift(state1, x)
        else:
            past = torch.cat([state1.unsqueeze(1), x[:, :-1, :]], dim=1)
            sx = self.sub_shift(past, x)
            state1_out = x[:, -1, :]

        xxx = self.add_time_maa(x, self.mul_time_maa(sx, self.time_maa_x))
        if seq_length == 1:
            xxx = self.tanh0(self.matmul_time_maa_w1(xxx)).view(5, 1, -1)
        else:
            xxx = self.tanh0(self.matmul_time_maa_w1(xxx)).view(seq_length, 5, -1).transpose(0, 1)
        xxx = self.matmul_time_maa_w2(xxx, self.time_maa_w2)
        xxx = self.mul_time_maa(sx, self.add_time_maa(xxx, self.time_maa))
        xxx = self.add_x(x, xxx)
        mw, mk, mv, mr, mg = self.split0(xxx, split_size_or_sections=1, dim=0)

        receptance = self.receptance(mr).view(self.num_heads * seq_length, self.head_size, 1)
        key = self.key(mk).view(self.num_heads * seq_length, 1, self.head_size)
        value = self.value(mv).view(self.num_heads * seq_length, self.head_size, 1)
        gate = self.silu0(self.gate(mg))

        mw = self.tanh1(self.matmul_time_decay_w1(mw))
        time_decay = self.matmul_time_decay_w2(mw)
        time_decay = self.add_time_decay0(self.time_decay, time_decay.view(seq_length, self.num_heads, 1, self.head_size))
        time_decay = self.exp1(self.neg(self.exp0(time_decay.clip(-9.72, 2.27))))

        # wkv
        if self.custom_wkv and self.wkv_func is not None and self.wkv_chunk_func is not None:
            if seq_length == 1:
                wkv, state2_out = self.wkv_func(key, value, receptance, state2, self.time_first, time_decay)
            else:
                wkv, state2_out = self.wkv_chunk_func(key, value, receptance, state2, self.time_first, time_decay)
        else:
            # kv = self.matmul_kv(key, value)
            kv = self.matmul_kv(value, key)
            if seq_length == 1:
                wkv = self.add_time_first(self.mul_time_first(kv, self.time_first), state2)
                wkv = self.matmul_rkv(wkv, receptance).view(self.num_heads, 1, self.head_size)
                state2_out = self.add_time_decay1(kv, self.mul_time_decay(state2, time_decay))
            else:
                kv = kv.view(seq_length, self.num_heads, self.head_size, self.head_size)
                receptance = receptance.view(seq_length, self.num_heads, self.head_size, 1)
                time_decay = time_decay.view(seq_length, self.num_heads, 1, self.head_size)
                wkv = torch.zeros(seq_length, self.num_heads, self.head_size, 1, device=x.device)
                for i in range(seq_length):
                    tmp = self.add_time_first(self.mul_time_first(kv[i, :, :, :], self.time_first), state2)
                    wkv[i, :, :, :] = self.matmul_rkv(tmp, receptance[i, :, :, :])
                    state2 = self.add_time_decay1(kv[i, :, :, :], self.mul_time_decay(state2, time_decay[i, :, :, :]))
                state2_out = state2
                wkv = wkv.view(seq_length, self.num_heads, 1, self.head_size)

        x = self.ln_x(wkv).view(batch_size, seq_length, self.hidden_size)

        x = self.mul_ln_x(x, self.ln_x_w)
        x = self.add_ln_x(x, self.ln_x_b)
        x = self.mul_attention(x, gate)
        x = self.output(x)

        return self.add_attention(last_x, x), state1_out, state2_out

class Rwkv6FeedForward(nn.Module):
    def __init__(self, state_dict, hidden_size, intermediate_size, layer_id=0, rescale_layer=0):
        super().__init__()
        prefix = f'blocks.{layer_id}.ffn.'
        self.layer_id = layer_id
        self.hidden_size = hidden_size

        self.time_maa_k = nn.Parameter(state_dict[prefix + 'time_maa_k'])
        self.time_maa_r = nn.Parameter(state_dict[prefix + 'time_maa_r'])

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.key.weight = nn.Parameter(state_dict[prefix + 'key.weight'])
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance.weight = nn.Parameter(state_dict[prefix + 'receptance.weight'])
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)
        if rescale_layer > 0:
            self.value.weight = nn.Parameter(state_dict[prefix + 'value.weight'] / (2 ** int(layer_id // rescale_layer)))
        else:
            self.value.weight = nn.Parameter(state_dict[prefix + 'value.weight'])

        self.ln_2                   = nn.LayerNorm(hidden_size, eps=1e-5)
        self.ln_2.weight            = nn.Parameter(state_dict[f'blocks.{layer_id}.ln2.weight'])
        self.ln_2.bias              = nn.Parameter(state_dict[f'blocks.{layer_id}.ln2.bias'])

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
        self.add_feed_forward       = op.Add()

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

        xk = self.add_time_maa_k(x, self.mul_time_maa_k(sx, self.time_maa_k))
        xr = self.add_time_maa_r(x, self.mul_time_maa_r(sx, self.time_maa_r))

        key = self.pow(self.relu(self.key(xk)), 2)
        value = self.value(key)
        receptance = self.sigmoid(self.receptance(xr))

        return self.add_feed_forward(self.mul_rv(receptance, value), last_x), state_out