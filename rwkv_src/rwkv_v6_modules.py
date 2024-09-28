import torch
import torch.nn as nn
import rwkv_src.elemwise_ops as op

class Rwkv6SelfAttention(nn.Module):
    def __init__(self, state_dict, hidden_size, head_size, layer_id=0, rescale_layer=0):
        super().__init__()
        prefix = f'blocks.{layer_id}.att.'
        self.layer_id = layer_id
        self.num_heads = hidden_size // head_size
        self.head_size = head_size
        self.hidden_size = hidden_size

        self.TIME_MIX_EXTRA_DIM = 64 if hidden_size == 4096 else 32
        self.TIME_DECAY_EXTRA_DIM = 128 if hidden_size == 4096 else 64

        self.time_maa_x = nn.Parameter(state_dict[prefix + 'time_maa_x'])
        self.time_maa_w = nn.Parameter(state_dict[prefix + 'time_maa_w'].flatten())
        self.time_maa_k = nn.Parameter(state_dict[prefix + 'time_maa_k'].flatten())
        self.time_maa_v = nn.Parameter(state_dict[prefix + 'time_maa_v'].flatten())
        self.time_maa_r = nn.Parameter(state_dict[prefix + 'time_maa_r'].flatten())
        self.time_maa_g = nn.Parameter(state_dict[prefix + 'time_maa_g'].flatten())

        self.time_maa_w1 = nn.Parameter(state_dict[prefix + 'time_maa_w1'])
        self.time_maa_w2 = nn.Parameter(state_dict[prefix + 'time_maa_w2'])

        self.time_decay = nn.Parameter(state_dict[prefix + 'time_decay'].reshape(self.num_heads, -1, 1))
        self.time_decay_w1 = nn.Parameter(state_dict[prefix + 'time_decay_w1'])
        self.time_decay_w2 = nn.Parameter(state_dict[prefix + 'time_decay_w2'])

        self.time_faaaa = nn.Parameter(state_dict[prefix + 'time_faaaa'].reshape(self.num_heads, -1, 1))

        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance.weight = nn.Parameter(state_dict[prefix + 'receptance.weight'])
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key.weight = nn.Parameter(state_dict[prefix + 'key.weight'] / 2)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value.weight = nn.Parameter(state_dict[prefix + 'value.weight'] / 4)
        self.gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate.weight = nn.Parameter(state_dict[prefix + 'gate.weight'])
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        if rescale_layer > 0:
            self.output.weight = nn.Parameter(state_dict[prefix + 'output.weight'] / (2 ** int(layer_id // rescale_layer)))
        else:
            self.output.weight = nn.Parameter(state_dict[prefix + 'output.weight'])
        self.ln_x = nn.InstanceNorm2d(self.num_heads)
        self.ln_x_weight = nn.Parameter(state_dict[prefix + 'ln_x.weight'])
        self.ln_x_bias = nn.Parameter(state_dict[prefix + 'ln_x.bias'])
        self.mul_ln_x = op.Multiply()
        self.add_ln_x = op.Add()

        self.sub_shift              = op.Subtract()
        self.mul_time_maa           = op.Multiply()
        self.add_time_maa           = op.Add()
        self.matmul_time_maa_w1     = nn.Linear(hidden_size, self.TIME_MIX_EXTRA_DIM*5, bias=False)
        self.matmul_time_maa_w1.weight = nn.Parameter(state_dict[prefix + 'time_maa_w1'].t())
        self.matmul_time_maa_w2_0   = nn.Linear(self.TIME_MIX_EXTRA_DIM, hidden_size, bias=False)
        self.matmul_time_maa_w2_0.weight = nn.Parameter(state_dict[prefix + 'time_maa_w2'][0,:,:].t())
        self.matmul_time_maa_w2_1   = nn.Linear(self.TIME_MIX_EXTRA_DIM, hidden_size, bias=False)
        self.matmul_time_maa_w2_1.weight = nn.Parameter(state_dict[prefix + 'time_maa_w2'][1,:,:].t())
        self.matmul_time_maa_w2_2   = nn.Linear(self.TIME_MIX_EXTRA_DIM, hidden_size, bias=False)
        self.matmul_time_maa_w2_2.weight = nn.Parameter(state_dict[prefix + 'time_maa_w2'][2,:,:].t())
        self.matmul_time_maa_w2_3   = nn.Linear(self.TIME_MIX_EXTRA_DIM, hidden_size, bias=False)
        self.matmul_time_maa_w2_3.weight = nn.Parameter(state_dict[prefix + 'time_maa_w2'][3,:,:].t())
        self.matmul_time_maa_w2_4   = nn.Linear(self.TIME_MIX_EXTRA_DIM, hidden_size, bias=False)
        self.matmul_time_maa_w2_4.weight = nn.Parameter(state_dict[prefix + 'time_maa_w2'][4,:,:].t())
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

        self.ln_1                   = nn.LayerNorm(hidden_size, eps=1e-5)
        self.ln_1_w                 = nn.Parameter(state_dict[f'blocks.{layer_id}.ln1.weight'])
        self.ln_1_b                 = nn.Parameter(state_dict[f'blocks.{layer_id}.ln1.bias'])
        self.mul_ln_1               = op.Multiply()
        self.add_ln_1               = op.Add()
        self.matmul_time_decay_w1   = nn.Linear(hidden_size, self.TIME_DECAY_EXTRA_DIM, bias=False)
        self.matmul_time_decay_w1.weight = nn.Parameter(state_dict[prefix + 'time_decay_w1'].t())
        self.matmul_time_decay_w2   = nn.Linear(self.TIME_DECAY_EXTRA_DIM, hidden_size, bias=False)
        self.matmul_time_decay_w2.weight = nn.Parameter(state_dict[prefix + 'time_decay_w2'].t())
        self.add_time_decay0        = op.Add()

        self.matmul_kv              = op.MatMul()
        self.mul_time_first         = op.Multiply()
        self.add_time_first         = op.Add()
        self.mul_scale_kv           = op.Multiply()
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
        self.time_first             = nn.Parameter(state_dict[prefix + 'time_faaaa'].view(self.num_heads, self.head_size, 1))
    
    def forward(self, x, state1, state2):
        last_x = x
        x = self.ln_1(x)
        x = self.mul_ln_1(x, self.ln_1_w)
        x = self.add_ln_1(x, self.ln_1_b)
        state1_out = x
        sx = self.sub_shift(state1, x)
        xxx = self.add_time_maa(x, self.mul_time_maa(sx, self.time_maa_x))
        xxx = self.tanh0(self.matmul_time_maa_w1(xxx))
        mw, mk, mv, mr, mg = self.split0(xxx, split_size_or_sections=self.TIME_MIX_EXTRA_DIM, dim=-1)
        mw = self.add_time_maa_w(self.matmul_time_maa_w2_0(mw), self.time_maa_w)
        mk = self.add_time_maa_k(self.matmul_time_maa_w2_1(mk), self.time_maa_k)
        mv = self.add_time_maa_v(self.matmul_time_maa_w2_2(mv), self.time_maa_v)
        mr = self.add_time_maa_r(self.matmul_time_maa_w2_3(mr), self.time_maa_r)
        mg = self.add_time_maa_g(self.matmul_time_maa_w2_4(mg), self.time_maa_g)

        mw = self.mul_time_maa_w(sx, mw)
        mk = self.mul_time_maa_k(sx, mk)
        mv = self.mul_time_maa_v(sx, mv)
        mr = self.mul_time_maa_r(sx, mr)
        mg = self.mul_time_maa_g(sx, mg)

        mw = self.add_w_state0(x, mw)
        mk = self.add_k_state0(x, mk)
        mv = self.add_v_state0(x, mv)
        mr = self.add_r_state0(x, mr)
        mg = self.add_g_state0(x, mg)

        receptance = self.receptance(mr).view(self.num_heads, 1, self.head_size)
        key = self.key(mk).view(self.num_heads, self.head_size, 1)
        value = self.value(mv).view(self.num_heads, 1, self.head_size)
        gate = self.silu0(self.gate(mg))

        mw = self.tanh1(self.matmul_time_decay_w1(mw))
        time_decay = self.matmul_time_decay_w2(mw)
        time_decay = self.add_time_decay0(self.time_decay, time_decay.view(self.num_heads, self.head_size, 1))
        time_decay = self.exp1(self.neg(self.exp0(time_decay)))

        # wkv
        kv = self.matmul_kv(key, value)
        wkv = self.add_time_first(self.mul_time_first(kv, self.time_first), state2)
        wkv = self.matmul_rkv(receptance, wkv).view(1, self.num_heads, 1, self.head_size)
        state2_out = self.add_time_decay1(kv, self.mul_time_decay(state2, time_decay))

        x = self.ln_x(wkv).view(1, 1, self.hidden_size)
        x = self.mul_ln_x(x, self.ln_x_weight)
        x = self.add_ln_x(x, self.ln_x_bias)
        x = self.mul_attention(x, gate)
        x = self.output(x)

        return last_x + x, state1_out, state2_out

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
            self.value.weight = nn.Parameter(state_dict[prefix + 'value.weight']) / (2 ** int(layer_id // rescale_layer))
        else:
            self.value.weight = nn.Parameter(state_dict[prefix + 'value.weight'])

        self.ln_2                   = nn.LayerNorm(hidden_size, eps=1e-5)
        self.ln_2_w                 = nn.Parameter(state_dict[f'blocks.{layer_id}.ln2.weight'])
        self.ln_2_b                 = nn.Parameter(state_dict[f'blocks.{layer_id}.ln2.bias'])
        self.mul_ln_2               = op.Multiply()
        self.add_ln_2               = op.Add()

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

    def forward(self, x, state):
        last_x = x
        x = self.ln_2(x)
        x = self.mul_ln_2(x, self.ln_2_w)
        x = self.add_ln_2(x, self.ln_2_b)

        sx = self.sub_shifted(state, x)
        xk = self.add_time_maa_k(x, self.mul_time_maa_k(sx, self.time_maa_k))
        xr = self.add_time_maa_r(x, self.mul_time_maa_r(sx, self.time_maa_r))

        key = self.pow(self.relu(self.key(xk)), 2)
        value = self.value(key)
        receptance = self.sigmoid(self.receptance(xr))

        return self.mul_rv(receptance, value) + last_x, x