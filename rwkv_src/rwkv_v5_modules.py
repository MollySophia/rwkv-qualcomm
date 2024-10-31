import torch
import torch.nn as nn
import rwkv_src.elemwise_ops as op

class Rwkv5SelfAttention(nn.Module):
    def __init__(self, state_dict, hidden_size, head_size, version=5.2, layer_id=0, rescale_layer=0):
        super().__init__()
        prefix = f'blocks.{layer_id}.att.'
        self.layer_id = layer_id
        self.num_heads = hidden_size // head_size
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.version = version

        self.time_mix_k = nn.Parameter(state_dict[prefix + 'time_mix_k'].flatten())
        self.time_mix_v = nn.Parameter(state_dict[prefix + 'time_mix_v'].flatten())
        self.time_mix_r = nn.Parameter(state_dict[prefix + 'time_mix_r'].flatten())
        if version != 5.0:
            self.time_mix_g = nn.Parameter(state_dict[prefix + 'time_mix_g'].flatten())

        td = torch.exp(-torch.exp(state_dict[prefix + 'time_decay'])).reshape(-1, 1, 1)
        if version == 5.2:
            td = td.reshape(self.num_heads, -1, 1)
        self.time_decay = nn.Parameter(td)

        try:
            tf = state_dict[prefix + 'time_faaaa'].reshape(-1, 1, 1)
        except:
            tf = torch.exp(state_dict[prefix + 'time_first'].float()).reshape(-1, 1, 1)
        
        if version == 5.2:
            tf = tf.reshape(self.num_heads, -1, 1)
        self.time_first = nn.Parameter(tf)

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

        self.sub_time_mix_k         = op.Subtract()
        self.mul_time_mix_k_1       = op.Multiply()
        self.mul_time_mix_k_2       = op.Multiply()
        self.add_time_mix_k         = op.Add()
        self.sub_time_mix_v         = op.Subtract()
        self.mul_time_mix_v_1       = op.Multiply()
        self.mul_time_mix_v_2       = op.Multiply()
        self.add_time_mix_v         = op.Add()
        self.sub_time_mix_r         = op.Subtract()
        self.mul_time_mix_r_1       = op.Multiply()
        self.mul_time_mix_r_2       = op.Multiply()
        self.add_time_mix_r         = op.Add()
        if version != 5.0:
            self.sub_time_mix_g         = op.Subtract()
            self.mul_time_mix_g_1       = op.Multiply()
            self.mul_time_mix_g_2       = op.Multiply()
            self.add_time_mix_g         = op.Add()


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

        self.matmul_kv              = op.MatMul()
        self.mul_time_first         = op.Multiply()
        self.add_time_first         = op.Add()
        self.mul_scale_kv           = op.Multiply()
        self.matmul_rkv             = op.MatMul()
        self.mul_time_decay         = op.Multiply()
        self.add_time_decay         = op.Add()
        if version != 5.0:
            self.mul_attention          = op.Multiply()

        self.tanh0                  = op.Tanh()
        self.tanh1                  = op.Tanh()
        self.silu0                  = op.SiLU()
        self.split0                 = op.Split()
        self.exp0                   = op.Exponential()
        self.exp1                   = op.Exponential()
        self.neg                    = op.Neg()
    
    def forward(self, x, state1, state2):
        last_x = x
        x = self.ln_1(x)
        x = self.mul_ln_1(x, self.ln_1_w)
        x = self.add_ln_1(x, self.ln_1_b)
        state1_out = x
        
        xk = self.mul_time_mix_k_1(state1, self.sub_time_mix_k(1, self.time_mix_k))
        xk = self.add_time_mix_k(xk, self.mul_time_mix_k_2(x, self.time_mix_k))

        xv = self.mul_time_mix_v_1(state1, self.sub_time_mix_v(1, self.time_mix_v))
        xv = self.add_time_mix_v(xv, self.mul_time_mix_v_2(x, self.time_mix_v))

        xr = self.mul_time_mix_r_1(state1, self.sub_time_mix_r(1, self.time_mix_r))
        xr = self.add_time_mix_r(xr, self.mul_time_mix_r_2(x, self.time_mix_r))

        if self.version != 5.0:
            xg = self.mul_time_mix_g_1(state1, self.sub_time_mix_g(1, self.time_mix_g))
            xg = self.add_time_mix_g(xg, self.mul_time_mix_g_2(x, self.time_mix_g))

        receptance = self.receptance(xr).view(self.num_heads, 1, self.head_size)
        key = self.key(xk).view(self.num_heads, self.head_size, 1)
        value = self.value(xv).view(self.num_heads, 1, self.head_size)
        if self.version != 5.0:
            gate = self.silu0(self.gate(xg))

        # wkv
        kv = self.matmul_kv(key, value)
        wkv = self.add_time_first(self.mul_time_first(kv, self.time_first), state2)
        wkv = self.matmul_rkv(receptance, wkv).view(1, self.num_heads, 1, self.head_size)
        state2_out = self.add_time_decay(kv, self.mul_time_decay(state2, self.time_decay))

        x = self.ln_x(wkv).view(1, 1, self.hidden_size)
        x = self.mul_ln_x(x, self.ln_x_weight)
        x = self.add_ln_x(x, self.ln_x_bias)
        if self.version != 5.0:
            x = self.mul_attention(x, gate)
        x = self.output(x)

        return last_x + x, state1_out, state2_out

class Rwkv5FeedForward(nn.Module):
    def __init__(self, state_dict, hidden_size, intermediate_size, layer_id=0, rescale_layer=0):
        super().__init__()
        prefix = f'blocks.{layer_id}.ffn.'
        self.layer_id = layer_id
        self.hidden_size = hidden_size

        self.time_mix_k = nn.Parameter(state_dict[prefix + 'time_mix_k'])
        self.time_mix_r = nn.Parameter(state_dict[prefix + 'time_mix_r'])

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
        self.ln_2_w                 = nn.Parameter(state_dict[f'blocks.{layer_id}.ln2.weight'])
        self.ln_2_b                 = nn.Parameter(state_dict[f'blocks.{layer_id}.ln2.bias'])
        self.mul_ln_2               = op.Multiply()
        self.add_ln_2               = op.Add()

        #Add new Op def
        self.sub_time_mix_k         = op.Subtract()
        self.mul_time_mix_k_1       = op.Multiply()
        self.mul_time_mix_k_2       = op.Multiply()
        self.add_time_mix_k         = op.Add()
        self.sub_time_mix_r         = op.Subtract()
        self.mul_time_mix_r_1       = op.Multiply()
        self.mul_time_mix_r_2       = op.Multiply()
        self.add_time_mix_r         = op.Add()
        self.relu                   = nn.ReLU()
        self.pow                    = op.Pow()
        self.sigmoid                = nn.Sigmoid()
        self.mul_rv                 = op.Multiply()

    def forward(self, x, state):
        last_x = x
        x = self.ln_2(x)
        x = self.mul_ln_2(x, self.ln_2_w)
        x = self.add_ln_2(x, self.ln_2_b)

        xk = self.mul_time_mix_k_1(state, self.sub_time_mix_k(1, self.time_mix_k))
        xk = self.add_time_mix_k(xk, self.mul_time_mix_k_2(x, self.time_mix_k))
        xr = self.mul_time_mix_r_1(state, self.sub_time_mix_r(1, self.time_mix_r))
        xr = self.add_time_mix_r(xr, self.mul_time_mix_r_2(x, self.time_mix_r))

        key = self.pow(self.relu(self.key(xk)), 2)
        value = self.value(key)
        receptance = self.sigmoid(self.receptance(xr))

        return self.mul_rv(receptance, value) + last_x, x