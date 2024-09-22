import torch

model2config_lut = {
    'Rwkv6ForCausalLM': ('num_hidden_layers', 'num_attention_heads', 'attention_hidden_size'),
}

def extract_info_from_model_cfg(model_cfg):
    return [getattr(model_cfg, key) for key in model2config_lut[model_cfg.architectures[0]]]

def get_dummy_state_kvcache(batch_size, model_cfg, device):
    def _cache(shape):
        return torch.zeros(shape).to(device=device)

    num_layers, num_heads, embed_dim = extract_info_from_model_cfg(model_cfg)

    state_0 = (batch_size, embed_dim)
    state_1 = (batch_size, embed_dim//num_heads, num_heads, num_heads)
    state_2 = (batch_size, embed_dim)
 
    state = []
    for _ in range(0, num_layers):
        state += [_cache(state_0), _cache(state_1), _cache(state_2)]
    return state

def get_dummy_input_for_rwkv_causal_llm(batch_size, input_length, device, model_cfg=None):
    input_ids = torch.LongTensor([[0]*input_length for _ in range(batch_size)]).to(device)
    
    inputs = {
        'input_ids': input_ids,
        'attention_mask': input_ids,
        'inputs_embeds': input_ids.float(),
        'state': get_dummy_state_kvcache(batch_size, model_cfg, device),
    }
    return inputs