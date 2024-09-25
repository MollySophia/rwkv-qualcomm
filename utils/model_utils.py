import torch
import contextlib
import sys
import os
import pathlib
import onnx
import re
from .split_onnx import OnnxSplitter, save_model

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

@contextlib.contextmanager
def freeze_quantizers(quantizers):
    originally_frozen = [q.is_encoding_frozen for q in quantizers]
    try:
        for q in quantizers:
            q._is_encoding_frozen = True
        yield
    finally:
        for q, frozen in zip(quantizers, originally_frozen):
            q._is_encoding_frozen = frozen

def to_device(t, device):
    if isinstance(t, torch.Tensor):
        return t.detach().clone().to(device)
    if isinstance(t, tuple):
        return tuple([to_device(i, device) for i in t])
    if isinstance(t, list):
        return [to_device(i, device) for i in t]
    if isinstance(t, dict):
        return {k:to_device(v, device) for k,v in t.items()}
    return t

def to_cpu(t):
    return to_device(t, torch.device('cpu'))

def get_input_output_names(model_cfg):
    num_layers, _, _ = extract_info_from_model_cfg(model_cfg)
    def _get_state_names(sfx, n_layers):
        all = []
        for i in range(n_layers):
            for j in range(0, 3):
                all.extend([f'layer{i}_state{j}_{sfx}'])
        return all

    input_names = ['input_ids']
    input_names += _get_state_names('in', num_layers)
    output_names = ['logits']
    output_names += _get_state_names('out', num_layers)
    return input_names, output_names

def _qnn_name(name, deco_digit=True):
    name = f'_{name}' if deco_digit and name.isdigit() else name
    #name = name.replace('.', '_')
    name = name.replace('/', '-')
    return name

def get_onnx_input_output_names(onnxfile, onnxmodel=None, deco_digit=True):
    onnxmodel = _load_model(onnxfile) if onnxmodel is None else onnxmodel
    input_names = [_qnn_name(i.name, deco_digit=deco_digit) for i in onnxmodel.graph.input]
    output_names = [_qnn_name(i.name, deco_digit=deco_digit) for i in onnxmodel.graph.output]
    return input_names, output_names

def _load_model(onnxfile, load_external_data=False, model_cache={}):
    if onnxfile not in model_cache:
        print(f'Loading {onnxfile}', file=sys.stderr)
        model_cache[onnxfile] = onnx.load(onnxfile, load_external_data=load_external_data)
    return model_cache[onnxfile]

def split_onnx_by_names(onnxfile, modelname, *list_of_output_tensors, output_dir='.', onnxmodel=None):
    onnxmodel = _load_model(onnxfile, load_external_data=False) if onnxmodel is None else onnxmodel
    splitter = OnnxSplitter(onnxmodel, verbose=True)
    base_dir = os.path.dirname(onnxfile)
    using_external_data = OnnxSplitter.is_using_external_data(onnxmodel)

    list_of_output_tensors = [i.split(',') for i in list_of_output_tensors]
    num_splits = len(list_of_output_tensors) + 1
    pathlib.Path(f'{output_dir}/split_onnx').mkdir(parents=True, exist_ok=True)

    # 1. split model
    new_model_info = []
    for i, subgraph in enumerate(splitter.split(list_of_output_tensors)):
        new_basename = f'{modelname}_{i+1}_of_{num_splits}'
        input_tensor_names = [i.name for i in subgraph.input]
        output_tensor_names = [i.name for i in subgraph.output]
        new_model_info.append([new_basename, input_tensor_names, output_tensor_names])

        submodel = onnx.helper.make_model(subgraph, opset_imports=onnxmodel.opset_import)
        if not using_external_data and submodel.ByteSize() < onnx.checker.MAXIMUM_PROTOBUF:
            onnx.checker.check_model(submodel)

        if using_external_data:
            onnx.load_external_data_for_model(submodel, base_dir=base_dir)

        newonnxfile = f'{output_dir}/split_onnx/{new_basename}.onnx'
        print(f'Saving {newonnxfile}')
        save_model(submodel, newonnxfile, using_external_data)
        return

def _get_lm_head_sizes(onnxmodel):
    lm_head_weight = [i for i in onnxmodel.graph.initializer if 'lm_head' in i.name and 'weight' in i.name]
    if not lm_head_weight: #RWKV
        lm_head_weight = [i for i in onnxmodel.graph.initializer if 'head' in i.name and 'weight' in i.name]
    if len(lm_head_weight[0].dims) == 2:
        embedding_size, vocab_size = lm_head_weight[0].dims
    else:
        vocab_size, embedding_size, _, _ = lm_head_weight[0].dims
    return embedding_size, vocab_size

def get_per_layer_name_formats(onnxnodes):
    if all(i in onnxnodes for i in ('transformer.h.0.ln_1', 'transformer.ln_f')):
        block_input, attn_output, ln_f = 'transformer.h.{}.ln_1', 'transformer.h.{}.ln_2', 'transformer.ln_f'
    elif all(i in onnxnodes for i in ('model.layers.0.input_layernorm', 'model.norm')):
        # these are for the legacy export of LLaMa model
        block_input, attn_output, ln_f = 'model.layers.{}.input_layernorm', 'model.layers.{}.post_attention_layernorm', 'model.norm'
    elif all(i in onnxnodes for i in ('model_layers_0_input_layernorm_Pow', 'model_norm_Pow')):
        block_input, attn_output, ln_f = 'model_layers_{}_input_layernorm_Pow', 'model_layers_{}_post_attention_layernorm_Pow', 'model_norm_Pow'
    elif all(i in onnxnodes for i in ('model.layers.0.input_layernorm.cast', 'model.norm.cast')):
        # aimet onnx saver?
        block_input, attn_output, ln_f = 'model.layers.{}.input_layernorm.cast', 'model.layers.{}.post_attention_layernorm.cast', 'model.norm.cast'
    elif all(i in onnxnodes for i in ('/model/layers.0/input_layernorm/cast/Cast', '/model/norm/cast/Cast')):
        # torch onnx export
        block_input, attn_output, ln_f = '/model/layers.{}/input_layernorm/cast/Cast', '/model/layers.{}/post_attention_layernorm/cast/Cast', '/model/norm/cast/Cast'
    elif all(i in onnxnodes for i in ('/model_layers_0_input_layernorm_Cast/Cast', '/model_norm_Cast/Cast')):
        block_input, attn_output, ln_f = '/model_layers_{}_input_layernorm_Cast/Cast', '/model_layers_{}_post_attention_layernorm_Cast/Cast', '/model_norm_Cast/Cast'
       # torch onnx export Aimet1.30 debug
    elif all(i in onnxnodes for i in ('/model_layers_0_input_layernorm_Pow/Pow', '/model_norm_Pow/Pow')):
        block_input, attn_output, ln_f = '/model_layers_{}_input_layernorm_Pow/Pow', '/model_layers_{}_post_attention_layernorm_Pow/Pow', '/model_norm_Pow/Pow'
    elif all(i in onnxnodes for i in ('rwkv.blocks.0.ln1', 'rwkv.blocks.0.add_feed_forward')):
        block_input, attn_output, ln_f = 'rwkv.blocks.{}.ln1', 'rwkv.blocks.{}.add_feed_forward', 'rwkv.ln_out'
    else:
        raise RuntimeError(f"Unexpected ONNX model, couldn't get the per-layer names")
    return block_input, attn_output, ln_f

def get_per_layer_output_names(onnxfile, onnxmodel=None, deco_digit=True, include_first_layer=False):
    '''
    Return the output tensor names of all transformer blocks in the model
    '''
    onnxmodel = _load_model(onnxfile) if onnxmodel is None else onnxmodel

    per_layer_output_names = []

    nodes = {i.name: i for i in onnxmodel.graph.node}

    # new pipeline
    fmt_node_name, _, ln_f = get_per_layer_name_formats(nodes)

    start_from = 0 if include_first_layer else 1
    for i in range(start_from, 10000): # skip the first because it's the beginning of the block
        name = fmt_node_name.format(i)
        if name not in nodes:
            break
        per_layer_output_names.append(nodes[name].input[0]) # use input

    # add the end of the last transformer block, to have the same number names as the number of layers
    assert ln_f in nodes, "Failed to find the end of the last transformer block"
    per_layer_output_names.append(nodes[ln_f].input[0]) # use input

    return per_layer_output_names


def split_onnx(onnxfile, modelname, num_splits, output_dir='./', split_embedding=False):
    def _is_rwkv_state(layer, name):
        return re.search(f'layer{layer}_state', name) != None

    num_splits = int(num_splits)

    onnxmodel = _load_model(onnxfile, load_external_data=False)
    input_names, output_names = get_onnx_input_output_names(onnxfile, onnxmodel=onnxmodel, deco_digit=False)
    per_layer_output_names = get_per_layer_output_names(onnxfile, onnxmodel=onnxmodel, deco_digit=False, include_first_layer=split_embedding)
    print('Per_layer_output_names:', per_layer_output_names)


    # Infer the shape of per-layer tensors
    input_ids, = [i for i in onnxmodel.graph.input if i.name == 'input_ids']
    batch_size, seq_length = [i.dim_value for i in input_ids.type.tensor_type.shape.dim]

    embedding_size, vocab_size = _get_lm_head_sizes(onnxmodel)
    print(f'Using per-layer output shape: {[batch_size, seq_length, embedding_size]}')

    per_layer_output_value_info = [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [batch_size, seq_length, embedding_size]) for name in per_layer_output_names]
    onnxmodel.graph.value_info.extend(per_layer_output_value_info)

    num_layers = len(per_layer_output_names)
    num_layers_per_split = num_layers//num_splits
    past_key_values = {layer:[output for output in output_names if _is_rwkv_state(layer, output)] for layer in range(num_layers)}


    names_to_split = []
    if split_embedding:
        names_to_split.append(per_layer_output_names[0])
        per_layer_output_names.pop(0)

    num_layers = len(per_layer_output_names)
    num_layers_per_split = (num_layers // (num_splits-1)) if split_embedding else (num_layers // num_splits)
    past_key_values = {layer:[output for output in output_names if _is_rwkv_state(layer, output)] for layer in range(num_layers)}

    for layer_end in range(num_layers_per_split,num_layers,num_layers_per_split):
        outputs = [per_layer_output_names[layer_end-1]]
        for layer in range(layer_end-num_layers_per_split, layer_end):
            outputs += past_key_values[layer]
        names_to_split.append(','.join(outputs))

    print('Names_to_split', names_to_split)
    assert num_splits == len(names_to_split)+1, f"Failed to split into {num_splits} pieces!"
    return split_onnx_by_names(onnxfile, modelname, *names_to_split, output_dir=output_dir, onnxmodel=onnxmodel)
