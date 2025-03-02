import torch
import contextlib
import sys
import os
import pathlib
import onnx
import re
from .split_onnx import OnnxSplitter, save_model

def extract_info_from_model_cfg(model_cfg):
    return model_cfg.n_layer, model_cfg.n_head, model_cfg.n_embd

def get_dummy_state_kvcache(batch_size, model_cfg, device):
    def _cache(shape, fp16):
        if fp16:
            return torch.zeros(shape, dtype=torch.float16).to(device=device)
        else:
            return torch.zeros(shape).to(device=device)

    num_layers, num_heads, embed_dim = extract_info_from_model_cfg(model_cfg)
    head_size = embed_dim // num_heads

    state_0 = (batch_size, embed_dim)
    state_1 = (num_heads, head_size, head_size)
    state_2 = (batch_size, embed_dim)
 
    state = []
    for _ in range(0, num_layers):
        state += [_cache(state_0, model_cfg.fp16), _cache(state_1, model_cfg.fp16), _cache(state_2, model_cfg.fp16)]
    return state

def get_dummy_input_for_rwkv_causal_llm(batch_size, input_length, device, model_cfg=None):
    input_ids = torch.LongTensor([[0]*input_length for _ in range(batch_size)]).to(device)
    inputs = {'in0': input_ids, 'state': get_dummy_state_kvcache(batch_size, model_cfg, device)}
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
    splitter = OnnxSplitter(onnxmodel, verbose=False)
    base_dir = os.path.dirname(onnxfile)
    using_external_data = OnnxSplitter.is_using_external_data(onnxmodel)

    list_of_output_tensors = [i.split(',') for i in list_of_output_tensors]
    num_splits = len(list_of_output_tensors) + 1
    pathlib.Path(f'{output_dir}/split_onnx').mkdir(parents=True, exist_ok=True)

    # 1. split model
    new_model_info = []
    for i, subgraph in enumerate(splitter.split(list_of_output_tensors)):
        new_basename = f'{modelname}_chunk{i+1}of{num_splits}'
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
        lm_head_node = [i for i in onnxmodel.graph.node if 'head' in i.name]
        lm_head_weight_name = lm_head_node[0].input[1]
        lm_head_weight = [i for i in onnxmodel.graph.initializer if lm_head_weight_name in i.name]
    if len(lm_head_weight[0].dims) == 2:
        embedding_size, vocab_size = lm_head_weight[0].dims
    else:
        vocab_size, embedding_size, _, _ = lm_head_weight[0].dims
    return embedding_size, vocab_size

def get_per_layer_name_formats(onnxnodes):
    if all(i in onnxnodes for i in ('/blocks.0/att/ln_1/LayerNormalization', '/blocks.0/ffn/add_feed_forward/Add')):
        block_input, attn_output, ln_f = '/blocks.{}/att/ln_1/LayerNormalization', '/blocks.{}/ffn/add_feed_forward/Add', '/ln_out/LayerNormalization'
    elif all(i in onnxnodes for i in ('blocks.0.att.ln_1', 'blocks.0.ffn.add_feed_forward')):
        block_input, attn_output, ln_f = 'blocks.{}.att.ln_1', 'blocks.{}.ffn.add_feed_forward', 'ln_out'
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

    # print('Names_to_split', names_to_split)
    assert num_splits == len(names_to_split)+1, f"Failed to split into {num_splits} pieces!"
    return split_onnx_by_names(onnxfile, modelname, *names_to_split, output_dir=output_dir, onnxmodel=onnxmodel)

from torch.onnx import register_custom_op_symbolic
def onnx_custom_wkv6(g, k, v, r, state2, time_first, time_decay):
    n_head = state2.type().sizes()[0]
    head_size = state2.type().sizes()[1]
    out1, out2 = g.op("rwkv::wkv6", k, v, r, state2, time_first, time_decay, outputs=2)
    return out1.setType(k.type().with_dtype(torch.float32).with_sizes([k.type().sizes()[0], n_head, head_size])),\
        out2.setType(k.type().with_dtype(torch.float32).with_sizes([n_head, head_size, head_size]))

def onnx_custom_wkv7(g, r, w, k, v, a, b, state):
    n_head = state.type().sizes()[0]
    head_size = state.type().sizes()[1]
    out1, out2 = g.op("rwkv::wkv7", r, w, k, v, a, b, state, outputs=2)
    return out1.setType(k.type().with_dtype(torch.float32).with_sizes([k.type().sizes()[0], n_head, head_size])),\
        out2.setType(k.type().with_dtype(torch.float32).with_sizes([n_head, head_size, head_size]))

def norm(g, self):
    return g.op("LpNormalization", self, p_i=2, axis_i=-1)

def register_customop_symbols():
    register_custom_op_symbolic('customop::l2norm', norm, 4)
    register_custom_op_symbolic("rwkv::wkv6", onnx_custom_wkv6, 9)
    register_custom_op_symbolic("rwkv::wkv7", onnx_custom_wkv7, 9)