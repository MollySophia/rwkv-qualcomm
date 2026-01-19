from rwkv_src.rwkv_model import RWKV_RNN, make_chunks
from rwkv_src.rwkv_v7_modules_conv import Wkv7, L2Norm
from utils.model_utils import register_customop_symbols, get_dummy_state_kvcache, apply_activation_quant_override, dummy_quant_override
import types
import os
import torch
import numpy as np
import argparse
import json
import copy
from pathlib import Path
import onnx
from onnx import shape_inference
from onnx.external_data_helper import convert_model_to_external_data
import onnx_graphsurgeon as gs
# from aimet_torch.onnx_utils import OnnxSaver
import aimet_torch.onnx_utils
aimet_torch.onnx_utils.EXPORT_TO_ONNX_DIRECT = True

register_customop_symbols()

parser = argparse.ArgumentParser(description='Convert model')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser.add_argument('--chunks', type=int, default=2, help='Number of chunks')
parser.add_argument('--qnn_float_width', type=int, default=16, help='QNN float width')
parser.add_argument('--ext_embedding', action='store_true', default=False, help='Use external embedding')
parser.add_argument('--ext_lmhead', action='store_true', default=False, help='Use external head')
parser.add_argument('--quant_encodings', type=Path, help='Path to quant encodings')
parser.add_argument('--prefill_model', action='store_true', help='Convert model for sequential prefill')
parser.add_argument('--prefill_seq_length', type=int, default=16, help='Prefill sequence length')
parser.add_argument('--wkv_customop', action='store_true', help='Use custom op for wkv')
parser.add_argument('--no_cleanup', action='store_true', help='Do not cleanup onnx files')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--save_input_vectors', action='store_true', help='Save input vectors')
parser.add_argument('--input_vectors_save_path', type=Path, default="test_vector", help='Path to save input vectors')
parser.add_argument('--heads_per_split', type=int, default=8, help='Number of heads per split')
parser.add_argument('--output_name', type=str, default=None, help='Output name for generated files (without extension). If not provided, uses input model filename.')
parser_args = parser.parse_args()

seq_length = parser_args.prefill_seq_length if parser_args.prefill_model else 1
batch_size = parser_args.batch_size

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False
model_args.fp16 = False
model_args.bf16 = False
model_args.wkv_customop = parser_args.wkv_customop
model_args.USE_EMBEDDING = False if parser_args.ext_embedding else True
model_args.MODEL_NAME = str(parser_args.model)
model_args.output_last = True
model_args.EXTERNAL_HEAD = True if parser_args.ext_lmhead else False
model_args.RESCALE_LAYER = 0
# model_args.heads_per_split = 2 if parser_args.prefill_model else 8
model_args.heads_per_split = parser_args.heads_per_split

model = make_chunks(parser_args.chunks, model_args) if parser_args.chunks > 1 else RWKV_RNN(model_args)
has_deepemb = model[0].args.has_deepemb if type(model) == list else model.args.has_deepemb
if has_deepemb:
    deep_emb_size = model[0].deep_emb[0].weight.shape[-1] if type(model) == list else model.deep_emb[0].weight.shape[-1]

qnn_sdk_root = os.environ["QNN_SDK_ROOT"]
if not qnn_sdk_root:
    print("Please set QNN_SDK_ROOT environment variable to the root of the Qualcomm Neural Processing SDK")
    exit(1)
os.path.exists("onnx") or os.mkdir("onnx")
if os.name == 'nt':
    qnn_tools_target = 'x86_64-windows-msvc'
else:
    qnn_tools_target = 'x86_64-linux-clang'

def save_input_vectors(input_tensors_list, input_vectors_save_path):
    input_vectors_save_path = str(input_vectors_save_path) + f"{'_prefill' if parser_args.prefill_model else '_bsz' + str(parser_args.batch_size) if parser_args.batch_size > 1 else ''}"
    os.path.exists(input_vectors_save_path) or os.mkdir(input_vectors_save_path)
    input_list = []
    for i in range(len(input_tensors_list)):
        input_tensors_list[i].cpu().numpy().astype(np.float32).tofile(input_vectors_save_path + f"/input_vector_{i}.bin")
        input_list.append(input_vectors_save_path + f"/input_vector_{i}.bin")
    with open(input_vectors_save_path + f"/input_list.txt", "w") as f:
        f.writelines([" ".join(input_list) + "\n"])
    print(f"Input vectors saved to {input_vectors_save_path}/input_list.txt")

if type(model) == list:
    args = model[0].args
    # Use output_name if provided, otherwise extract from input model path
    if parser_args.output_name:
        filename = parser_args.output_name
    else:
        filename = args.MODEL_NAME.split("/")[-1]
        # Remove extension if present
        if '.' in filename:
            filename = os.path.splitext(filename)[0]
    input_dtype = torch.float16 if args.fp16 else torch.float32

    if args.EXTERNAL_HEAD:
        model[-1].head.weight.detach().squeeze().cpu().numpy().astype(np.float16).tofile("onnx/" + filename + f"_chunk1of{len(model)}.fp16.head.weight")
        lm_head = torch.nn.Linear(args.n_embd, model[-1].head.weight.squeeze().shape[0], bias=False)
        lm_head.weight = torch.nn.Parameter(model[-1].head.weight.squeeze())
        torch.onnx.export(lm_head, (torch.zeros(1, args.n_embd, dtype=input_dtype),), "onnx/" + filename + f"_lmhead.onnx", opset_version=17, input_names=['in'], output_names=['out'])
        print(f"lmhead onnx model saved to onnx/{filename}_lmhead.onnx")

    states = get_dummy_state_kvcache(batch_size, args, model[0].device)
    if parser_args.quant_encodings:
        with open(parser_args.quant_encodings, 'r') as f:
            encodings_all = json.load(f)

        if parser_args.wkv_customop:
            for i in range(model[0].args.n_layer):
                apply_activation_quant_override(f'state{3*i+1}_in', dummy_quant_override, encodings_all)
                apply_activation_quant_override(f'state{3*i+1}_out', dummy_quant_override, encodings_all)
                apply_activation_quant_override(f'/blocks.{i}/att/wkv7/wkv/wkv7_output_0', dummy_quant_override, encodings_all)

    if not args.USE_EMBEDDING:
        if not parser_args.quant_encodings:
            model[0].emb_weight.cpu().numpy().astype(np.float16).tofile("onnx/" + filename + f"_chunk1of{len(model)}.fp16.emb")
        else:
            offset = 0
            scale = 0
            bitwidth = 0
            for v in encodings_all["param_encodings"]:
                if f'embedding.weight' in v['name']:
                    offset = v['offset'][0]
                    scale = v['scale'][0]
                    bitwidth = v['bw']
                    break

            true_max = pow(2, bitwidth) - 1
            enc_min = offset * scale 
            enc_max = (true_max + offset) * scale
            enc_range = enc_max - enc_min
            
            emb_quant = true_max * (model[0].emb_weight.cpu().squeeze() - enc_min) / enc_range
            emb_quant = emb_quant.clamp(0, true_max)
            emb_quant.numpy().astype(np.uint16).tofile("onnx/" + filename + f"_chunk1of{len(model)}.uint16.emb")
            print(f"Embedding quantized and saved to onnx/{filename}_chunk1of{len(model)}.uint16.emb")


    for i in range(len(model)):
        # Add prefill and bsz suffix to dirname to allow parallel execution
        dirname = "onnx/" + filename + f"_chunk{i+1}of{len(model)}"
        if parser_args.prefill_model:
            dirname += "_prefill"
        if parser_args.batch_size > 1:
            dirname += f"_bsz{parser_args.batch_size}"
        os.path.exists(dirname) or os.mkdir(dirname)
        if i == 0 and args.USE_EMBEDDING:
            in0 = torch.LongTensor([[1]*seq_length] * batch_size, device=model[i].device)
        else:
            in0 = torch.zeros(batch_size, seq_length, args.n_embd, dtype=input_dtype)

        inputs = [in0, [states[j] for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]]
        # input_name = ('in' if not parser_args.prefill_model else 'in_prefill') + f'_chunk{i+1}'
        input_name = 'in'
        if not args.USE_EMBEDDING and i == 0:
            input_name += '_embedding'
        if parser_args.prefill_model:
            input_name += '_prefill'
        if parser_args.batch_size > 1:
            input_name += '_bsz' + str(parser_args.batch_size)
        input_name += f'_chunk{i+1}'

        output_name = ('out' if not parser_args.prefill_model else 'out_prefill') + f'_chunk{i+1}'
        input_names = [input_name] + [f'state{j}_in' for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
        output_names = [output_name] + [f'state{j}_out' for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]

        encodings_chunk = None
        if i == 0:
            encodings_chunk = copy.deepcopy(encodings_all)
            for v in encodings_all["activation_encodings"]:
                if f'/blocks.{model[1].layer_begin-1}/ffn/add_feed_forward/Add_output_0' in v['name']:
                    apply_activation_quant_override(output_name, v, encodings_chunk)
            if not args.USE_EMBEDDING:
                for v in encodings_all["param_encodings"]:
                    if f'embedding.weight' in v['name']:
                        apply_activation_quant_override(input_name, v, encodings_chunk)
                        break
        else:
            encodings_chunk = {"activation_encodings": [], "excluded_layers": [], "param_encodings": [], "quantizer_args": encodings_all["quantizer_args"], "version": encodings_all["version"]}
            for v in encodings_all["activation_encodings"]:
                try:
                    encoding_block_id = int(v['name'].split(".")[1].split("/")[0]) if 'block' in v['name'] else args.n_layer-1
                except ValueError:
                    encoding_block_id = int(v['name'].split(".")[1]) if 'block' in v['name'] else args.n_layer-1
                if any([
                    'state' in v['name'],
                    (encoding_block_id >= model[i].layer_begin and encoding_block_id < model[i].layer_end),
                    v['name'] == '/blocks.0/att/post_permute_v/Transpose_output_0',
                    v['name'] == '/blocks.0/att/value/Conv_output_0',
                    v['name'] == f'/blocks.{model[i].layer_begin-1}/ffn/add_feed_forward/Add_output_0'
                ]):
                    if '/blocks.0/att/post_permute_v/Transpose_output_0' in v['name'] or '/blocks.0/att/value/Conv_output_0' in v['name']:
                        apply_activation_quant_override(f'v_first_in{"_prefill" if parser_args.prefill_model else ""}_chunk{i+1}', v, encodings_chunk)
                    elif f'/blocks.{model[i].layer_begin-1}/ffn/add_feed_forward/Add_output_0' in v['name']:
                        apply_activation_quant_override(input_name, v, encodings_chunk)
                    else:
                        apply_activation_quant_override(v['name'].replace(f"blocks.{encoding_block_id}", f"blocks.{encoding_block_id-model[i].layer_begin}"), v, encodings_chunk)
                if args.EXTERNAL_HEAD and i == len(model) - 1:
                    for v in encodings_all["activation_encodings"]:
                        if 'ln_out/LayerNormalization_output_0' in v['name']:
                            apply_activation_quant_override(output_name, v, encodings_chunk)
                            break
            for v in encodings_all["param_encodings"]:
                if 'embedding' in v['name']:
                    encoding_block_id = 0
                else:
                    encoding_block_id = int(v['name'].split(".")[1]) if 'block' in v['name'] else args.n_layer-1
                if 'state' in v['name'] or (encoding_block_id >= model[i].layer_begin and encoding_block_id < model[i].layer_end):
                    apply_activation_quant_override(v['name'].replace(f"blocks.{encoding_block_id}", f"blocks.{encoding_block_id-model[i].layer_begin}"), v, encodings_chunk)

        if args.version == 7:
            if i == 0:
                output_names += [('v_first_out' if not parser_args.prefill_model else 'v_first_out_prefill') + f'_chunk{i+1}']
            else:
                input_names += [('v_first_in' if not parser_args.prefill_model else 'v_first_in_prefill') + f'_chunk{i+1}']
                inputs += [torch.zeros(batch_size, seq_length, args.n_embd, dtype=input_dtype)]
            if has_deepemb:
                inputs += [[torch.zeros(batch_size, seq_length, deep_emb_size, dtype=input_dtype) for _ in range(model[i].layer_begin, model[i].layer_end)]]
                input_names += [f's_emb{j}_in' for j in range(model[i].layer_begin, model[i].layer_end)]

        if parser_args.save_input_vectors and i == 0:
            save_input_vectors([in0] + inputs[1], parser_args.input_vectors_save_path)
        onnx_output_path = f"{dirname}/{filename}"
        if parser_args.ext_embedding:
            onnx_output_path += "_embedding"
        if parser_args.prefill_model:
            onnx_output_path += "_prefill"
        if parser_args.batch_size > 1:
            onnx_output_path += "_bsz" + str(parser_args.batch_size)
        onnx_output_path += f"_chunk{i+1}of{len(model)}.onnx"
        aimet_torch.onnx_utils.OnnxSaver.create_onnx_model_with_pytorch_layer_names(onnx_output_path, model[i], tuple(inputs),
            False, None, {'input_names': input_names, 'output_names': output_names, 'opset_version': 17})
        onnxmodel = onnx.load(onnx_output_path, load_external_data=True)
        graph = gs.import_onnx(onnxmodel)
        # set output shape for wkv7_output
        heads_per_split = args.n_head if model_args.heads_per_split == -1 else model_args.heads_per_split
        for k, v in graph.tensors().items():
            if "wkv7_output_x_output_0" in k:
                graph.tensors()[k].to_variable(dtype=np.float32, shape=[seq_length, heads_per_split, 1, args.head_size])
            elif "wkv7_output_state_output_0" in k:
                graph.tensors()[k].to_variable(dtype=np.float32, shape=[1, heads_per_split, args.head_size, args.head_size])
            elif "wkv7_output_0" in k:
                graph.tensors()[k].to_variable(dtype=np.float32, shape=[1, heads_per_split, args.head_size + seq_length, args.head_size])

        onnxmodel = gs.export_onnx(graph)
        convert_model_to_external_data(onnxmodel)
        onnx.save(onnxmodel, onnx_output_path)
        shape_inference.infer_shapes_path(onnx_output_path)
        print(f"onnx model chunk{i} saved to {onnx_output_path}")

        for j in range(model[i].layer_end - model[i].layer_begin):
            apply_activation_quant_override(f'state{3*(j + model[i].layer_begin)+1}_in', dummy_quant_override, encodings_chunk)
            apply_activation_quant_override(f'state{3*(j + model[i].layer_begin)+1}_out', dummy_quant_override, encodings_chunk)
            for split in range(1 if args.heads_per_split == -1 else args.n_head // args.heads_per_split):
                apply_activation_quant_override(f'/blocks.{j}/att/heads.{split}/wkv7/wkv/wkv7_output_0', dummy_quant_override, encodings_chunk)
                apply_activation_quant_override(f'/blocks.{j}/att/heads.{split}/wkv7/wkv_output_state/wkv7_output_state_output_0', dummy_quant_override, encodings_chunk)
                if parser_args.batch_size > 1:
                    x_encoding = None
                    for e in encodings_chunk['activation_encodings']:
                        if "wkv7_output_x_output_0" in e['name'] and f"blocks.{j}/att/heads.{split}" in e['name']:
                            x_encoding = e
                            break
                    for k, v in graph.tensors().items():
                        if ("wkv7_output_state_output_0" in k or "wkv7_output_0" in k) and f"blocks.{j}" in k:
                            apply_activation_quant_override(k, dummy_quant_override, encodings_chunk)
                        elif "wkv7_output_x_output_0" in k and f"blocks.{j}/att/heads.{split}" in k:
                            apply_activation_quant_override(k, x_encoding, encodings_chunk)
        with open(f"{dirname}/quant_encodings_chunk{i}.encodings", "w") as f:
            json.dump(encodings_chunk, f, sort_keys=True, indent=4)

    print("Converting and compiling QNN models...")
    for i in range(len(model)):
        # Add prefill and bsz suffix to dirname to allow parallel execution
        dirname = "onnx/" + filename + f"_chunk{i+1}of{len(model)}"
        if parser_args.prefill_model:
            dirname += "_prefill"
        if parser_args.batch_size > 1:
            dirname += f"_bsz{parser_args.batch_size}"
        onnx_output_path = f"{dirname}/{filename}"
        if parser_args.ext_embedding:
            onnx_output_path += "_embedding"
        if parser_args.prefill_model:
            onnx_output_path += "_prefill"
        if parser_args.batch_size > 1:
            onnx_output_path += "_bsz" + str(parser_args.batch_size)
        onnx_output_path += f"_chunk{i+1}of{len(model)}.onnx"
        os.path.exists(dirname) or os.mkdir(dirname)

        states_layout = "NONTRIVIAL"
        converter_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qairt-converter --input_network {onnx_output_path} "
        converter_cmd += " ".join([f'--source_model_input_layout "state{3*j+1}_in" "{states_layout}"' for j in range(model[i].layer_begin, model[i].layer_end)])
        converter_cmd += f" --float_bitwidth {parser_args.qnn_float_width} --float_bias_bitwidth {parser_args.qnn_float_width}"
        # if args.version == 7:
        #     converter_cmd += f' --source_model_input_layout "v_first_in{"_prefill" if parser_args.prefill_model else ""}_chunk{i+1}" "NONTRIVIAL"'
        if i != 0:
            converter_cmd += f' --source_model_input_layout "{input_name}" "NFC"'

        if parser_args.quant_encodings:
            converter_cmd += f" --quantization_overrides {dirname}/quant_encodings_chunk{i}.encodings"

        if model_args.wkv_customop:
            converter_cmd += " --op_package_config hexagon/CPU/RwkvWkvOpPackage/config/RwkvWkvOpPackageCPU.xml"
        print(converter_cmd)

        if os.name == 'nt':
            converter_cmd = "python " + converter_cmd
        result = os.system(converter_cmd)
        if result != 0:
            print(f"Error converting QNN dlc model: {converter_cmd}")
            exit(1)

        if parser_args.quant_encodings:
            print("Quantizing QNN dlc model...")
            quant_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qairt-quantizer -i {onnx_output_path.replace('.onnx', '.dlc')} -o {onnx_output_path.replace('.onnx', '.dlc')} --enable_float_fallback  --act_bitwidth 16 --bias_bitwidth 8"
            if os.name == 'nt':
                quant_cmd = "python " + quant_cmd
            print(quant_cmd)

            result = os.system(quant_cmd)
            if result != 0:
                print(f"Error quantizing QNN dlc model: {quant_cmd}")
                exit(1)
        if not parser_args.no_cleanup:
            for file in os.listdir(dirname):
                filepath = os.path.join(dirname, file)
                if not (file.endswith('.dlc') or file.endswith('.json') or file.endswith('.encodings') or file.endswith('.onnx') or file.endswith('.deepemb')):
                    if os.path.isfile(filepath):
                        os.remove(filepath)
else:
    args = model.args
    # Use output_name if provided, otherwise extract from input model path
    if parser_args.output_name:
        filename = parser_args.output_name
    else:
        filename = args.MODEL_NAME.split("/")[-1]
        # Remove extension if present
        if '.' in filename:
            filename = os.path.splitext(filename)[0]
    # Add prefill and bsz suffix to dirname to allow parallel execution
    dirname = "onnx/" + filename
    if parser_args.prefill_model:
        dirname += "_prefill"
    if parser_args.batch_size > 1:
        dirname += f"_bsz{parser_args.batch_size}"
    os.path.exists(dirname) or os.mkdir(dirname)
    if not args.USE_EMBEDDING:
        if not parser_args.quant_encodings:
            model.emb_weight.cpu().numpy().astype(np.float16).tofile(dirname + '/' + filename + ".fp16.emb")
        else:
            offset = 0
            scale = 0
            bitwidth = 0
            with open(parser_args.quant_encodings, 'r') as f:
                encodings_all = json.load(f)
                for v in encodings_all["param_encodings"]:
                    if f'embedding.weight' in v['name']:
                        offset = v['offset'][0]
                        scale = v['scale'][0]
                        bitwidth = v['bw']
                        break

            true_max = pow(2, bitwidth) - 1
            enc_min = offset * scale 
            enc_max = (true_max + offset) * scale
            enc_range = enc_max - enc_min

            emb_quant = true_max * (model.emb_weight.cpu().squeeze() - enc_min) / enc_range
            emb_quant = emb_quant.clamp(0, true_max)
            emb_quant.numpy().astype(np.uint16).tofile(dirname + '/' + filename + ".uint16.emb")
            print(f"Embedding quantized and saved to {dirname}/{filename}.uint16.emb")

    if has_deepemb:
        if not parser_args.quant_encodings:
            deepemb_weights_rearranged = torch.cat([model.s_emb_weights[i].unsqueeze(0) for i in range(model.layer_begin, model.layer_end)], dim=0)
            deepemb_weights_rearranged = deepemb_weights_rearranged.permute(1, 0, 2).contiguous().reshape(args.vocab_size, -1)
            deepemb_weights_rearranged.numpy().astype(np.float16).tofile(dirname + '/' + filename + ".fp16.deepemb")
            print(f"Deep embedding saved to {dirname}/{filename}.fp16.deepemb")
        else:
            # assert False, "TODO"
            deepemb_weights_list = []
            with open(parser_args.quant_encodings, 'r') as f:
                encodings_all = json.load(f)
            for i in range(model.layer_begin, model.layer_end):
                weight = model.deep_emb[i].weight.cpu().squeeze()
                offset = 0
                scale = 0
                bitwidth = 0
                for v in encodings_all["param_encodings"]:
                    if f'deep_emb.{i}.weight' in v['name']:
                        offset = v['offset'][0]
                        scale = v['scale'][0]
                        bitwidth = v['bw']
                        break
                if offset == 0 and scale == 0 and bitwidth == 0:
                    assert False, "No quantization encoding found for deep embedding"

                true_max = pow(2, bitwidth) - 1
                enc_min = offset * scale 
                enc_max = (true_max + offset) * scale
                enc_range = enc_max - enc_min
                weight = true_max * (weight - enc_min) / enc_range
                weight = weight.clamp(0, true_max)
                deepemb_weights_list.append(weight)

            deepemb_weights = torch.stack(deepemb_weights_list, dim=0).permute(1, 0, 2).contiguous().reshape(args.vocab_size, -1)
            deepemb_weights.numpy().astype(np.uint16).tofile(dirname + '/' + filename + ".uint16.deepemb")
            print(f"Deep embedding quantized and saved to {dirname}/{filename}.uint16.deepemb")

    input_dtype = torch.float16 if args.fp16 else torch.float32

    in0 = torch.LongTensor([[1]*seq_length]*batch_size) if args.USE_EMBEDDING else torch.zeros(batch_size, seq_length, args.n_embd, dtype=input_dtype)
    states = get_dummy_state_kvcache(batch_size, args, model.device)

    input_name = 'in'
    if parser_args.ext_embedding:
        input_name += '_embedding'
    if parser_args.prefill_model:
        input_name += '_prefill'
    if parser_args.batch_size > 1:
        input_name += '_bsz' + str(batch_size)
    output_name  = 'out'
    input_names = [input_name] + [f'state{i}_in' for i in range(3*args.n_layer)]
    output_names = [output_name] + [f'state{i}_out' for i in range(3*args.n_layer)]

    onnx_output_path = f"{dirname}/{filename}"
    if parser_args.ext_embedding:
        onnx_output_path += "_ext_embedding"
    if parser_args.prefill_model:
        onnx_output_path += "_prefill"
    if parser_args.batch_size > 1:
        onnx_output_path += "_bsz" + str(batch_size)

    inputs = [in0, states, None]
    if has_deepemb:
        inputs += [[torch.zeros(batch_size, seq_length, deep_emb_size, dtype=input_dtype) for _ in range(model.layer_begin, model.layer_end)]]
        input_names += [f's_emb{j}_in{("_bsz" + str(batch_size)) if parser_args.batch_size > 1 else ""}' if not parser_args.prefill_model else f's_emb{j}_in_prefill{("_bsz" + str(batch_size)) if parser_args.batch_size > 1 else ""}' for j in range(model.layer_begin, model.layer_end)]

    if parser_args.save_input_vectors:
        save_input_vectors([in0] + [*states], parser_args.input_vectors_save_path)

    onnx_output_path += ".onnx"
    aimet_torch.onnx_utils.OnnxSaver.create_onnx_model_with_pytorch_layer_names(onnx_output_path, model, tuple(inputs),
        False, None, {'input_names': input_names, 'output_names': output_names, 'opset_version': 17})
    layer_begin = model.layer_begin
    layer_end = model.layer_end
    del model
    onnxmodel = onnx.load(onnx_output_path, load_external_data=True)
    graph = gs.import_onnx(onnxmodel)
    # set output shape for wkv7_output
    heads_per_split = args.n_head if model_args.heads_per_split == -1 else model_args.heads_per_split
    for k, v in graph.tensors().items():
        if "wkv7_output_x_output_0" in k:
            graph.tensors()[k].to_variable(dtype=np.float32, shape=[seq_length, heads_per_split, 1, args.head_size])
        elif "wkv7_output_state_output_0" in k:
            graph.tensors()[k].to_variable(dtype=np.float32, shape=[1, heads_per_split, args.head_size, args.head_size])
        elif "wkv7_output_0" in k:
            graph.tensors()[k].to_variable(dtype=np.float32, shape=[1, heads_per_split, args.head_size + seq_length, args.head_size])
    onnxmodel = gs.export_onnx(graph)
    convert_model_to_external_data(onnxmodel)
    onnx.save(onnxmodel, onnx_output_path)
    shape_inference.infer_shapes_path(onnx_output_path)
    print(f"onnx model saved to {onnx_output_path}")

    encoding_path = str(parser_args.quant_encodings) if parser_args.quant_encodings else None
    # if encoding_path and (not args.USE_EMBEDDING or has_deepemb or not parser_args.wkv_customop):
    if encoding_path:
        with open(encoding_path, 'r') as f:
            encodings_all = json.load(f)
        if not args.USE_EMBEDDING:
            for v in encodings_all["param_encodings"]:
                if f'embedding.weight' in v['name']:
                    apply_activation_quant_override(input_name, v, encodings_all)
                    break
        if has_deepemb:
            for v in encodings_all["param_encodings"]:
                for i in range(layer_begin, layer_end):
                    if f'deep_emb.{i}.weight' in v['name']:
                        apply_activation_quant_override(f's_emb{i}_in{"_prefill" if parser_args.prefill_model else ""}', v, encodings_all)
                        break

        for i in range(layer_begin, layer_end):
            apply_activation_quant_override(f'state{3*i+1}_in', dummy_quant_override, encodings_all)
            apply_activation_quant_override(f'state{3*i+1}_out', dummy_quant_override, encodings_all)
            for split in range(1 if args.heads_per_split == -1 else args.n_head // args.heads_per_split):
                apply_activation_quant_override(f'/blocks.{i}/att/heads.{split}/wkv7/wkv/wkv7_output_0', dummy_quant_override, encodings_all)
                apply_activation_quant_override(f'/blocks.{i}/att/heads.{split}/wkv7/wkv_output_state/wkv7_output_state_output_0', dummy_quant_override, encodings_all)
                if parser_args.batch_size > 1:
                    x_encoding = None
                    for e in encodings_all['activation_encodings']:
                        if "wkv7_output_x_output_0" in e['name'] and f"blocks.{i}/att/heads.{split}" in e['name']:
                            x_encoding = e
                            break
                    for k, v in graph.tensors().items():
                        if ("wkv7_output_state_output_0" in k or "wkv7_output_0" in k) and f"blocks.{i}" in k:
                            apply_activation_quant_override(k, dummy_quant_override, encodings_all)
                        elif "wkv7_output_x_output_0" in k and f"blocks.{i}/att/heads.{split}" in k:
                            apply_activation_quant_override(k, x_encoding, encodings_all)

        encoding_path = f"{dirname}/quant_encodings_prefill.encodings" if parser_args.prefill_model else f"{dirname}/quant_encodings.encodings"
        with open(encoding_path, 'w') as f:
            json.dump(encodings_all, f, sort_keys=True, indent=4)

    del onnxmodel
    del graph

    print("Converting to QNN model...")
    states_layout = "NONTRIVIAL"
    converter_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qairt-converter -i {onnx_output_path}  " + " ".join([f'--source_model_input_layout "state{3*j+1}_in" "{states_layout}"' for j in range(layer_begin, layer_end)])
    converter_cmd += f" --float_bitwidth {parser_args.qnn_float_width} --float_bias_bitwidth {parser_args.qnn_float_width}"
    if parser_args.ext_embedding:
        converter_cmd += f' --source_model_input_layout "{input_name}" "NFC"'

    if has_deepemb:
        converter_cmd += " ".join([f' --source_model_input_layout "s_emb{j}_in" "NFC"' for j in range(layer_begin, layer_end)])

    if parser_args.quant_encodings:
        converter_cmd += f" --quantization_overrides {encoding_path}"

    if model_args.wkv_customop:
        converter_cmd += " --op_package_config hexagon/CPU/RwkvWkvOpPackage/config/RwkvWkvOpPackageCPU.xml"

    if os.name == 'nt':
        converter_cmd = "python " + converter_cmd
    print(converter_cmd)
    exit_code = os.system(converter_cmd)
    if exit_code != 0:
        print(f"Error: qairt-converter failed with exit code {exit_code}")
        exit(1)

    if parser_args.quant_encodings:
        print("Quantizing QNN dlc model...")
        quant_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qairt-quantizer -i {onnx_output_path.replace('.onnx', '.dlc')} -o {onnx_output_path.replace('.onnx', '.dlc')} --enable_float_fallback --act_bitwidth 16 --bias_bitwidth 8"
        if os.name == 'nt':
            quant_cmd = "python " + quant_cmd
        print(quant_cmd)
        exit_code = os.system(quant_cmd)
        if exit_code != 0:
            print(f"Error: qairt-quantizer failed with exit code {exit_code}")
            exit(1)

    if not parser_args.no_cleanup:
        # Delete all files in output_path except .dlc files
        for file in os.listdir(dirname):
            filepath = os.path.join(dirname, file)
            if not (file.endswith('.dlc') or file.endswith('.json') or file.endswith('.emb') or file.endswith('.encodings') or file.endswith('.deepemb') or file.endswith('.onnx')):
                if os.path.isfile(filepath):
                    os.remove(filepath)
