from rwkv_src.rwkv_model import RWKV_RNN, make_chunks
from rwkv_src.rwkv_v7_modules_conv import Wkv7, L2Norm
from utils.model_utils import register_customop_symbols, get_dummy_state_kvcache
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
from aimet_torch.onnx_utils import OnnxSaver

register_customop_symbols()

parser = argparse.ArgumentParser(description='Convert model')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser.add_argument('--chunks', type=int, default=2, help='Number of chunks')
parser.add_argument('--qnn_float_width', type=int, default=32, help='QNN float width')
parser.add_argument('--ext_embedding', action='store_true', default=False, help='Use external embedding')
parser.add_argument('--ext_lmhead', action='store_true', default=False, help='Use external head')
parser.add_argument('--quant_encodings', type=Path, help='Path to quant encodings')
parser.add_argument('--prefill_model', action='store_true', help='Convert model for sequential prefill')
parser.add_argument('--prefill_seq_length', type=int, default=16, help='Prefill sequence length')
parser.add_argument('--wkv_customop', action='store_true', help='Use custom op for wkv')
parser_args = parser.parse_args()

seq_length = parser_args.prefill_seq_length if parser_args.prefill_model else 1

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False
model_args.fp16 = False
model_args.bf16 = False
model_args.wkv_customop = parser_args.wkv_customop
model_args.USE_EMBEDDING = False if parser_args.ext_embedding else True
model_args.MODEL_NAME = str(parser_args.model)
model_args.output_last = True
model_args.EXTERNAL_HEAD = True if parser_args.ext_lmhead else False

if 'ABC' in model_args.MODEL_NAME or 'MIDI' in model_args.MODEL_NAME or parser_args.quant_encodings or 'x070' in model_args.MODEL_NAME:
    model_args.RESCALE_LAYER = 0
else:
    model_args.RESCALE_LAYER = 6

model = make_chunks(parser_args.chunks, model_args) if parser_args.chunks > 1 else RWKV_RNN(model_args)

qnn_sdk_root = os.environ["QNN_SDK_ROOT"]
if not qnn_sdk_root:
    print("Please set QNN_SDK_ROOT environment variable to the root of the Qualcomm Neural Processing SDK")
    exit(1)
os.path.exists("onnx") or os.mkdir("onnx")
if os.name == 'nt':
    qnn_tools_target = 'x86_64-windows-msvc'
else:
    qnn_tools_target = 'x86_64-linux-clang'

if type(model) == list:
    args = model[0].args
    filename = args.MODEL_NAME.split("/")[-1]
    input_dtype = torch.float16 if args.fp16 else torch.float32

    if args.EXTERNAL_HEAD:
        model[-1].head.weight.detach().squeeze().cpu().numpy().astype(np.float16).tofile("onnx/" + filename + f"_chunk1of{len(model)}.fp16.head.weight")
        lm_head = torch.nn.Linear(args.n_embd, model[-1].head.weight.squeeze().shape[0], bias=False)
        lm_head.weight = torch.nn.Parameter(model[-1].head.weight.squeeze())
        torch.onnx.export(lm_head, (torch.zeros(1, args.n_embd, dtype=input_dtype),), "onnx/" + filename + f"_lmhead.onnx", opset_version=17, input_names=['in'], output_names=['out'], dynamic_axes={'in': {0: 'batch_size'}, 'out': {0: 'batch_size'}})
        print(f"lmhead onnx model saved to onnx/{filename}_lmhead.onnx")

    states = get_dummy_state_kvcache(1, args, model[0].device)
    if parser_args.quant_encodings:
        with open(parser_args.quant_encodings, 'r') as f:
            encodings_all = json.load(f)

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
        dirname = "onnx/" + filename + f"_chunk{i+1}of{len(model)}"
        os.path.exists(dirname) or os.mkdir(dirname)
        if i == 0 and args.USE_EMBEDDING:
            in0 = torch.LongTensor([[1]*seq_length], device=model[i].device)
        else:
            in0 = torch.zeros(1, seq_length, args.n_embd, dtype=input_dtype)

        inputs = [in0, [states[j] for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]]
        # input_name = ('in' if not parser_args.prefill_model else 'in_prefill') + f'_chunk{i+1}'
        input_name = 'in'
        if not args.USE_EMBEDDING and i == 0:
            input_name += '_embedding'
        if parser_args.prefill_model:
            input_name += '_prefill'
        input_name += f'_chunk{i+1}'

        output_name = ('out' if not parser_args.prefill_model else 'out_prefill') + f'_chunk{i+1}'
        input_names = [input_name] + [f'state{j}_in' for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]
        output_names = [output_name] + [f'state{j}_out' for j in range(3*model[i].layer_begin, 3*model[i].layer_end)]

        if i == 0:
            encodings_chunk = copy.deepcopy(encodings_all)
            for v in encodings_all["activation_encodings"]:
                if f'/blocks.{model[1].layer_begin-1}/ffn/add_feed_forward/Add_output_0' in v['name']:
                    tmp = copy.deepcopy(v)
                    tmp['name'] = output_name
                    encodings_chunk["activation_encodings"].append(tmp)
            if not args.USE_EMBEDDING:
                for v in encodings_all["param_encodings"]:
                    if f'embedding.weight' in v['name']:
                        tmp = copy.deepcopy(v)
                        tmp['name'] = input_name
                        encodings_chunk["activation_encodings"].append(tmp)
                        break
            with open(f"{dirname}/quant_encodings_chunk{i}.encodings", "w") as f:
                json.dump(encodings_chunk, f, sort_keys=True, indent=4)
        else:
            encodings_chunk = {"activation_encodings": [], "excluded_layers": [], "param_encodings": [], "quantizer_args": encodings_all["quantizer_args"], "version": encodings_all["version"]}
            for v in encodings_all["activation_encodings"]:
                try:
                    encoding_block_id = int(v['name'].split(".")[-1].split("/")[0]) if 'block' in v['name'] else args.n_layer-1
                except ValueError:
                    encoding_block_id = int(v['name'].split(".")[1]) if 'block' in v['name'] else args.n_layer-1
                if any([
                    'state' in v['name'],
                    (encoding_block_id >= model[i].layer_begin and encoding_block_id < model[i].layer_end),
                    v['name'] == '/blocks/blocks.0/att/post_permute_v/Transpose_output_0',
                    v['name'] == f'/blocks/blocks.{model[i].layer_begin-1}/ffn/add_feed_forward/Add_output_0'
                ]):
                    if '/blocks.0/att/post_permute_v/Transpose_output_0' in v['name']:
                        tmp = copy.deepcopy(v)
                        tmp['name'] = f'v_first_in{"_prefill" if parser_args.prefill_model else ""}_chunk{i+1}'
                        encodings_chunk["activation_encodings"].append(tmp)
                    elif f'/blocks.{model[i].layer_begin-1}/ffn/add_feed_forward/Add_output_0' in v['name']:
                        tmp = copy.deepcopy(v)
                        tmp['name'] = input_name
                        encodings_chunk["activation_encodings"].append(tmp)
                    else:
                        tmp = copy.deepcopy(v)
                        tmp['name'] = tmp['name'].replace(f"blocks.{encoding_block_id}", f"blocks.{encoding_block_id-model[i].layer_begin}")
                        encodings_chunk["activation_encodings"].append(tmp)
                if args.EXTERNAL_HEAD and i == len(model) - 1:
                    for v in encodings_all["activation_encodings"]:
                        if 'ln_out/LayerNormalization_output_0' in v['name']:
                            tmp = copy.deepcopy(v)
                            tmp['name'] = output_name
                            encodings_chunk["activation_encodings"].append(tmp)
                            break
            for v in encodings_all["param_encodings"]:
                if 'embedding' in v['name']:
                    encoding_block_id = 0
                else:
                    encoding_block_id = int(v['name'].split(".")[1]) if 'block' in v['name'] else args.n_layer-1
                if 'state' in v['name'] or (encoding_block_id >= model[i].layer_begin and encoding_block_id < model[i].layer_end):
                    # encodings_chunk["param_encodings"][k.replace(f"blocks.{encoding_block_id}", f"blocks.{encoding_block_id-model[i].layer_begin}")] = v
                    tmp = copy.deepcopy(v)
                    tmp['name'] = tmp['name'].replace(f"blocks.{encoding_block_id}", f"blocks.{encoding_block_id-model[i].layer_begin}")
                    encodings_chunk["param_encodings"].append(tmp)
            with open(f"{dirname}/quant_encodings_chunk{i}.encodings", 'w') as f:
                json.dump(encodings_chunk, f, sort_keys=True, indent=4)

        if args.version == 7:
            if i == 0:
                output_names += [('v_first_out' if not parser_args.prefill_model else 'v_first_out_prefill') + f'_chunk{i+1}']
            else:
                input_names += [('v_first_in' if not parser_args.prefill_model else 'v_first_in_prefill') + f'_chunk{i+1}']
                inputs += [torch.zeros(seq_length, args.n_embd, dtype=input_dtype)]

        onnx_output_path = f"{dirname}/{filename}"
        if parser_args.ext_embedding:
            onnx_output_path += "_embedding"
        if parser_args.prefill_model:
            onnx_output_path += "_prefill"
        onnx_output_path += f"_chunk{i+1}of{len(model)}.onnx"
        OnnxSaver.create_onnx_model_with_pytorch_layer_names(onnx_output_path, model[i], tuple(inputs),
            False, None, {'input_names': input_names, 'output_names': output_names, 'opset_version': 17})
        onnxmodel = onnx.load(onnx_output_path, load_external_data=True)
        graph = gs.import_onnx(onnxmodel)
        # set output shape for wkv7_output
        for k, v in graph.tensors().items():
            if "wkv7_output_x_output_0" in k:
                graph.tensors()[k].to_variable(dtype=np.float32, shape=[seq_length, args.n_head, 1, args.head_size])
            elif "wkv7_state_output_0" in k:
                graph.tensors()[k].to_variable(dtype=np.float32, shape=[seq_length, args.n_head, args.head_size, args.head_size])
            elif "wkv7_output_0" in k:
                graph.tensors()[k].to_variable(dtype=np.float32, shape=[args.n_head, args.head_size + seq_length, args.head_size])

        onnxmodel = gs.export_onnx(graph)
        convert_model_to_external_data(onnxmodel)
        onnx.save(onnxmodel, onnx_output_path)
        shape_inference.infer_shapes_path(onnx_output_path)
        print(f"onnx model chunk{i} saved to {onnx_output_path}")

    print("Converting and compiling QNN models...")
    for i in range(len(model)):
        dirname = "onnx/" + filename + f"_chunk{i+1}of{len(model)}"
        onnx_output_path = f"{dirname}/{filename}"
        if parser_args.ext_embedding:
            onnx_output_path += "_embedding"
        if parser_args.prefill_model:
            onnx_output_path += "_prefill"
        onnx_output_path += f"_chunk{i+1}of{len(model)}.onnx"
        os.path.exists(dirname) or os.mkdir(dirname)

        states_layout = "NONTRIVIAL"
        converter_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qairt-converter --input_network {onnx_output_path} "
        converter_cmd += " ".join([f'--source_model_input_layout "state{3*j+1}_in" "{states_layout}"' for j in range(model[i].layer_begin, model[i].layer_end)])
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
        os.system(converter_cmd)

        print("Quantizing QNN dlc model...")
        quant_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qairt-quantizer -i {onnx_output_path.replace('.onnx', '.dlc')} -o {onnx_output_path.replace('.onnx', '.dlc')} --enable_float_fallback  --act_bitwidth 16 --bias_bitwidth 8"
        if os.name == 'nt':
            quant_cmd = "python " + quant_cmd
        print(quant_cmd)

        os.system(quant_cmd)
        for file in os.listdir(dirname):
            filepath = os.path.join(dirname, file)
            if not (file.endswith('.dlc') or file.endswith('.json') or file.endswith('.encodings') or file.endswith('.onnx')):
                if os.path.isfile(filepath):
                    os.remove(filepath)
else:
    args = model.args
    filename = args.MODEL_NAME.split("/")[-1]
    dirname = "onnx/" + filename
    os.path.exists(dirname) or os.mkdir(dirname)
    if not args.USE_EMBEDDING:
        if not parser_args.quant_encodings:
            model.emb_weight.cpu().numpy().astype(np.float16 if args.fp16 else np.float32).tofile(dirname + '/' + filename + (".fp16" if args.fp16 else ".fp32") + ".emb")
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
    input_dtype = torch.float16 if args.fp16 else torch.float32

    in0 = torch.LongTensor([[1]*seq_length]) if args.USE_EMBEDDING else torch.zeros(1, seq_length, args.n_embd, dtype=input_dtype)
    states = get_dummy_state_kvcache(1, args, model.device)

    input_name = 'in'
    if parser_args.ext_embedding:
        input_name += '_embedding'
    if parser_args.prefill_model:
        input_name += '_prefill'
    output_name  = 'out'
    if parser_args.prefill_model:
        output_name += '_prefill'
    input_names = [input_name] + [f'state{i}_in' for i in range(3*args.n_layer)]
    output_names = [output_name] + [f'state{i}_out' for i in range(3*args.n_layer)]

    onnx_output_path = f"{dirname}/{filename}"
    if parser_args.ext_embedding:
        onnx_output_path += "_ext_embedding"
    if parser_args.prefill_model:
        onnx_output_path += "_prefill"
 
    onnx_output_path += ".onnx"
    OnnxSaver.create_onnx_model_with_pytorch_layer_names(onnx_output_path, model, (in0, states),
        False, None, {'input_names': input_names, 'output_names': output_names, 'opset_version': 17})
    layer_begin = model.layer_begin
    layer_end = model.layer_end
    del model
    onnxmodel = onnx.load(onnx_output_path, load_external_data=True)
    graph = gs.import_onnx(onnxmodel)
    # set output shape for wkv7_output
    for k, v in graph.tensors().items():
        if "wkv7_output_x_output_0" in k:
            graph.tensors()[k].to_variable(dtype=np.float32, shape=[seq_length, args.n_head, 1, args.head_size])
        elif "wkv7_state_output_0" in k:
            graph.tensors()[k].to_variable(dtype=np.float32, shape=[seq_length, args.n_head, args.head_size, args.head_size])
        elif "wkv7_output_0" in k:
            graph.tensors()[k].to_variable(dtype=np.float32, shape=[args.n_head, args.head_size + seq_length, args.head_size])

    onnxmodel = gs.export_onnx(graph)
    convert_model_to_external_data(onnxmodel)
    onnx.save(onnxmodel, onnx_output_path)
    shape_inference.infer_shapes_path(onnx_output_path)
    print(f"onnx model saved to {onnx_output_path}")
    del onnxmodel
    del graph

    encoding_path = str(parser_args.quant_encodings) if parser_args.quant_encodings else None
    if encoding_path and not args.USE_EMBEDDING:
        with open(encoding_path, 'r') as f:
            encodings_all = json.load(f)
        for v in encodings_all["param_encodings"]:
            if f'embedding.weight' in v['name']:
                tmp = copy.deepcopy(v)
                tmp['name'] = input_name
                encodings_all["activation_encodings"].append(tmp)
                break
        encoding_path = f"{dirname}/quant_encodings_prefill.encodings" if parser_args.prefill_model else f"{dirname}/quant_encodings.encodings"
        with open(encoding_path, 'w') as f:
            json.dump(encodings_all, f, sort_keys=True, indent=4)

    print("Converting to QNN model...")
    states_layout = "NONTRIVIAL"
    converter_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qairt-converter -i {onnx_output_path}  " + " ".join([f'--source_model_input_layout "state{3*j+1}_in" "{states_layout}"' for j in range(layer_begin, layer_end)])
    if parser_args.ext_embedding:
        converter_cmd += f' --source_model_input_layout "{input_name}" "NFC"'

    if parser_args.quant_encodings:
        converter_cmd += f" --quantization_overrides {encoding_path}"

    if model_args.wkv_customop:
        converter_cmd += " --op_package_config hexagon/CPU/RwkvWkvOpPackage/config/RwkvWkvOpPackageCPU.xml"

    if os.name == 'nt':
        converter_cmd = "python " + converter_cmd
    print(converter_cmd)
    os.system(converter_cmd)
    print("Quantizing QNN dlc model...")
    quant_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qairt-quantizer -i {onnx_output_path.replace('.onnx', '.dlc')} -o {onnx_output_path.replace('.onnx', '.dlc')} --enable_float_fallback --act_bitwidth 16 --bias_bitwidth 8"
    if os.name == 'nt':
        quant_cmd = "python " + quant_cmd
    print(quant_cmd)
    os.system(quant_cmd)

    # Delete all files in output_path except .dlc files
    for file in os.listdir(dirname):
        filepath = os.path.join(dirname, file)
        if not (file.endswith('.dlc') or file.endswith('.json') or file.endswith('.emb') or file.endswith('.encodings')):
            if os.path.isfile(filepath):
                os.remove(filepath)
