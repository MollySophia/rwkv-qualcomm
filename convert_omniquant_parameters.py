import torch
import json
import argparse
import math

sigmoid = torch.nn.Sigmoid()
BITWIDTH = 4
num_steps = 2 ** BITWIDTH - 1
qmin = 0
qmax = num_steps
symmetric_offset = -round((qmin + qmax) / 2)

parser = argparse.ArgumentParser()
parser.add_argument('--omniquant_parameters', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--num_head_splits', type=int, default=4, help='Number of head splits')
args = parser.parse_args()

omniquant_parameters = torch.load(args.omniquant_parameters, map_location='cpu')
model = torch.load(args.model_path, map_location='cpu')

encodings = {"activation_encodings": {}, "param_encodings": {}}

for i in omniquant_parameters.keys():
    for key, value in omniquant_parameters[i].items():
        if 'lowbound' in key:
            continue
        model_key = key.replace('attn', 'att').replace('r_proj', 'receptance')
        model_key = model_key.replace('v_proj', 'value').replace('k_proj', 'key')
        model_key = model_key.replace('o_proj', 'output')
        model_key = model_key.replace('_lora.lora.0', '1')
        model_key = model_key.replace('weight_quantizer', 'weight')
        model_key = f'blocks.{i}.{model_key.replace(".upbound_factor", "")}'
        if "lora" in key:
            model[model_key] = model[model_key.replace(".weight", "")].t()
        if "lm_head" in model_key:
            model_key = "head.weight"

        xmax = model[model_key].max(dim=1, keepdim=True)[0].float()
        xmin = model[model_key].min(dim=1, keepdim=True)[0].float()

        xmax = sigmoid(value.float()) * xmax
        xmin = sigmoid(omniquant_parameters[i][f'{key.replace(".upbound_factor", "")}.lowbound_factor'].float()) * xmin

        # make floating point zero exactly representable
        abs_max = torch.max(xmax.abs(), xmin.abs())
        step_size = (abs_max * 2) / num_steps
        xmax = abs_max - (step_size / 2)
        xmin = -abs_max - (step_size / 2)

        scale = (xmax - xmin) / (qmax - qmin)
        offset = symmetric_offset * torch.ones_like(xmax)
        if "lora" in key:
            model_key = model_key.replace("a1", "matmul_a1").replace("g1", "matmul_g1")
            model_key = model_key.replace("w1", "matmul_time_decay_w1").replace("v1", "matmul_v1")

        if 'blocks.0.att.value.weight' in model_key:
            continue

        # print(model_key, xmax.shape)
        if any([n in model_key for n in ['att.key.weight', 'att.receptance.weight', 'att.value.weight']]):
            for split in range(args.num_head_splits):
                xmax_split = xmax.reshape(args.num_head_splits, -1, 1)[split]
                xmin_split = xmin.reshape(args.num_head_splits, -1, 1)[split]
                scale_split = scale.reshape(args.num_head_splits, -1, 1)[split]
                offset_split = offset.reshape(args.num_head_splits, -1, 1)[split]
                model_key_split = model_key.replace('att.', f'att.heads.{split}.')
                encodings["param_encodings"][model_key_split] = []
                print(model_key_split, xmax_split.shape)
                for c in range(xmax_split.shape[0]):
                    encodings["param_encodings"][model_key_split].append({
                        "bitwidth": BITWIDTH,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": xmax_split[c].item(),
                        "min": xmin_split[c].item(),
                        "offset": offset_split[c].item(),
                        "scale": scale_split[c].item(),
                    })
        else:
            encodings["param_encodings"][model_key] = []
            print(model_key, xmax.shape)
            for c in range(xmax.shape[0]):
                encodings["param_encodings"][model_key].append({
                    "bitwidth": BITWIDTH,
                    "dtype": "int",
                    "is_symmetric": "True",
                    "max": xmax[c].item(),
                    "min": xmin[c].item(),
                    "offset": offset[c].item(),
                    "scale": scale[c].item(),
                })

with open(args.output_file, 'w') as f:
    json.dump(encodings, f, indent=4)
