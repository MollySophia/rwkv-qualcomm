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

        encodings["param_encodings"][model_key] = []
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
