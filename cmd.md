## convert omniquant parameters
```
python convert_omniquant_parameters.py --omniquant_parameters omni_parameters_g1f_1b5.pth --model_path /models/rwkv7-g1f-1.5b-20260419-ctx8192.pth --output_file omniquant_encodings_g1f_1b5.json --num_head_splits 8
```

```
python convert_omniquant_parameters.py --omniquant_parameters omni_parameters_g1f_2b9.pth --model_path /models/rwkv7-g1f-2.9b-20260420-ctx8192.pth --output_file omniquant_encodings_g1f_2b9.json --num_head_splits 10
```

```
python convert_omniquant_parameters.py --omniquant_parameters omni_parameters_g1d_7b.pth --model_path /models/rwkv7-g1d-7.2b-20260131-ctx8192.pth --output_file omniquant_encodings_g1d_7b.json --num_head_splits 8
```

## compute quant encodings
### w8 0.1B and 0.4B
```
python compute_quant_encodings_experimental.py /models/rwkv7-g1d-0.1b-20260129-ctx8192.pth --output_folder quant_export/g1d-0b1 --binidx_dataset ./1 --calib_num_batches 2 --heads_per_split 3
```

```
python compute_quant_encodings_experimental.py /models/rwkv7-g1d-0.4b-20260210-ctx8192.pth --output_folder quant_export/g1d-0b4 --binidx_dataset ./1 --calib_num_batches 2 --heads_per_split 4
```

### w4 1.5B
```
python compute_quant_encodings_experimental.py /models/rwkv7-g1f-1.5b-20260419-ctx8192.pth --output_folder quant_export/g1f-1b5-w4/ --binidx_dataset ./1 --calib_num_batches 1 --heads_per_split 4 --load_encodings omniquant_encodings_g1f_1b5.json
```

### w8 1.5B
```
python compute_quant_encodings_experimental.py /models/rwkv7-g1f-1.5b-20260419-ctx8192.pth --output_folder quant_export/g1f-1b5-w8/ --binidx_dataset ./1 --calib_num_batches 1 --heads_per_split 4
```

### w4 2.9B and 7.2B
```
python compute_quant_encodings_experimental.py /models/rwkv7-g1f-2.9b-20260420-ctx8192.pth --output_folder quant_export/g1f-2b9-w4/ --binidx_dataset ./1 --calib_num_batches 1 --heads_per_split 4 --load_encodings omniquant_encodings_g1f_2b9.json
```

```
python compute_quant_encodings_experimental.py /models/rwkv7-g1d-7.2b-20260131-ctx8192.pth --output_folder quant_export/g1d-7b-w4/ --binidx_dataset ./1 --calib_num_batches 1 --heads_per_split 8 --load_encodings omniquant_encodings_g1d_7b.json
```