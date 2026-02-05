## convert omniquant parameters
```
python convert_omniquant_parameters.py --omniquant_parameters omni_parameters_g1c_1b5.pth --model_path /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --output_file omniquant_encodings_g1c_1b5.json --num_head_splits 8
```

```
python convert_omniquant_parameters.py --omniquant_parameters omni_parameters_g1d_2b9.pth --model_path /models/rwkv7-g1d-2.9b-20260131-ctx8192.pth --output_file omniquant_encodings_g1d_2b9.json --num_head_splits 10
```

## compute quant encodings

```
python compute_quant_encodings_experimental.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --output_folder quant_export/g1c-1b5-w4/ --binidx_dataset ./1 --calib_num_batches 1 --heads_per_split 4 --load_encodings omniquant_encodings_g1c_1b5.json
```

```
python compute_quant_encodings_experimental.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --output_folder quant_export/g1c-1b5-w8/ --binidx_dataset ./1 --calib_num_batches 1 --heads_per_split 4
```

```
python compute_quant_encodings_experimental.py /models/rwkv7-g1d-2.9b-20260131-ctx8192.pth --output_folder quant_export/g1d-2b9-w4/ --binidx_dataset ./1 --calib_num_batches 1 --heads_per_split 4 --load_encodings omniquant_encodings_g1d_2b9.json
```

## convert model to dlc

```
python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 1 --quant_encodings quant_export/g1c-1b5-w4/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 --ext_embedding
```


```
python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w4/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 \
&& python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w4/rwkv7-g1c-1.5b-20260110-ctx8192_prefill.encodings --wkv_customop --prefill_model --heads_per_split 4 \
&& python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w4/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 2 \
&& python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w4/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 4 \
&& python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w4/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 6 \
&& python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w4/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 8 \
&& python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w4/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 10 \
&& python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w4/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 12 \
&& python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w4/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 14
```

```
python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w8/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 \
&& python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w8/rwkv7-g1c-1.5b-20260110-ctx8192_prefill.encodings --wkv_customop --prefill_model --heads_per_split 4 \
&& python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w8/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 2 \
&& python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w8/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 4 \
&& python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w8/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 6 \
&& python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w8/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 8 \
&& python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w8/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 10 \
&& python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w8/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 12 \
&& python convert_model_dlc.py /models/rwkv7-g1c-1.5b-20260110-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1c-1b5-w8/rwkv7-g1c-1.5b-20260110-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 14
```


```
python convert_model_dlc.py /models/rwkv7-g1d-2.9b-20260131-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1d-2b9-w4/rwkv7-g1d-2.9b-20260131-ctx8192.encodings --wkv_customop --heads_per_split 4 \
&& python convert_model_dlc.py /models/rwkv7-g1d-2.9b-20260131-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1d-2b9-w4/rwkv7-g1d-2.9b-20260131-ctx8192_prefill.encodings --wkv_customop --prefill_model --heads_per_split 4 \
&& python convert_model_dlc.py /models/rwkv7-g1d-2.9b-20260131-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1d-2b9-w4/rwkv7-g1d-2.9b-20260131-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 2 \
&& python convert_model_dlc.py /models/rwkv7-g1d-2.9b-20260131-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1d-2b9-w4/rwkv7-g1d-2.9b-20260131-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 4 \
&& python convert_model_dlc.py /models/rwkv7-g1d-2.9b-20260131-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1d-2b9-w4/rwkv7-g1d-2.9b-20260131-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 6 \
&& python convert_model_dlc.py /models/rwkv7-g1d-2.9b-20260131-ctx8192.pth --chunks 4 --quant_encodings quant_export/g1d-2b9-w4/rwkv7-g1d-2.9b-20260131-ctx8192.encodings --wkv_customop --heads_per_split 4 --batch_size 8
```