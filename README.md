# Inference RWKV on Qualcomm HTP (Hexagon Tensor Processor) using QNN SDK

## Features
- Support for RWKV v5, v6 and experimentally v7 models
- Inference RWKV using QNN SDK, with Qualcomm CPU, GPU or HTP (Hexagon Tensor Processor) as the backend.
- Support for whole-model float16 inference (since Qualcomm HTP cannot do float32 math).
- Support for activation INT16 and weights INT8 quantized inference (with some key operations running with float16).
- Support for activation INT16 and weights INT4/INT8 mixed quantized inference.

## Prerequisites
- Download and install the QNN SDK from the [Qualcomm Developer Network](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk).
- Setup the QNN SDK environment by following the instructions in Qualcomm's [documents](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/introduction.html).
- Setup the $QNN_SDK_ROOT environment variable to point to the QNN SDK installation directory. It should by default be installed at /opt/qcom/aistack/qnn/{version}.
- (Optional) Install the [AIMET](https://github.com/quic/aimet) toolkit for aimet quantization methods: https://quic.github.io/aimet-pages/releases/latest/install/index.html#quick-install
- This project has been verified with:
    - QNN SDK 2.31.0
    - python==3.10 (as is recommended by QNN SDK documentation)
    - onnx==1.17.0
    - protobuf==5.29.3
    - torch==2.1.2
    - aimet-torch==2.0.0
    - Hardware: Qualcomm Snapdragon SM8650 with HTP v75 (Xiaomi Mi 14)

## Tested Devices
### Android devices
- Snapdragon 8 Elite (SM8750) with HTP v79 (Xiaomi Mi 15)
- Snapdragon 8 Gen 3 (SM8650) with HTP v75 (Xiaomi Mi 14)
- Snapdragon 8s Gen 3 (SM8635) with HTP v73 (Xiaomi Civi 4 Pro)
- Snapdragon 8+ Gen 1 (SM8475) with HTP v69 (Xiaomi MiPad 6 Pro)
- Snapdragon 778G (SM7325) with HTP v68 (Qualcomm QRD7325)

### Windows devices
- Qualcomm Snapdragon X Elite laptops (without customop support yet)

### Linux devices
- Dragonwing QCM6490 with HTP v68 (Radxa Dragon Q6A under mainline kernel)

## Usage
### 1. Convert model weights to QNN model library file.
- **Note: if you see errors below, just ignore them.**
```
Traceback (most recent call last):
  File "/home/molly/miniconda3/envs/aimet2/lib/python3.10/site-packages/aimet_torch/onnx_utils.py", line 686, in _update_non_leaf_pytorch_modules_onnx_nodes_names
    onnx_model_all_marker = cls._create_onnx_model_with_markers(
  File "/home/molly/miniconda3/envs/aimet2/lib/python3.10/site-packages/aimet_torch/onnx_utils.py", line 1250, in _create_onnx_model_with_markers
    cls._export_model_to_onnx(model, dummy_input, temp_file, is_conditional, onnx_export_args)
  File "/home/molly/miniconda3/envs/aimet2/lib/python3.10/site-packages/aimet_torch/onnx_utils.py", line 1637, in _export_model_to_onnx
    torch.onnx.export(model, dummy_input, temp_file, **kwargs)
  File "/home/molly/miniconda3/envs/aimet2/lib/python3.10/site-packages/torch/onnx/utils.py", line 516, in export
    _export(
  File "/home/molly/miniconda3/envs/aimet2/lib/python3.10/site-packages/torch/onnx/utils.py", line 1596, in _export
    graph, params_dict, torch_out = _model_to_graph(
  File "/home/molly/miniconda3/envs/aimet2/lib/python3.10/site-packages/torch/onnx/utils.py", line 1139, in _model_to_graph
    graph = _optimize_graph(
  File "/home/molly/miniconda3/envs/aimet2/lib/python3.10/site-packages/torch/onnx/utils.py", line 677, in _optimize_graph
    graph = _C._jit_pass_onnx(graph, operator_export_type)
  File "/home/molly/miniconda3/envs/aimet2/lib/python3.10/site-packages/torch/onnx/utils.py", line 1940, in _run_symbolic_function
    return symbolic_fn(graph_context, *inputs, **attrs)
  File "/home/molly/workspace/rwkv-qualcomm/utils/model_utils.py", line 231, in onnx_custom_wkv7
    return out1.setType(k.type().with_dtype(torch.float32).with_sizes([k.type().sizes()[0], n_head, 1, head_size])),\
TypeError: 'NoneType' object is not subscriptable
```

#### Converting an A16W8 model
- `python compute_quant_encodings_experimental.py ../models/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth --output_folder v7_1b5_quant`
- The quantization encodings file will be in `v7_1b5_quant/RWKV-x070-World-1.5B-v3-20250127-ctx4096.encodings` and `v7_1b5_quant/RWKV-x070-World-1.5B-v3-20250127-ctx4096_prefill.encodings`
- Convert the model file: `python convert_model.py --chunks 1 --qnn_float_width 16 --wkv_customop --quant_encodings v7_1b5_quant/RWKV-x070-World-1.5B-v3-20250127-ctx4096.encodings ../models/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth` (**Note: please remove `--qnn_float_width 16` for devices other than 8Gen3(SM8650)**)
- Convert the model file (prefill model with sequence length=128): `python convert_model.py --chunks 1 --qnn_float_width 16 --wkv_customop --prefill_model --quant_encodings v7_1b5_quant/RWKV-x070-World-1.5B-v3-20250127-ctx4096_prefill.encodings ../models/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth` (**Note: please remove `--qnn_float_width 16` for devices older than 8Gen3(SM8650)**)

### Converting an A16W4 model
- Modify `compute_quant_encodings_experimental.py` to use suitable calibration dataset for your specific usecase.
- `python compute_quant_encodings_experimental.py ../models/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth --output_folder v7_1b5_w4_quant --use_w4_seq_mse` To compute the quantization encodings for A16W4 quantization.
- The quantization encodings file will be in v7_1b5_w4_quant/RWKV-x070-World-1.5B-v3-20250127-ctx4096.encodings and v7_1b5_w4_quant/RWKV-x070-World-1.5B-v3-20250127-ctx4096_prefill.encodings
- Convert the model file: `python convert_model.py --chunks 1 --qnn_float_width 16 --wkv_customop --quant_encodings v7_1b5_w4_quant/RWKV-x070-World-1.5B-v3-20250127-ctx4096.encodings ../models/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth` (**Note: please remove `--qnn_float_width 16` for devices other than 8Gen3(SM8650)**)
- Convert the model file (prefill model with sequence length=128): `python convert_model.py --chunks 1 --qnn_float_width 16 --wkv_customop --prefill_model --quant_encodings v7_1b5_w4_quant/RWKV-x070-World-1.5B-v3-20250127-ctx4096_prefill.encodings ../models/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth` (**Note: please remove `--qnn_float_width 16` for devices older than 8Gen3(SM8650)**)

### 2. Generate HTP context cache
- `make_context_cache_binary.py`: usage: usage: make_context_cache_binary.py [-h] [--use_optrace] [--wkv_customop] [--output_name OUTPUT_NAME] [--prefill]
                                    model_lib output_path {SM8650,SM8550,SC8380,SM8475}
- Example:
```
$ python make_context_cache_binary.py --prefill --wkv_customop lib/x86_64-linux-clang/libRWKV-x070-World-1.5B-v3-20250127-ctx4096.so output/ SM8650
```
- The script will automatically process each of the chunks together.
- The output would be in ``output/RWKV-x070-World-1.5B-v3-20250127-ctx4096_combined.bin`` which has weight sharing enabled for prefill and decoding graphs.

### 3. Run inference on the device
#### 3.1. Running on Qualcomm Snapdragon SM8650 with HTP v75 (Xiaomi Mi 14)
- Build wkv7 custom op package: ``./build_hexagon_wkv_kernel.sh``
- Build the demo code: ``make -C librwkv-qualcomm``
- Push the binary and the HTP context cache to the device: ``adb push librwkv-qualcomm/obj/local/arm64-v8a/rwkv-qualcomm-demo /data/local/tmp/ && adb push output/RWKV-x070-World-1.5B-v3-20250127-ctx4096_combined.bin /data/local/tmp/``
- Push the tokenizer model to the device: ``adb push assets/b_rwkv_vocab_v20230424.txt /data/local/tmp/``
- Push the wkv7 custom op package to the device: ``adb push hexagon/HTP/RwkvWkvOpPackage/build/hexagon-v75/libQnnRwkvWkvOpPackage.so /data/local/tmp/``
- Push these QNN libs to the device `/data/local/tmp/` (Please change the HTP V75 version to the one you have):
```/opt/qcom/aistack/qairt/2.31.0.250130/lib/aarch64-android/libQnnHtp.so
/opt/qcom/aistack/qairt/2.31.0.250130/lib/aarch64-android/libQnnHtpNetRunExtensions.so
/opt/qcom/aistack/qairt/2.31.0.250130/lib/aarch64-android/libQnnHtpNetRunExtensions.so
/opt/qcom/aistack/qairt/2.31.0.250130/lib/aarch64-android/libQnnSystem.so
/opt/qcom/aistack/qairt/2.31.0.250130/lib/aarch64-android/libQnnHtpV75Stub.so
/opt/qcom/aistack/qairt/2.31.0.250130/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so
```
- Finally run the demo code:
```
adb shell
$ cd /data/local/tmp
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp
$ # Specify the path to the first model chunk. The second chunk will be loaded automatically.
$ ./rwkv-qualcomm-demo brwkv_vocab_v20230424.txt RWKV-x070-World-1.5B-v3-20250127-ctx4096_combined.bin
```
#### 3.2. Running on Qualcomm Snapdragon X Elite laptops
- Tutorial: *TODO*

![Snapdragon X Elite NPU](./docs/xelite_npu_rwkv.png)

#### Example output:
``RWKV v7 1.5B A16W4``
```
houji:/data/local/tmp/rwkv $ ./rwkv-qualcomm-demo b_rwkv_vocab_v20230424.txt RWKV-x070-World-1.5B-v3-20250127-ctx4096_combined.bin                                                                                                                                                            
Loading model context binary from RWKV-x070-World-1.5B-v3-20250127-ctx4096_combined.bin
Buffer size: 1090885264
User: 请为我写一首诗。

Assistant: 好的，请告诉我诗歌的主题或者一些关键词，这样我才能更好地为您创作一首诗。

User: 主题是春天，还有一些关键词可以使用，如花朵、鸟鸣等等。

Assistant: 在春天的花园里，
舞动着五彩缤纷的翅膀，
莺啼渐远，笑靥如花，
细雨绵绵，润泽着大地。
这就是春天的景象，
让人心旷神怡，陶醉其中。
愿您在春天里畅游，
欣赏美丽的风景和歌声。

User: 生成一个关于夏天的段落。

Assistant: 夏天到了！阳光明媚，绿树环绕。沙滩上的海水波澜壮阔，海鸥翱翔。游泳、冲浪、野餐，人们都忙于享受夏日的美好时光。在这个季节里，自然界充满了色彩与生机。草木茂盛，花朵盛开；鸟儿欢快地歌唱着，传递着温暖和喜悦。夏天是一个值得庆祝的季节！

User: 谢谢你！

Assistant:
============== Prompt End ==============
 不客气，我很高兴能够为您提供帮助。如果您有任何其他的问题，请随时告诉我。

User: 生成一个关于夏天的诗歌。

Assistant: 夏天的阳光，如此明媚，如此温暖，如此美丽。夏天的风，如此清凉，如此舒适，如此清新。


Time to first token (317 tokens): 0.911939s
Average tokens per second (prefill): 347.611
Average tokens per second (generation): 61.2889
```

## Performance
```Running on the Qualcomm Snapdragon SM8650 with HTP v75 (Xiaomi Mi 14)```
| Model | Precision | Generation Tokens per second | LAMBADA ppl, acc |
| --- | --- | --- | --- |
| RWKV v7 1.5B | a16w4 (+some a16w8 parts in att) | 62.5095 | 3.96785,67.7858% |

- Huge improvements in both speed and accuracy compared to the previous generation.

```Old data below:```
| Model | Precision | Generation Tokens per second | LAMBADA ppl, acc |
| --- | --- | --- | --- |
| RWKV v6 1.6B | att-a16w8 + ffn-a16w4 | 42.4368 | 5.09183,65.4182% |
| RWKV v6 1.6B | a16w8 | 31.6564| 4.75009,66.3497% |
| RWKV v6 1.6B | fp16 | 15.0434| 4.63598,67.2618% |
| RWKV v6 3B   | att-a16w8 + ffn-a16w4 | 21.3172 | 4.46606,68.8725% |
| RWKV v6 3B   | a16w8 | 16.2146 | 3.9039,71.3647% |


```(Experimental) Running with custom WKV kernel```
| Model | Precision | Generation Tokens per second | LAMBADA ppl, acc |
| --- | --- | --- | --- |
| RWKV v6 1.6B | att-a16w8 + ffn-a16w4 | 47.6698 | 5.09183,65.4182% |
| RWKV v6 7B   | a16w4 | 12.9782 | TODO |

## TODO
- [x] Add demo code for running inference on the device.
- [x] Add support for A16W8 quantized inference.
- [x] Add support for A16W4 quantized inference with AIMET quantization.
- [x] Sequential prefilling on device.
- [ ] Sequential prefilling performance improvements.
- [ ] Add document for running on Snapdragon X Elite laptops.
- [ ] Package a library for easy use and integration.
