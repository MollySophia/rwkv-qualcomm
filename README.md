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

## Usage
### 1. Convert model weights to QNN model library file.
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
``RWKV v6 1B6 A16W4``
```
130|houji:/data/local/tmp/rwkv $ ./rwkv-qualcomm-demo b_rwkv_vocab_v20230424.txt RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk1of2.bin
Loading model context binary from RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk1of2.bin
Reading chunk: RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk1of2.bin
Buffer size: 719802320
Reading chunk: RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk2of2.bin
Buffer size: 586727640
User: 请为我写一首诗。

Assistant: 当然，请告诉我你喜欢什么类型的诗歌。

User: 请写一首描写秋天景色的诗。

Assistant: 秋意渐浓，寒意渐深，
大地已是金黄如火，
落英纷飞，树影绰约，
人心也随之变得清静。
夜空中的繁星在闪闪，
思念似要被所有握住，
但又像是永不消散的孤注，
在这个秋天里如此特别。

请问这首诗符合您需求吗？

Average time per token: 0.0235644s
Average tokens per second: 42.4368
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
