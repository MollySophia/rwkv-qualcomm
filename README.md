# Inference RWKV on Qualcomm HTP (Hexagon Tensor Processor) using QNN SDK
## Features
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
    - QNN SDK 2.26.0
    - python==3.10 (as is recommended by QNN SDK documentation)
    - onnx==1.16.1
    - torch==2.2.2 (although QNN SDK is verified to work with torch==1.13.0, it's okay to use latest version of torch since we are only using torch for model conversion and onnx exporting) (2.2.2 is recommended by AIMET toolkit)
    - Hardware: Qualcomm Snapdragon SM8650 with HTP v75 (Xiaomi Mi 14)

## Usage
### 1. Convert model weights to QNN model library file.

#### Converting a FP16 model
- `convert_model.py`: usage: convert_model.py [-h] [--chunks CHUNKS] [--use_qnn_quant] [--act_bitwidth ACT_BITWIDTH] [--weights_bitwidth WEIGHTS_BITWIDTH] [--ext_embedding] model
- Convert the model: `python convert_model.py ../models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth --chunks 4`

#### Converting an A16W8 model
- `make_calibration_samples.py`: usage: make_calibration_samples.py [-h] [--ext_embedding] model output chunks
- Make calibration samples: `python make_calibration_samples.py ../models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth ./samples_1b6 2`
- Convert the model file: `python convert_model.py ../models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth --chunks 2 --use_qnn_quant --calib_data_path ./samples_1b6`
- The act_bitwidth and weights_bitwidth default to 16 and 8 respectively.
- Note: Please keep the `chunks` parameter the same in both scripts.

### Converting an A16W4 model
- `make_calibration_samples.py`: usage: make_calibration_samples.py [-h] [--ext_embedding] model output chunks
- Make calibration samples: `python make_calibration_samples.py ../models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth ./samples_1b6 2`
- Convert the model file: `--linear_param_encodings quant_encodings/RWKV-x060-World-1B6-v2.1-20240328-ctx4096_mse_rwkv_gptq_exceptions_asym_torch_w4.encodings` (The quantization encodings are either from the pre-calculated ones ([GDrive](https://drive.google.com/drive/folders/1IXp6FwdiZjV4fn8HXRUoGHM91WzvEwqj?usp=drive_link)), or generated using AIMET. Refer to: [AIMET_quant.md](docs/AIMET_quant.md))
- Some large Linear modules are quantized to 4-bit weights, while some are kept 8-bit for better accuracy.
- Note: Please keep the `chunks` parameter the same in both scripts.

The outputs will be in ``lib/`` directory. The model library contains weights, as well as the functions to prepare the graph. This can either be called on device using libraries in ``lib/aarch64-android/``, or be prepared on the x86 host machine using ``lib/x86_64-linux-clang/`` to generate an HTP context cache. Qualcomm HTP has a limitation on the size of the model library file, so the model will be split into multiple chunks.

### 2. Generate HTP context cache
- `make_context_cache_binary.py`: usage: make_context_cache_binary.py [-h] model_lib output_path {SM8650,SM8550,SC8380}
- Example:
```
$ python make_context_cache_binary.py ./lib/x86_64-linux-clang/libRWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk1of2.so output/ SM8650
```
- The script will automatically process each of the chunks together.
- The output would be in ``output/RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk1of2.bin`` and ``output/RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk2of2.bin``.

### 3. Run inference on the device
#### 3.1. Running on Qualcomm Snapdragon SM8650 with HTP v75 (Xiaomi Mi 14)
- Build the demo code: ``make -C librwkv-qualcomm``
- Push the binary and the HTP context cache to the device: ``adb push librwkv-qualcomm/obj/local/arm64-v8a/rwkv-qualcomm-demo /data/local/tmp/ && adb push output/RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk1of2.bin /data/local/tmp/ && adb push output/RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk2of2.bin /data/local/tmp/``
- Push the tokenizer model to the device: ``adb push assets/brwkv_vocab_v20230424.txt /data/local/tmp/``
- Push these QNN libs to the device `/data/local/tmp/` (Please change the HTP V75 version to the one you have):
```/opt/qcom/aistack/qairt/2.22.6.240515/lib/aarch64-android/libQnnHtp.so
/opt/qcom/aistack/qairt/2.22.6.240515/lib/aarch64-android/libQnnHtpNetRunExtensions.so
/opt/qcom/aistack/qairt/2.22.6.240515/lib/aarch64-android/libQnnHtpNetRunExtensions.so
/opt/qcom/aistack/qairt/2.22.6.240515/lib/aarch64-android/libQnnSystem.so
/opt/qcom/aistack/qairt/2.22.6.240515/lib/aarch64-android/libQnnHtpV75Stub.so
/opt/qcom/aistack/qairt/2.22.6.240515/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so
```
- *If using external embedding, please push `onnx/RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk1of2.emb` to `/data/local/tmp/rwkv/` too.*
- Finally run the demo code:
```
adb shell
$ cd /data/local/tmp
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp
$ # Specify the path to the first model chunk. The second chunk will be loaded automatically.
$ ./rwkv-qualcomm-demo brwkv_vocab_v20230424.txt RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk1of2.bin
```
#### 3.2. Running on Qualcomm Snapdragon X Elite laptops
- *TODO*

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
| RWKV v6 1.6B | att-a16w8 + ffn-a16w4 | 42.4368 | TODO |
| RWKV v6 1.6B | a16w8 | 31.6564| 4.75009,66.3497% |
| RWKV v6 1.6B | fp16 | 15.0434| 4.63598,67.2618% |
| RWKV v6 3B   | att-a16w8 + ffn-a16w4 | 21.3172 | TODO |
| RWKV v6 3B   | a16w8 | 16.2146 | TODO |

#### Obsolete data in previous versions for comparison:
| Model | Precision | Generation Tokens per second | LAMBADA ppl, acc |
| --- | --- | --- | --- |
| RWKV v6 1.6B | att-a16w8 + ffn-a16w4 | 32.6703| 4.65837,66.7378% |
| RWKV v6 1.6B | a16w8 | 26.0707| 4.59243,67.3006% |
| RWKV v6 1.6B | fp16 | 15.0434| 4.63598,67.2618% |
| RWKV v6 3B   | att-a16w8 + ffn-a16w4 | 17.3968 | 4.46606,68.8725% |
- RWKV-5-World-0.4B-v2-20231113-ctx4096, fp16: ```Average tokens per second: 50.7313```
- RWKV-5-ABC-82M-v1-20230901-ctx1024, fp16: ```Average tokens per second: 142.286```

## TODO
- [x] Add demo code for running inference on the device.
- [x] Add support for A16W8 quantized inference.
- [x] Add support for A16W4 quantized inference with AIMET quantization.
- [ ] Add document for running on Snapdragon X Elite laptops.
- [ ] Sequential prefilling on device.
- [ ] Package a library for easy use and integration.
