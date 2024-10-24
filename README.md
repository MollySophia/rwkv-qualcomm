# Inference RWKV on Qualcomm HTP (Hexagon Tensor Processor) using QNN SDK
## Features
- Inference RWKV using QNN SDK, with Qualcomm CPU, GPU or HTP (Hexagon Tensor Processor) as the backend.
- Support for whole-model float16 inference (since Qualcomm HTP cannot do float32 math).
- Support for activation INT16 and weights INT8 quantized inference (with some key operations running with float16).
- (Experimental) Support for activation INT16 and partial weights INT4 quantized inference. (Precision is yet to be improved)

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
Refer to: 
- [QNN-only method](./docs/Legacy_convert.md): This method is good enough for fp16 and a16w8 models.
- [AIMET method](./docs/Aimet_convert.md): This is for experimental a16w4 models. The precision is yet to be improved.

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
```
houji:/data/local/tmp/rwkv $ ./rwkv-qualcomm-demo b_rwkv_vocab_v20230424.txt RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk1of2.bin                                                                                                        
Loading model context binary from RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk1of2.bin
Reading chunk: RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk1of2.bin
Buffer size: 945721328
Reading chunk: RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk2of2.bin
Buffer size: 812641488
User: 请为我写一首诗。

Assistant: 好的，请问这首诗需要怎样的主题和格式？

User: 可以是一首关于爱情的诗，但不要太长。

Assistant: 在欢欣中你漫步于我边，心中不停地想着我，我也不停地向着你靠近。爱情是一种令人陶醉的感觉，它让人无法抗拒。相爱是生命中最美好的时刻，有时候我们需要用生命去表达我们的爱。

User: 还可以再写一首吗？

Assistant: 当然可以！您希望这首诗表达什么样的情感呢？

User: 请为我写一首自然风景描写的诗歌。

Assistant: 好的，请给我一些背景信息或者选定的景点名称。

Average time per token: 0.0328688s
Average tokens per second: 30.424
```

## Performance
```Running on the Qualcomm Snapdragon SM8650 with HTP v75 (Xiaomi Mi 14)```
| Model | Precision | Generation Tokens per second | LAMBADA ppl, acc |
| --- | --- | --- | --- |
| RWKV v6 1.6B | att-a16w8 + ffn-a16w4 | 41.1176 | TODO |
| RWKV v6 1.6B | a16w8 | 30.5982| 4.75009,66.3497% |
| RWKV v6 1.6B | fp16 | 15.0434| 4.63598,67.2618% |

#### Obsolete data:
| Model | Precision | Generation Tokens per second | LAMBADA ppl, acc |
| --- | --- | --- | --- |
| RWKV v6 1.6B | att-a16w8 + ffn-a16w4 | 32.6703| 4.65837,66.7378% |
| RWKV v6 1.6B | a16w8 | 26.0707| 4.59243,67.3006% |
| RWKV v6 1.6B | fp16 | 15.0434| 4.63598,67.2618% |
| RWKV v6 3B   | att-a16w8 + ffn-a16w4 | 17.3968 | 4.46606,68.8725% |
(Needs to be updated)
- RWKV-5-World-0.4B-v2-20231113-ctx4096, fp16: ```Average tokens per second: 50.7313```
- RWKV-5-ABC-82M-v1-20230901-ctx1024, fp16: ```Average tokens per second: 142.286```

## TODO
- [x] Add demo code for running inference on the device.
- [x] Add support for A16W8 quantized inference.
- [x] Add support for A16W4 quantized inference with AIMET quantization.
- [ ] Add document for running on Snapdragon X Elite laptops.
- [ ] Sequential prefilling on device.
- [ ] Package a library for easy use and integration.
