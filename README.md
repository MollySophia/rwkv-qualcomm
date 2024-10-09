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
- [Old QNN-only method](./docs/Legacy_convert.md)
- [AIMET method](./docs/Aimet_convert.md)

### 2. Generate HTP context cache
- `make_context_cache_binary.py`: usage: make_context_cache_binary.py [-h] model_lib output_path {SM8650,SM8550,SC8380}
- Example:
```
$ python make_context_cache_binary.py ./lib/x86_64-linux-clang/libRWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk1of2.so output/ SM8650
```
- The script will automatically process each of the chunks together.
- The output would be in ``output/RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk1of2.bin`` and ``output/RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk2of2.bin``.

### 3. Run inference on the device
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
- Example output:
```
Loading model context binary from RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk1of2.bin
Reading chunk: RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk1of2.bin
Buffer size: 1043223288
Reading chunk: RWKV-x060-World-1B6-v2.1-20240328-ctx4096_chunk2of2.bin
Buffer size: 910193528

我们发现，这个函数的输入是一个字符串，输出是一个字符串，这是因为这个函数使用了一个递归的方法，将输入的字符串作为输入，并且使用了一个if语句来判断是否是回文字符串。
在这个递归函数中，我们使用了两个指针来指向字符串的开始和结束位置，然后将当前位置的字符和前一个指针所指向的字符相加，并将结果存储在一个变量中。如果当前位置的字符不是回文字符串，那么我们就需要将当前位置的字符转化为回文字符串，然后将结果加到指针所指向的字符串的结果中。如果当前位置的字符是回文字符串，那么将当前位置的字符和前一个指针所指向的字符相加，然后将结果存储在另一个指针所指向的字符串的结果中，循环直到指针所指向的字符串的结束位置为止。
我们可
Average time per token: 0.0457569s
Average tokens per second: 21.8546
```

## Tested models
```Running on the Qualcomm Snapdragon SM8650 with HTP v75 (Xiaomi Mi 14)```
- RWKV-RWKV-x060-World-1B6-v2.1-20240328-ctx4096, a16w8: ```Average tokens per second: 23.5765```
- RWKV-RWKV-x060-World-1B6-v2.1-20240328-ctx4096, fp16: ```Average tokens per second: 13.0575```
- RWKV-5-World-0.4B-v2-20231113-ctx4096, fp16: ```Average tokens per second: 50.7313```
- RWKV-5-ABC-82M-v1-20230901-ctx1024, fp16: ```Average tokens per second: 142.286```

## TODO
- [x] Add demo code for running inference on the device.
- [x] Add support for INT16/INT8 quantized inference.
- [ ] Add document for running on Snapdragon X Elite laptops
- [ ] Add support for AIMET and A16W4 quantization
- [ ] Package a library for easy use and integration.

## Questions
Q: How to solve the problem of outputing NaNs when inferencing RWKV's all operations with FP16?

A: The NaNs are from several operations:
- In wkv, the "k @ v" operation sometimes gets insanely large values, excedding the range of FP16, and becomes NaNs. This can be solved by applying a scale when calculating wkv. This doesn't affect the final result, since the wkv output value gets into the GroupNorm layer. Currently the scale is applied on ``k`` and ``v``. E.g: scale = 1/8, then ``k = k / 2``, ``v = v / 4``. By applying this, the output of ``k @ v`` won't be so large; the output of ``wkv`` is also scaled so that groupnorm has a smaller input too; the ``state`` for wkv is also scaled.
- LayerNorm layers: The hidden states between layers can get very large values, excedding the range of FP16, and becomes NaNs in following operations. This can be solved by rescaling the hidden states by half every 6 layers.
