# Inference RWKV on Qualcomm HTP (Hexagon Tensor Processor) using QNN SDK
## Features
- Inference RWKV using QNN SDK, with Qualcomm CPU, GPU or HTP (Hexagon Tensor Processor) as the backend.
- Support for whole-model float16 inference (since Qualcomm HTP cannot do float32 math).

## Prerequisites
- Download and install the QNN SDK from the [Qualcomm Developer Network](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk).
- Setup the QNN SDK environment by following the instructions in Qualcomm's [documents](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/introduction.html).
- Setup the $QNN_SDK_ROOT environment variable to point to the QNN SDK installation directory. It should by default be installed at /opt/qcom/aistack/qnn/{version}.
- This project has been verified with:
    - QNN SDK 2.20.0
    - python==3.8 (as is recommended by QNN SDK documentation)
    - onnx==1.11.0
    - torch==2.2.0 (although QNN SDK is verified to work with torch==1.13.0, it's okay to use latest version of torch since we are only using torch for model conversion and onnx exporting)
    - Hardware: Qualcomm Snapdragon SM8650 with HTP v75 (Xiaomi Mi 14)

## Usage
- Convert the rwkv5 model using `rwkv_model.py`
- Convert the tokenizer using `convert_tokenizer.py`
- Build the demo code:
```
$ make -C chatrwkv-qualcomm
```

## Tested models
```Running on the Qualcomm Snapdragon SM8650 with HTP v75 (Xiaomi Mi 14)```
- RWKV-5-World-0.4B-v2-20231113-ctx4096, fp16: ```Average tokens per second: 50.7313```
- RWKV-5-ABC-82M-v1-20230901-ctx1024, fp16: ```Average tokens per second: 142.286```

## TODO
- [x] Add demo code for running inference on the device.
- [ ] Calibrate the GroupNorm scales with a more elegant method, like calculating KL-divergence.
- [ ] Add support for INT16/INT8 quantized inference.
- [ ] Package a library for easy use and integration.

## Questions
Q: How to solve the problem of outputing NaNs when inferencing RWKV's all operations with FP16?

A: The NaNs are from several operations:
- In wkv, the "k @ v" operation sometimes gets insanely large values, excedding the range of FP16, and becomes NaNs. This can be solved by applying a scale of 1/128 or 1/64 when calculating wkv. This doesn't affect the final result, since the wkv output value gets into the GroupNorm layer.
- In GroupNorm and LayerNorm layers, the "x - E(x)" values gets squared on the denominator, which can also exceed the range of FP16. For the GroupNorm in time_mixing, this can be solved by applying a layer-specific pre-calibrated scale to the input tensor. For the LayerNorm across the model, this can be solved by scaling the output by 2 every two layers. This limits the range of the output tensor to a reasonable range, and doesn't do much affect on the final result. For now, we are using a simple method to calibrate the scales like some quantization techniques, which is to run the model with FP32 and record the maximum and minimum values of the input tensor to the GroupNorm layer. This can be improved by using a more elegant method, like calculating KL-divergence.