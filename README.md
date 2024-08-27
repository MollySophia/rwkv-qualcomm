# Inference RWKV on Qualcomm HTP (Hexagon Tensor Processor) using QNN SDK
## Features
- Inference RWKV using QNN SDK, with Qualcomm CPU, GPU or HTP (Hexagon Tensor Processor) as the backend.
- Support for whole-model float16 inference (since Qualcomm HTP cannot do float32 math).
- Support for activation INT16 and weights INT8 quantized inference (with some key operations running with float16).

## Prerequisites
- Download and install the QNN SDK from the [Qualcomm Developer Network](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk).
- Setup the QNN SDK environment by following the instructions in Qualcomm's [documents](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/introduction.html).
- Setup the $QNN_SDK_ROOT environment variable to point to the QNN SDK installation directory. It should by default be installed at /opt/qcom/aistack/qnn/{version}.
- This project has been verified with:
    - QNN SDK 2.22.6
    - python==3.10 (as is recommended by QNN SDK documentation)
    - onnx==1.16.1
    - torch==2.3.1 (although QNN SDK is verified to work with torch==1.13.0, it's okay to use latest version of torch since we are only using torch for model conversion and onnx exporting)
    - Hardware: Qualcomm Snapdragon SM8650 with HTP v75 (Xiaomi Mi 14)

## Usage
### Converting a FP16 model
- `convert_model.py`: Modify the model path, split chunks and other parameters in the script, then run it to convert the model to QNN SDK format.
- Keep these parameters: ```
USE_SNPE_DLC = False
USE_QNN_QUANT = False
```

### Converting an A16W8 model
- `make_calibration_samples.py`: Modify the model path. This script will generate calibration samples for the model. Note: Keep the value of split chunks the same as in the `convert_model.py` script.
- `convert_model.py`: Modify the model path, split chunks and other parameters in the script, then run it to convert the model to QNN SDK format.
- Keep these parameters: ```
USE_SNPE_DLC = False
USE_QNN_QUANT = True
ACT_BITWIDTH = 16
WEIGHTS_BITWIDTH = 8
```

## Tested models
```Running on the Qualcomm Snapdragon SM8650 with HTP v75 (Xiaomi Mi 14)```
- RWKV-5-World-0.4B-v2-20231113-ctx4096, fp16: ```Average tokens per second: 50.7313```
- RWKV-5-ABC-82M-v1-20230901-ctx1024, fp16: ```Average tokens per second: 142.286```

## TODO
- [x] Add demo code for running inference on the device.
- [x] Add support for INT16/INT8 quantized inference.
- [ ] Package a library for easy use and integration.

## Questions
Q: How to solve the problem of outputing NaNs when inferencing RWKV's all operations with FP16?

A: The NaNs are from several operations:
- In wkv, the "k @ v" operation sometimes gets insanely large values, excedding the range of FP16, and becomes NaNs. This can be solved by applying a scale when calculating wkv. This doesn't affect the final result, since the wkv output value gets into the GroupNorm layer. Currently the scale is applied on ``k`` and ``v``. E.g: scale = 1/8, then ``k = k / 2``, ``v = v / 4``. By applying this, the output of ``k @ v`` won't be so large; the output of ``wkv`` is also scaled so that groupnorm has a smaller input too; the ``state`` for wkv is also scaled.
- LayerNorm layers: The hidden states between layers can get very large values, excedding the range of FP16, and becomes NaNs in following operations. This can be solved by rescaling the hidden states by half every 6 layers.
