# Converting models using QNN's built-in quantization

#### Converting a FP16 model
- `convert_model.py`: usage: convert_model.py [-h] [--chunks CHUNKS] [--use_qnn_quant] [--act_bitwidth ACT_BITWIDTH] [--weights_bitwidth WEIGHTS_BITWIDTH] [--ext_embedding] model
- Convert the model: `python convert_model.py ../models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth --chunks 4`

#### Converting an A16W8 model
- `make_calibration_samples.py`: usage: make_calibration_samples.py [-h] [--ext_embedding] model output chunks
- Make calibration samples: `python make_calibration_samples.py ../models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth ./samples_1b6 2`
- Convert the model file: `python convert_model.py ../models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth --chunks 2 --use_qnn_quant --calib_data_path ./samples_1b6`
- The act_bitwidth and weights_bitwidth default to 16 and 8 respectively.
- Note: Please keep the `chunks` parameter the same in both scripts.

The outputs will be in ``lib/`` directory. The model library contains weights, as well as the functions to prepare the graph. This can either be called on device using libraries in ``lib/aarch64-android/``, or be prepared on the x86 host machine using ``lib/x86_64-linux-clang/`` to generate an HTP context cache. Qualcomm HTP has a limitation on the size of the model library file, so the model will be split into multiple chunks.