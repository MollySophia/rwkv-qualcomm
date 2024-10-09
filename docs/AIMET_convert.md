#### Converting an A16W8 model
- `make_calibration_samples.py`: usage: make_calibration_samples.py [-h] [--ext_embedding] model output chunks
- Make calibration samples: `python make_calibration_samples.py ../models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth ./samples_1b6 2`
- Convert the model file: `python convert_model.py ../models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth --num_chunks 2 --weights_bitwidth 8 --calib_data_path ./samples_1b6`
- The act_bitwidth and weights_bitwidth default to 16 and 8 respectively.
- Note: Please keep the `chunks` parameter the same in both scripts.

#### (Experimental) Converting an A16W4 model
- `make_calibration_samples.py`: usage: make_calibration_samples.py [-h] [--ext_embedding] model output chunks
- Make calibration samples: `python make_calibration_samples.py ../models/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth ./samples_3b 4`
- Compute 4 bit quantization encodings using AIMET: `python compute_linear_param_encodings.py ../models/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth --weights_bitwidth 4` (This may take some time and VRAM to finish)
- Convert the model file: `python export_quantized_model.py ../models/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth --linear_param_encodings quant_export/RWKV-x060-World-3B-v2.1-20240417-ctx4096_mse_rwkv_gptq_exceptions_symqt_torch_w4.encodings --calib_data_path ./samples_3b/ --num_chunks 4`

The outputs will be in ``lib/`` directory. The model library contains weights, as well as the functions to prepare the graph. This can either be called on device using libraries in ``lib/aarch64-android/``, or be prepared on the x86 host machine using ``lib/x86_64-linux-clang/`` to generate an HTP context cache. Qualcomm HTP has a limitation on the size of the model library file, so the model will be split into multiple chunks.