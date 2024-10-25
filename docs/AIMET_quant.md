#### (Experimental) Make 4-bit quantization encodings for Linear modules using AIMET
- Compute 4 bit quantization encodings using AIMET: `python compute_linear_param_encodings.py ../models/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth --weights_bitwidth 4` (This may take some time and VRAM to finish)

The output encoding file will be in ``quant_export/`` directory. E.g. ``quant_export/RWKV-x060-World-3B-v2.1-20240417-ctx4096_mse_rwkv_gptq_exceptions_asym_torch_w4_processed.encodings``