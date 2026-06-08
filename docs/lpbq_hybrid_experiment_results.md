# 7.2B LPBQ / Per-Channel Hybrid Experiment Results

Date: 2026-05-22

Conversion/reproduction flow: `7b_lpbq_hybrid_conversion.md`.

## Setup

- Model: `/models/rwkv7-g1f-7.2b-20260414-ctx8192.pth`
- Target: SM8850 (`8elitegen5`), QNN custom op, 5 chunks, external embedding graph
- Quantization: A16W4 variants with `heads_per_split=8`
- Pack: `--spill_fill_buffer_size 320000000`
- Accuracy test: first 500 samples from `assets/lambada_test.txt`
- Speed test: `simple_benchmark`, 512-token random prefill + 128 decode, 3-run average
- Logs/results:
  - `profiling_logs/7b_hybrid_20260522_all_results.{json,csv}`
  - `profiling_logs/7b_hybrid_20260522_combined_results.{json,csv}`
  - `profiling_logs/7b_hybrid_20260522_rkv_output/results.{json,csv}`

## Weight Groups

The original 7.2B W4 per-channel set has 824 W4 tensors:

| group | tensor count |
|---|---:|
| `blocks.*.att.heads.*.receptance.weight` | 256 |
| `blocks.*.att.heads.*.key.weight` | 256 |
| `blocks.*.att.heads.*.value.weight` | 248 |
| `blocks.*.ffn.key.weight` | 32 |
| `blocks.*.ffn.value.weight` | 32 |

Hybrid variants keep the same activation encodings and non-selected parameter encodings, but replace selected per-channel W4 parameter encodings with LPBQ encodings.

## Results

| variant | LPBQ scope | acc | PPL | prefill tok/s | decode tok/s |
|---|---|---:|---:|---:|---:|
| `pc` | none, baseline per-channel W4 | 0.664 | 4.3330 | 125.14 | 11.74 |
| `lpbq` | all original 824 W4 tensors | 0.744 | 3.4096 | 101.39 | 10.09 |
| `mix-att-r` | attention receptance only | 0.698 | 3.6548 | 130.06 | 11.34 |
| `mix-att-k` | attention key only | 0.698 | 3.9816 | 115.78 | 13.86 |
| `mix-att-v` | attention value only | 0.674 | 4.1251 | 140.23 | 12.89 |
| `mix-ffn-key` | FFN key only | 0.660 | 4.4088 | 135.87 | 12.36 |
| `mix-ffn-value` | FFN value only | 0.646 | 4.5498 | 139.26 | 12.47 |
| `mix-att-rkv` | attention r/k/v | 0.736 | 3.3726 | 136.94 | 12.16 |
| `mix-ffn-kv` | FFN key/value | 0.666 | 4.5062 | 129.52 | 11.73 |
| `mix-att-rkv-output` | attention r/k/v + att.output | 0.740 | 3.4403 | 133.02 | 12.07 |
| `lpbq-extra-output` | full LPBQ + att.output | 0.750 | 3.4363 | 127.73 | 12.46 |
| `lpbq-extra-lora` | full LPBQ + LoRA low-rank matrices | 0.744 | 3.4108 | 115.57 | 11.07 |
| `lpbq-extra-head` | full LPBQ + head.weight | 0.746 | 3.4583 | 123.32 | 10.81 |

## Findings

- `mix-att-rkv` is the best balanced variant. It keeps FFN as per-channel, uses LPBQ only for attention r/k/v, has the best PPL in this run, and is much faster than full LPBQ.
- Most LPBQ accuracy gain comes from attention r/k/v. `mix-att-rkv` reaches 0.736 accuracy vs full LPBQ 0.744, while improving PPL from 3.4096 to 3.3726 and prefill from 101.39 to 136.94 tok/s.
- `att.receptance` and `att.key` are individually useful. `att.value` alone is much weaker.
- FFN LPBQ is not useful in this run. `mix-ffn-key`, `mix-ffn-value`, and `mix-ffn-kv` are all worse than or near the per-channel baseline.
- Adding `att.output` to `mix-att-rkv` is runnable but not clearly better. It improves exact accuracy from 0.736 to 0.740, but PPL worsens from 3.3726 to 3.4403 and speed drops slightly.
- `lpbq-extra-output` has the highest 500-sample exact accuracy, but it is a more aggressive variant because it includes full LPBQ plus `att.output` W4 LPBQ. Its PPL is worse than `mix-att-rkv`.
- `lpbq-extra-lora` is almost identical to full LPBQ in accuracy/PPL and does not justify the added W4 scope from this run.
- `lpbq-extra-head` runs after retry; the first failure was host disk exhaustion during ONNX external-data write, not a QNN conversion limitation.

## Current Recommendation

Use `mix-att-rkv` as the primary candidate:

- It isolates the useful LPBQ part to attention r/k/v.
- It avoids FFN LPBQ, which appears to hurt accuracy.
- It has the best PPL among tested variants.
- It is materially faster than full LPBQ and close to per-channel speed.
- It changes only the original W4 tensor set, instead of adding new W4 tensors such as `att.output` or `head.weight`.

Keep `lpbq-extra-output` and `mix-att-rkv-output` as secondary candidates if exact LAMBADA accuracy is prioritized over PPL and minimal W4 scope.

## Long-Context Decode-Only Check

Additional checks for `mix-att-rkv` and full `lpbq`, using RWKV7 G1G 7.2B ctx8192 on SM8850 (`8elitegen5`).

Loss is next-token cross entropy over tokenizer output directly. For token ids `t[0..8192]`, the test runs decode for `t[i]` and scores target `t[i+1]`, for 8192 targets per file. This does not include prefill loss.

| variant | model | input | tokens used | decode targets | avg loss | PPL | tok/s |
|---|---|---|---:|---:|---:|---:|---:|
| `mix-att-rkv` | G1G 7.2B ctx8192 | `eval_src_8192token.txt` | 8193 | 8192 | 1.1565 | 3.17878 | 10.3356 |
| `mix-att-rkv` | G1G 7.2B ctx8192 | `eval_src2_8192token.txt` | 8193 | 8192 | 1.73974 | 5.69586 | 8.42758 |
| `mix-att-rkv` | G1G 7.2B ctx8192 | weighted average | - | 16384 | 1.44812 | 4.2551 | - |
| `lpbq` | G1G 7.2B ctx8192 | `eval_src_8192token.txt` | 8193 | 8192 | 1.11209 | 3.04072 | 6.98066 |
| `lpbq` | G1G 7.2B ctx8192 | `eval_src2_8192token.txt` | 8193 | 8192 | 1.68381 | 5.38602 | 7.08501 |
| `lpbq` | G1G 7.2B ctx8192 | weighted average | - | 16384 | 1.39795 | 4.04689 | - |

### Full-Context Calibration Check

This run calibrates activation encodings on the first 8192 tokens from `eval_src_8192token.txt`, using streamed calibration chunks to avoid host OOM, then tests decode-only loss on `eval_src2_8192token.txt`.

The exported AIMET checkpoint was converted back into QNN-format encodings using the previous QNN encoding file as a template. All parameter encodings were replaced; most activation encodings were replaced, while unmapped activation names kept the previous template values.

| variant | calibration text | test input | tokens used | decode targets | avg loss | PPL | tok/s |
|---|---|---|---:|---:|---:|---:|---:|
| `mix-att-rkv` | `eval_src_8192token.txt` first 8192 tokens | `eval_src2_8192token.txt` | 8193 | 8192 | 1.74088 | 5.70238 | 6.54262 |

Compared with the earlier `mix-att-rkv` result on `eval_src2_8192token.txt` (`loss=1.73974`, `PPL=5.69586`), full-context calibration on `eval_src_8192token.txt` did not improve cross-text long-context decode loss in this run. The speed number is lower than prior runs and should be treated as a device-state measurement, not a confirmed regression, unless repeated under controlled thermals.

Conversion note: QNN context binary generation requires the tensors feeding `shift_gather1/Gather` and the corresponding `state*_out` tensors to have matching quantization parameters. For prefill graphs, `ln_1/ln_2` outputs, `shift_gather1/Gather_output_0`, and `concat_shift/Concat_output_0` were aligned to the matching state output encodings. Without this, chunk1 failed during graph compose with an offset mismatch such as `-34961` vs `-34207`.

## Reproduction Commands

Generate standard hybrid encodings from per-channel + full LPBQ:

```bash
python tools/make_7b_hybrid_encodings.py
```

Generate `mix-att-rkv-output` from per-channel + `lpbq-extra-output`:

```bash
python tools/make_7b_hybrid_encodings.py \
  --variants mix-att-rkv-output \
  --lpbq_dir quant_export/g1f-7b-lpbq-extra-output-w4
```

Generate full LPBQ plus extra `att.output` LPBQ encodings:

```bash
python compute_quant_encodings_experimental.py \
  /models/rwkv7-g1f-7.2b-20260414-ctx8192.pth \
  --output_folder quant_export/g1f-7b-lpbq-extra-output-w4 \
  --binidx_dataset ./1 \
  --calib_num_batches 1 \
  --heads_per_split 8 \
  --blockwise_quant \
  --reuse_encodings_folder quant_export/g1f-7b-lpbq-w4 \
  --blockwise_extra_modules att_output \
  --allow_extra_lpbq
```

Run selected QNN experiments:

```bash
RWKV_ADB_SERIAL=3e56125d \
python tools/run_7b_hybrid_qnn_experiments.py \
  --variants mix-att-rkv,mix-att-rkv-output,lpbq-extra-output
```

## Code Notes

- `compute_quant_encodings_experimental.py`
  - Added `--blockwise_extra_modules att_output,lora,head`.
  - Added `--allow_extra_lpbq` for exporting LPBQ tensors beyond the reused baseline W4 set.
  - Default behavior is unchanged when these flags are not used.
- `tools/make_7b_hybrid_encodings.py`
  - Builds hybrid encodings by replacing selected per-channel W4 parameter entries with LPBQ entries.
  - Default variants exclude `mix-att-rkv-output` to avoid accidentally generating it from a base LPBQ file without `att.output` LPBQ entries.
  - Validates that requested LPBQ groups exist in the source LPBQ encodings.
- `tools/run_7b_hybrid_qnn_experiments.py`
  - Converts, context-binary-generates, packs, pushes, benchmarks, and runs 500-sample LAMBADA for 7.2B variants.
  - Defaults to the base comparison set; extra variants such as `mix-att-rkv-output` must be selected explicitly after their encodings are generated.
  - It is intentionally fixed to the 7.2B/SM8850 setup used here.

## Caveats

- The accuracy result is only the first 500 LAMBADA samples. Use full LAMBADA before finalizing a production quantization choice.
- Benchmark speeds have normal device thermal/DVFS variability.
- `simple_benchmark` uses random tokens and is a coarse speed check, not a full application benchmark.
