# 7.2B LPBQ Hybrid Quantization And Conversion

This is the reproducible flow for RWKV7 G1F 7.2B A16W4 per-channel/LPBQ
hybrid models. Current primary candidate is `mix-att-rkv`.

## Fixed Setup

- Model: `/models/rwkv7-g1f-7.2b-20260414-ctx8192.pth`
- Base model name: `rwkv7-g1f-7.2b-20260414-ctx8192`
- Target SoC: `8elitegen5` / `SM8850`
- Chunks: `5`
- `heads_per_split`: `8`
- Hidden size: `4096`
- Vocab size: `65536`
- Quant type: `a16w4`
- Embedding: external embedding graph only
- Pack spill/fill buffer: `--spill_fill_buffer_size 320000000`

For 7B conversion, keep host ONNX conversion serial. `parallel_convert_pipeline.py`
already forces `NUM_PARALLEL_CONVERT=1` for selected models with size `>= 7.0`.

## 1. Convert OmniQuant Parameters

Convert OmniQuant parameters into AIMET-compatible encodings. The split count
must match the model graph split used later.

```bash
python convert_omniquant_parameters.py \
  --omniquant_parameters omni_parameters_g1f_7b2.pth \
  --model_path /models/rwkv7-g1f-7.2b-20260414-ctx8192.pth \
  --output_file omniquant_encodings_g1f_7b.json \
  --num_head_splits 8
```

## 2. Build Per-Channel Baseline Encodings

This creates the baseline W4 per-channel decode and prefill encoding files:

- `quant_export/g1f-7b-w4/rwkv7-g1f-7.2b-20260414-ctx8192.encodings`
- `quant_export/g1f-7b-w4/rwkv7-g1f-7.2b-20260414-ctx8192_prefill.encodings`

```bash
python compute_quant_encodings_experimental.py \
  /models/rwkv7-g1f-7.2b-20260414-ctx8192.pth \
  --output_folder quant_export/g1f-7b-w4/ \
  --binidx_dataset ./1 \
  --calib_num_batches 1 \
  --heads_per_split 8 \
  --load_encodings omniquant_encodings_g1f_7b.json
```

Use the same command with a larger `--calib_num_batches` only when deliberately
regenerating calibration. Do not mix encodings generated with different
`heads_per_split`.

## 3. Build Full LPBQ Encodings

Full LPBQ reuses the per-channel baseline activation and non-W4 parameter
encodings, then replaces the original W4 tensor set with LPBQ entries. This
keeps decode/prefill parameter sets aligned.

```bash
python compute_quant_encodings_experimental.py \
  /models/rwkv7-g1f-7.2b-20260414-ctx8192.pth \
  --output_folder quant_export/g1f-7b-lpbq-w4/ \
  --binidx_dataset ./1 \
  --calib_num_batches 1 \
  --heads_per_split 8 \
  --blockwise_quant \
  --reuse_encodings_folder quant_export/g1f-7b-w4
```

The baseline W4 set is:

- `blocks.*.att.heads.*.receptance.weight`
- `blocks.*.att.heads.*.key.weight`
- `blocks.*.att.heads.*.value.weight`, excluding layer 0 value
- `blocks.*.ffn.key.weight`
- `blocks.*.ffn.value.weight`

## 4. Build Hybrid Encodings

Generate hybrid encodings by replacing selected per-channel W4 parameter
entries with LPBQ entries from the full LPBQ encoding files.

```bash
python tools/make_7b_hybrid_encodings.py
```

Default outputs:

- `quant_export/g1f-7b-mix-att-r-w4`
- `quant_export/g1f-7b-mix-att-k-w4`
- `quant_export/g1f-7b-mix-att-v-w4`
- `quant_export/g1f-7b-mix-att-rkv-w4`
- `quant_export/g1f-7b-mix-ffn-key-w4`
- `quant_export/g1f-7b-mix-ffn-value-w4`
- `quant_export/g1f-7b-mix-ffn-kv-w4`

Recommended candidate:

```text
mix-att-rkv = LPBQ for attention receptance/key/value, per-channel for FFN
```

## 5. Optional Extra LPBQ Scopes

Extra scopes are not part of the baseline W4 tensor set. Use them only for
controlled experiments.

Full LPBQ plus `att.output`:

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

Then generate `mix-att-rkv-output` from that LPBQ source:

```bash
python tools/make_7b_hybrid_encodings.py \
  --variants mix-att-rkv-output \
  --lpbq_dir quant_export/g1f-7b-lpbq-extra-output-w4
```

Supported extra module groups are:

- `att_output`
- `lora`
- `head`

Use `--allow_extra_lpbq` only when adding these extra groups intentionally.

## 6. Convert To QNN And Pack

For one-off conversion plus benchmark/LAMBADA automation, use:

```bash
RWKV_ADB_SERIAL=3e56125d \
python tools/run_7b_hybrid_qnn_experiments.py \
  --variants mix-att-rkv
```

That script performs:

1. `convert_model_dlc.py` decode graph with `--wkv_customop --ext_embedding`
2. `convert_model_dlc.py` prefill graph with `--wkv_customop --ext_embedding --prefill_model`
3. `make_context_cache_binary_dlc.py` for chunks `1..5`
4. `pack_model_file.py` with external embedding and spill/fill buffer
5. `adb push`
6. `simple_benchmark`
7. 500-sample LAMBADA

Equivalent manual conversion for `mix-att-rkv`:

```bash
MODEL=/models/rwkv7-g1f-7.2b-20260414-ctx8192.pth
OUT=rwkv7-g1f-7.2b-20260414-ctx8192-a16w4-mix-att-rkv
ENC=quant_export/g1f-7b-mix-att-rkv-w4/rwkv7-g1f-7.2b-20260414-ctx8192.encodings
ENC_PREFILL=quant_export/g1f-7b-mix-att-rkv-w4/rwkv7-g1f-7.2b-20260414-ctx8192_prefill.encodings

python convert_model_dlc.py "$MODEL" \
  --chunks 5 \
  --wkv_customop \
  --ext_embedding \
  --heads_per_split 8 \
  --output_name "$OUT" \
  --quant_encodings "$ENC"

python convert_model_dlc.py "$MODEL" \
  --chunks 5 \
  --wkv_customop \
  --ext_embedding \
  --prefill_model \
  --heads_per_split 8 \
  --output_name "$OUT" \
  --quant_encodings "$ENC_PREFILL"

for chunk in 1 2 3 4 5; do
  python make_context_cache_binary_dlc.py --wkv_customop \
    --output_name "${OUT}-8elitegen5_chunk${chunk}of5" \
    "onnx/${OUT}_chunk${chunk}of5_embedding/${OUT}_embedding_chunk${chunk}of5.dlc,onnx/${OUT}_chunk${chunk}of5_embedding_prefill/${OUT}_embedding_prefill_chunk${chunk}of5.dlc" \
    output/ SM8850
done

python pack_model_file.py \
  --hidden_size 4096 \
  --vocab_size 65536 \
  --quant_type a16w4 \
  --target_platform SM8850 \
  --model_files output/${OUT}-8elitegen5_chunk1of5.bin,output/${OUT}-8elitegen5_chunk2of5.bin,output/${OUT}-8elitegen5_chunk3of5.bin,output/${OUT}-8elitegen5_chunk4of5.bin,output/${OUT}-8elitegen5_chunk5of5.bin \
  --external_embedding_file "onnx/${OUT}_chunk1of5.uint16.emb" \
  --external_embedding_dtype uint16 \
  --spill_fill_buffer_size 320000000 \
  --output "output/${OUT}-8elitegen5.rmpack"
```

## 7. Using `parallel_convert_pipeline.py`

`parallel_convert_pipeline.py` currently has a built-in `7.2B-w4` entry for the
per-channel baseline:

```bash
python parallel_convert_pipeline.py \
  --filter_model 7.2B-w4 \
  --filter_soc 8elitegen5
```

To use it for `mix-att-rkv`, change or add a `MODEL_FILES` entry with:

```python
"encoding": "quant_export/g1f-7b-mix-att-rkv-w4/rwkv7-g1f-7.2b-20260414-ctx8192.encodings",
"size": "7.2",
"quant": "a16w4",
"extra_pack_args": ["--spill_fill_buffer_size 320000000"],
"need_embed_graph": True,
"embed_graph_only": True,
"heads_per_split": 8,
```

The generated prefill encoding path is inferred by replacing `.encodings` with
`_prefill.encodings`, so both files must exist in the same directory.

Useful stage skips:

```bash
python parallel_convert_pipeline.py --filter_model 7.2B-w4 --filter_soc 8elitegen5 --skip_convert
python parallel_convert_pipeline.py --filter_model 7.2B-w4 --filter_soc 8elitegen5 --skip_binary_gen
python parallel_convert_pipeline.py --filter_model 7.2B-w4 --filter_soc 8elitegen5 --skip_pack
```

## Verification

Minimum checks after producing a new `.rmpack`:

- Generate a short text sample and compare qualitatively with previous good run.
- Run `simple_benchmark` and record prefill/decode token/s.
- Run at least 500-sample LAMBADA before comparing variants.
- For final choice, run full LAMBADA, because the current summary table is based
  on the first 500 samples.

Existing experiment summary is in `lpbq_hybrid_experiment_results.md`.
