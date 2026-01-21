#!/usr/bin/env python3
"""
vibed:
Tool for running make_context_cache_binary_dlc.py in parallel.
Supports parallel processing across multiple device targets.
"""

import subprocess
import os
import sys
import time
import threading
from queue import Queue
import random
import argparse

MODEL_FILES = {
    # "sudoku": {
    #     "path": "/home/molly/workspace/rwkv-qualcomm/sudoku_rwkv-v7-L10-D320.pth",

    # },
    "ABC-Music": {
        "path": "/home/molly/workspace/models/RWKV-7-ABC-2024-11-22-15-50-00.pth",
        "encoding": "quant_export/v7-abc/RWKV-7-ABC-2024-11-22-15-50-00.encodings",
        "size": "0.08",
        "quant": "a16w8",
    },
    "0.1B-respark": {
        "path": "/models/rwkv7-0.1B-g1-respark-voice-tunable_ipa.pth",
        "encoding": "quant_export/respark-0b1/rwkv7-0.1B-g1-respark-voice-tunable_ipa.encodings",
        "size": "0.1",
        "quant": "a16w8",
    },
    "0.1B": {
        "path": "/models/rwkv7-g1a-0.1b-20250728-ctx4096.pth",
        "encoding": "quant_export/g1a-0b1/rwkv7-g1a-0.1b-20250728-ctx4096.encodings",
        "size": "0.1",
        "quant": "a16w8",
    },
    "0.4B": {
        "path": "/models/rwkv7-g1a-0.4b-20250905-ctx4096.pth",
        "encoding": "quant_export/g1a-0b4/rwkv7-g1a-0.4b-20250905-ctx4096.encodings",
        "size": "0.4",
        "quant": "a16w8",
    },
    "0.4B-respark": {
        "path": "/models/rwkv7-0.4B-g1-respark-voice-tunable_ipa.pth",
        "encoding": "quant_export/respark-0b4/rwkv7-0.4B-g1-respark-voice-tunable_ipa.encodings",
        "size": "0.4",
        "quant": "a16w8",
    },
    "0.4B-vl": {
        "path": "/models/modrwkv-v3-0.4b-251113.pth",
        "encoding": "quant_export/modrwkv-v3-0b4-251113/modrwkv-v3-0.4b-251113.encodings",
        "size": "0.4",
        "quant": "a16w8",
        "need_embed_graph": True,
    },
    "0.4B-translate": {
        "path": "/models/RWKV_v7_G1a_0.4B_Translate_ctx4096_20250915_latest.pth",
        "encoding": "quant_export/g1a-0b4-translate/RWKV_v7_G1a_0.4B_Translate_ctx4096_20250915_latest.encodings",
        "size": "0.4",
        "quant": "a16w8",
    },
    "1.5B-w8": {
        "path": "/models/rwkv7-g1c-1.5b-20260110-ctx8192.pth",
        "encoding": "quant_export/g1c-1b5-w8/rwkv7-g1c-1.5b-20260110-ctx8192.encodings",
        "size": "1.5",
        "quant": "a16w8",
    },
    "1.5B-w4": {
        "path": "/models/rwkv7-g1c-1.5b-20260110-ctx8192.pth",
        "encoding": "quant_export/g1c-1b5-w4/rwkv7-g1c-1.5b-20260110-ctx8192.encodings",
        "size": "1.5",
        "quant": "a16w4",
    },
    "1.5B-Neko": {
        "path": "/models/rwkv7-g1-1.5b-Lonely-Neko.pth",
        "encoding": "quant_export/g1-1b5-neko/rwkv7-g1-1.5b-Lonely-Neko.encodings",
        "size": "1.5",
        "quant": "a16w8",
    },
    "1.5B-RolePlay": {
        "path": "/models/rwkv7-g1c-1.5B-MiSS-Roleplay-260116-ctx8192.pth",
        "encoding": "quant_export/g1c-1b5-w8-roleplay/rwkv7-g1c-1.5B-MiSS-Roleplay-260116-ctx8192.encodings",
        "size": "1.5",
        "quant": "a16w8",
    },
    "1.5B-translate-w8": {
        "path": "/models/RWKV_v7_G1c_1.5B_Translate_ctx4096_20260118.pth",
        "encoding": "quant_export/g1c-1b5-translate-w8/RWKV_v7_G1c_1.5B_Translate_ctx4096_20260118.encodings",
        "size": "1.5",
        "quant": "a16w8",
    },
    "2.9B-w4": {
        "path": "/models/rwkv7-g1c-2.9b-20251231-ctx8192.pth",
        "encoding": "quant_export/g1c-2b9-w4/rwkv7-g1c-2.9b-20251231-ctx8192.encodings",
        "size": "2.9",
        "quant": "a16w4",
    },
    "2.9B-roleplay-w4": {
        "path": "/models/rwkv7-g1c-2.9B-MiSS-Roleplay-260115-ctx8192.pth",
        "encoding": "quant_export/g1c-2b9-w4-roleplay/rwkv7-g1c-2.9B-MiSS-Roleplay-260115-ctx8192.encodings",
        "size": "2.9",
        "quant": "a16w4",
    },
    # TODO
    # "7.2B-w4": {
    #     "path": "/models/rwkv7-g1c-7.2b-20251231-ctx8192.pth",
    #     "encoding": "quant_export/g0-7b-w4-split8/rwkv7-g0-7.2b-20250722-ctx4096.encodings",
    #     "size": "7.2",
    #     "quant": "a16w4",
    # }
}

HEADS_PER_SPLIT_BY_SIZE = {
    "0.08": 2,
    "0.1": 3,
    "0.4": 4,
    "1.5": 4,
    "2.9": 4,
}

DEVICE_MATRIX = {
    "8gen3": "SM8650",
    "8elite": "SM8750",
    "8sgen3": "SM8635",
    "8elitegen5": "SM8850",
    "8gen2": "SM8550",
    "8plusgen1": "SM8475",
    "6490": "SM7325",
    "xelite": "SC8380",
}

EXTRA_BSZ_CONVERTING = {
    "0.4B-translate": [2, 4, 6, 8, 10, 12, 14, 16],
    "1.5B-w8": [2, 4, 6, 8, 10, 12, 14],
    "1.5B-translate-w8": [2, 4, 6, 8, 10, 12, 14],
    "2.9B-w4": [2, 4, 6, 8],
}

FULL_BSZ_DEVICES = ["8gen3", "8elite", "8sgen3", "8elitegen5", "xelite"]
LIMITED_BSZ_DEVICES = ["8gen2"]

VOCAB_SIZE = 65536

CHUNKS_BY_SIZE = {
    "0.08": 1,
    "0.1": 1,
    "0.4": 4,
    "1.5": 4,
    "2.9": 4,
    "7.2": 4,
}

HIDDEN_SIZE_BY_SIZE = {
    "0.08": 512,
    "0.1": 768,
    "0.4": 1024,
    "1.5": 2048,
    "2.9": 2560,
    "7.2": 4096,
}

def resolve_model_name(model_key, cfg):
    model_name = cfg.get("onnx_name") or cfg.get("model_name")
    if model_name:
        return model_name

    model_path = cfg.get("path")
    if model_path:
        return os.path.splitext(os.path.basename(model_path))[0]

    encoding_path = cfg.get("encoding")
    if encoding_path:
        return os.path.splitext(os.path.basename(encoding_path))[0]

    return model_key

def construct_context_binary_cmd(model_name, num_chunks, added_bszs, quant_type, device_name, device_codename, output_path="output/", need_embed_graph=False):
    """
    Build make_context_cache_binary_dlc.py commands.
    
    Args:
        need_embed_graph: If True, include ext_embedding variants (decode and prefill) in the model list
    """
    # model_name already includes quant suffix from build_context_cache_commands
    if num_chunks == 1:
        # decode and prefill use different directories now
        models = [
            f"onnx/{model_name}/{model_name}.dlc",
            f"onnx/{model_name}_prefill/{model_name}_prefill.dlc"
        ]
        
        # Add ext_embedding variants if needed
        if need_embed_graph:
            models.extend([
                f"onnx/{model_name}_embedding/{model_name}_ext_embedding.dlc",
                f"onnx/{model_name}_embedding_prefill/{model_name}_ext_embedding_prefill.dlc"
            ])
        
        if added_bszs is not None:
            for bsz in added_bszs:
                # dirname includes bsz suffix for batch_size > 1
                dirname = f"onnx/{model_name}_bsz{bsz}"
                models.append(f"{dirname}/{model_name}_bsz{bsz}.dlc")
        model_args = ",".join(models)
        return [
            f"python make_context_cache_binary_dlc.py --wkv_customop --output_name {model_name}-{device_name} {model_args} {output_path} {device_codename}"
        ]
    else:
        cmds = []
        def get_models_for_chunk(chunk_id, num_chunks):
            base_dirname = f"onnx/{model_name}_chunk{chunk_id+1}of{num_chunks}"
            # decode and prefill use different directories now
            models = [
                f"{base_dirname}/{model_name}_chunk{chunk_id+1}of{num_chunks}.dlc",
                f"{base_dirname}_prefill/{model_name}_prefill_chunk{chunk_id+1}of{num_chunks}.dlc"
            ]
            
            # Add ext_embedding variants if needed
            if need_embed_graph:
                models.extend([
                    f"{base_dirname}_embedding/{model_name}_embedding_chunk{chunk_id+1}of{num_chunks}.dlc",
                    f"{base_dirname}_embedding_prefill/{model_name}_embedding_prefill_chunk{chunk_id+1}of{num_chunks}.dlc"
                ])
            
            if added_bszs is not None:
                for bsz in added_bszs:
                    # dirname includes bsz suffix for batch_size > 1
                    dirname = f"{base_dirname}_bsz{bsz}"
                    models.append(f"{dirname}/{model_name}_bsz{bsz}_chunk{chunk_id+1}of{num_chunks}.dlc")
            return models
        for i in range(num_chunks):
            m = get_models_for_chunk(i, num_chunks)
            model_args = ",".join(m)
            cmds.append(
                f"python make_context_cache_binary_dlc.py --wkv_customop "
                f"--output_name {model_name}-{device_name}_chunk{i+1}of{num_chunks} "
                f"{model_args} {output_path} {device_codename}"
            )
        return cmds

def construct_convert_cmd(model_pth, encoding_path, num_chunks, heads_per_split, needed_batchsizes, quant_type, model_key, need_embed_graph=False):
    """
    Build convert_model_dlc.py commands. Returns a list of commands that can be executed in parallel.
    Uses --output_name parameter to ensure different quant types produce different output paths.
    
    Args:
        need_embed_graph: If True, also generate ext_embedding variants (decode and prefill)
    """
    # Generate output name from model path and quant type
    model_basename = os.path.basename(model_pth)
    model_name_without_ext = os.path.splitext(model_basename)[0]
    output_name = f"{model_name_without_ext}-{quant_type}"
    
    prefill_encoding_path = encoding_path.replace(".encodings", "_prefill.encodings")
    cmds = [
        f"python convert_model_dlc.py {model_pth} --chunks {num_chunks} --quant_encodings {encoding_path} --wkv_customop --heads_per_split {heads_per_split} --output_name {output_name}",
        f"python convert_model_dlc.py {model_pth} --chunks {num_chunks} --quant_encodings {prefill_encoding_path} --wkv_customop --prefill_model --heads_per_split {heads_per_split} --output_name {output_name}",
    ]
    
    # Add ext_embedding variants if needed
    if need_embed_graph:
        cmds.extend([
            f"python convert_model_dlc.py {model_pth} --chunks {num_chunks} --quant_encodings {encoding_path} --wkv_customop --ext_embedding --heads_per_split {heads_per_split} --output_name {output_name}",
            f"python convert_model_dlc.py {model_pth} --chunks {num_chunks} --quant_encodings {prefill_encoding_path} --wkv_customop --ext_embedding --prefill_model --heads_per_split {heads_per_split} --output_name {output_name}",
        ])
    
    if needed_batchsizes is not None:
        for bsz in needed_batchsizes:
            cmds.append(f"python convert_model_dlc.py {model_pth} --chunks {num_chunks} --quant_encodings {encoding_path} --wkv_customop --batch_size {bsz} --heads_per_split {heads_per_split} --output_name {output_name}")

    return cmds

def construct_pack_cmd(
    model_name: str,
    model_size: str,
    num_chunks: int,
    quant_type: str,
    device_name: str,
    target_platform: str,
    output_path: str = "output/",
    pack_variant: str = "",
    pack_bszs: list = None,
    extra_pack_args=None,
    cleanup: bool = True,
):
    """
    Build a pack_model_file.py command, matching the formatting style used in packing_commands:
    - Multi-chunk: output/{name}_chunk{i}of{n}.bin -> output/{name}-{pack_variant}.rmpack && rm -rf output/{name}_chunk*.bin
    - Single-chunk: output/{name}.bin -> output/{name}-{pack_variant}.rmpack (rm is omitted unless cleanup=True)
    
    Args:
        pack_bszs: List of batch sizes to include in batch variant. If None, pack nobatch variant.
                   Note: The actual bin files are generated by context cache stage based on get_supported_bszs,
                   so this parameter is mainly for documentation/logging purposes.
    """
    if extra_pack_args is None:
        extra_pack_args = []
    if not isinstance(extra_pack_args, (list, tuple)):
        raise TypeError("extra_pack_args must be a list/tuple, e.g. ['--spill_fill_buffer_size 320000000']")

    hidden_size = HIDDEN_SIZE_BY_SIZE.get(str(model_size))
    if hidden_size is None:
        raise ValueError(f"Unknown model_size={model_size}; please add it to HIDDEN_SIZE_BY_SIZE")

    # model_name already includes quant suffix from build_pack_commands
    name = f"{model_name}-{device_name}"
    if num_chunks == 1:
        model_bins = [f"{output_path}{name}.bin"]
    else:
        model_bins = [f"{output_path}{name}_chunk{i+1}of{num_chunks}.bin" for i in range(num_chunks)]

    model_files_arg = ",".join(model_bins)
    extra = (" " + " ".join(extra_pack_args)) if len(extra_pack_args) > 0 else ""
    output_pack = f"{output_path}{name}{pack_variant}.rmpack"

    if cleanup:
        if num_chunks == 1:
            cleanup_cmd = f" && rm -rf {output_path}{name}.bin"
        else:
            cleanup_cmd = f" && rm -rf {output_path}{name}_chunk*.bin"
    else:
        cleanup_cmd = ""

    return (
        f"python pack_model_file.py --hidden_size {hidden_size} --vocab_size {VOCAB_SIZE} "
        f"--quant_type {quant_type} --target_platform {target_platform} "
        f"--model_files {model_files_arg}{extra} --output {output_pack}{cleanup_cmd}"
    )

NUM_PARALLEL_CONVERT = 8
NUM_PARALLEL_CONTEXT_BINARY = 10
NUM_PARALLEL_PACK = 20

def execute_command_subprocess(device_name, soc_code, cmd, result_queue):
    print(f"[{device_name}] Starting: {cmd}")

    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )

        stdout, stderr = process.communicate()
        if process.returncode == 0:
            return True, f"[{device_name}] Success"
        else:
            err = stderr or stdout
            return False, f"[{device_name}] command failed: {cmd}\nError: {err}"

    except Exception as e:
        return False, f"[{device_name}] Exception: {str(e)}"

def worker_thread(task_queue, result_queue):
    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:
                break

            if len(task) == 3:
                device_name, soc_code, base_cmd = task
                retries_left = 0
            else:
                device_name, soc_code, base_cmd, retries_left = task

            ok, msg = execute_command_subprocess(device_name, soc_code, base_cmd, result_queue)

            if ok:
                result_queue.put(msg)
            else:
                if retries_left > 0:
                    task_queue.put((device_name, soc_code, base_cmd, retries_left - 1))
                    print(f"[{device_name}] Requeued (retries left: {retries_left - 1})")
                else:
                    print(f"=======================\n[{device_name}] failed:")
                    print(msg)
                    print("========================")
                    result_queue.put(msg)

            task_queue.task_done()

        except:
            break

def run_task_batch(
    commands,
    num_parallel: int,
    stage_name: str,
    retries_per_task: int = 0,
):
    """
    Run a list of shell command strings using the shared worker_thread implementation.
    If per_device=True, each command is replicated across DEVICE_MATRIX (with placeholder substitution).
    Failed tasks are requeued to the tail up to `retries_per_task` times.
    """
    if not commands:
        print(f"\n=== {stage_name}: no commands to run ===")
        return []

    print(f"\n=== {stage_name}: starting ({len(commands)} base commands, num_parallel={num_parallel}) ===")

    task_queue = Queue()
    result_queue = Queue()

    total_tasks = 0
    for cmd in commands:
        task_queue.put(("host", "host", cmd, retries_per_task))
        total_tasks += 1

    threads = []
    for _ in range(num_parallel):
        t = threading.Thread(target=worker_thread, args=(task_queue, result_queue), daemon=True)
        t.start()
        threads.append(t)

    task_queue.join()

    # Tell workers to exit cleanly
    for _ in range(num_parallel):
        task_queue.put(None)

    for t in threads:
        t.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    success_count = 0
    fail_count = 0
    for r in results:
        if "Success" in r:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n=== {stage_name}: summary ===")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total results: {len(results)}")
    print(f"Expected tasks: {total_tasks}")
    if fail_count:
        print("Failed task details:")
        for r in results:
            if "Success" not in r:
                print(r)

    return results

def build_convert_commands(selected_models=None):
    """
    Generate convert_model_dlc.py command strings using:
    - MODEL_FILES
    - HEADS_PER_SPLIT_BY_SIZE
    - EXTRA_BSZ_CONVERTING
    - CHUNKS_BY_SIZE (as default, overridable via MODEL_FILES[model]["chunks"])
    """
    cmds = []
    for model_key, cfg in MODEL_FILES.items():
        if selected_models is not None and model_key not in selected_models:
            continue

        model_pth = cfg["path"]
        encoding_path = cfg["encoding"]
        model_size = str(cfg["size"])
        quant_type = cfg["quant"]

        num_chunks = int(cfg.get("chunks") or CHUNKS_BY_SIZE.get(model_size, 1))
        heads_per_split = int(HEADS_PER_SPLIT_BY_SIZE[model_size])
        extra_bszs = EXTRA_BSZ_CONVERTING.get(model_key)
        need_embed_graph = cfg.get("need_embed_graph", False)

        model_cmds = construct_convert_cmd(
            model_pth=model_pth,
            encoding_path=encoding_path,
            num_chunks=num_chunks,
            heads_per_split=heads_per_split,
            needed_batchsizes=extra_bszs,
            quant_type=quant_type,
            model_key=model_key,
            need_embed_graph=need_embed_graph,
        )
        # Extend with list of commands to allow parallel execution
        cmds.extend(model_cmds)

    return cmds

def get_supported_bszs(device_name, model_key, model_size):
    extra_bszs = EXTRA_BSZ_CONVERTING.get(model_key)
    if not extra_bszs:
        return None

    if device_name in FULL_BSZ_DEVICES:
        return extra_bszs

    if device_name in LIMITED_BSZ_DEVICES:
        if str(model_size) == "2.9":
            return None
        if str(model_size) in {"1.5", "0.4"}:
            return [bsz for bsz in extra_bszs if bsz <= 8]
        return None

    return None

def build_context_cache_commands(selected_models=None, selected_socs=None):
    """
    Build make_context_cache_binary_dlc.py commands for each device and model.
    """
    cmds = []
    for device_name, device_codename in DEVICE_MATRIX.items():
        if selected_socs is not None and device_name not in selected_socs:
            continue
        for model_key, cfg in MODEL_FILES.items():
            if selected_models is not None and model_key not in selected_models:
                continue

            base_model_name = resolve_model_name(model_key, cfg)
            model_size = str(cfg["size"])
            quant_type = cfg["quant"]
            # Add quant suffix to match convert stage output paths
            model_name = f"{base_model_name}-{quant_type}"
            num_chunks = int(cfg.get("chunks") or CHUNKS_BY_SIZE.get(model_size, 1))
            added_bszs = get_supported_bszs(device_name, model_key, model_size)
            need_embed_graph = cfg.get("need_embed_graph", False)

            cmds.extend(
                construct_context_binary_cmd(
                    model_name=model_name,
                    num_chunks=num_chunks,
                    added_bszs=added_bszs,
                    quant_type=quant_type,
                    device_name=device_name,
                    device_codename=device_codename,
                    need_embed_graph=need_embed_graph,
                )
            )

    return cmds

def get_pack_bszs(device_name, model_key, model_size):
    """
    Determine which batch sizes to pack based on device type and model size.
    Returns:
    - List of bsz values to pack (for batch variant), or None for nobatch variant
    """
    extra_bszs = EXTRA_BSZ_CONVERTING.get(model_key)
    if not extra_bszs:
        return None  # No batch sizes, pack nobatch only
    
    if device_name in FULL_BSZ_DEVICES:
        # Pack all batch sizes
        return extra_bszs
    
    if device_name in LIMITED_BSZ_DEVICES:
        if str(model_size) == "2.9":
            # 2.9B: pack nobatch only
            return None
        if str(model_size) in {"1.5", "0.4"}:
            # 0.4 and 1.5: pack bsz <= 8
            return [bsz for bsz in extra_bszs if bsz <= 8]
        return None
    
    # Other devices: pack nobatch only
    return None

def build_pack_commands(selected_models=None, selected_socs=None, cleanup=True):
    """
    Build pack_model_file.py commands that correspond to the context cache outputs.
    Packing strategy:
    - FULL_BSZ_DEVICES: pack all bsz sizes (batch variant)
    - LIMITED_BSZ_DEVICES:
      - 2.9B: pack nobatch only
      - 0.4 and 1.5: pack bsz <= 8 (batch variant)
    - Other devices: pack nobatch only
    """
    cmds = []
    for device_name, device_codename in DEVICE_MATRIX.items():
        if selected_socs is not None and device_name not in selected_socs:
            continue
        for model_key, cfg in MODEL_FILES.items():
            if selected_models is not None and model_key not in selected_models:
                continue

            base_model_name = resolve_model_name(model_key, cfg)
            model_size = str(cfg["size"])
            quant_type = cfg["quant"]
            # Add quant suffix to match convert stage output paths
            model_name = f"{base_model_name}-{quant_type}"
            num_chunks = int(cfg.get("chunks") or CHUNKS_BY_SIZE.get(model_size, 1))

            pack_bszs = get_pack_bszs(device_name, model_key, model_size)

            if pack_bszs is not None:
                # Pack batch variant with specified bsz sizes
                cmds.append(
                    construct_pack_cmd(
                        model_name=model_name,
                        model_size=model_size,
                        num_chunks=num_chunks,
                        quant_type=quant_type,
                        device_name=device_name,
                        target_platform=device_codename,
                        pack_variant="-batch",
                        pack_bszs=pack_bszs,
                        cleanup=cleanup,
                    )
                )
            else:
                # Pack nobatch variant
                cmds.append(
                    construct_pack_cmd(
                        model_name=model_name,
                        model_size=model_size,
                        num_chunks=num_chunks,
                        quant_type=quant_type,
                        device_name=device_name,
                        target_platform=device_codename,
                        pack_variant="",
                        pack_bszs=None,
                        cleanup=cleanup,
                    )
                )

    return cmds

def main():
    parser = argparse.ArgumentParser(description='Parallel model conversion pipeline')
    parser.add_argument('--skip_convert', action='store_true', help='Skip convert_model_dlc.py stage')
    parser.add_argument('--skip_binary_gen', action='store_true', help='Skip make_context_cache_binary_dlc.py stage')
    parser.add_argument('--skip_pack', action='store_true', help='Skip pack_model_file.py stage')
    parser.add_argument('--filter_model', type=str, default=None, 
                        help='Filter models to process, comma-separated list (e.g., --filter_model 2.9B-roleplay-w4,1.5B-translate)')
    parser.add_argument('--filter_soc', type=str, default=None,
                        help='Filter SOCs to process, comma-separated list (e.g., --filter_soc 8plusgen1,8gen3)')
    args = parser.parse_args()

    # Parse filter_model parameter
    selected_models = None
    if args.filter_model:
        selected_models = [m.strip() for m in args.filter_model.split(',') if m.strip()]
        print(f"Filtering models: {selected_models}")
        # Validate that all specified models exist
        invalid_models = [m for m in selected_models if m not in MODEL_FILES]
        if invalid_models:
            print(f"Warning: Unknown models specified: {invalid_models}")
            print(f"Available models: {list(MODEL_FILES.keys())}")
            selected_models = [m for m in selected_models if m in MODEL_FILES]
            if not selected_models:
                print("Error: No valid models to process")
                return

    # Parse filter_soc parameter
    selected_socs = None
    if args.filter_soc:
        selected_socs = [s.strip() for s in args.filter_soc.split(',') if s.strip()]
        print(f"Filtering SOCs: {selected_socs}")
        # Validate that all specified SOCs exist
        invalid_socs = [s for s in selected_socs if s not in DEVICE_MATRIX]
        if invalid_socs:
            print(f"Warning: Unknown SOCs specified: {invalid_socs}")
            print(f"Available SOCs: {list(DEVICE_MATRIX.keys())}")
            selected_socs = [s for s in selected_socs if s in DEVICE_MATRIX]
            if not selected_socs:
                print("Error: No valid SOCs to process")
                return

    # Stage 1: convert_model_dlc.py (host-only)
    if not args.skip_convert:
        convert_commands = build_convert_commands(selected_models=selected_models)
        random.shuffle(convert_commands)
        run_task_batch(convert_commands, NUM_PARALLEL_CONVERT, "convert_model_dlc")
    else:
        print("Skipping convert_model_dlc.py stage")

    # Stage 2: make_context_cache_binary_dlc.py (device-specific)
    if not args.skip_binary_gen:
        context_commands = build_context_cache_commands(selected_models=selected_models, selected_socs=selected_socs)
        random.shuffle(context_commands)
        run_task_batch(context_commands, NUM_PARALLEL_CONTEXT_BINARY, "make_context_cache_binary_dlc")
    else:
        print("Skipping make_context_cache_binary_dlc.py stage")

    # Stage 3: pack_model_file.py (device-specific)
    if not args.skip_pack:
        pack_commands = build_pack_commands(selected_models=selected_models, selected_socs=selected_socs)
        run_task_batch(pack_commands, NUM_PARALLEL_PACK, "pack_model_file")
    else:
        print("Skipping pack_model_file.py stage")


if __name__ == "__main__":
    main()
