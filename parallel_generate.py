#!/usr/bin/env python3
"""
vibed:
并行执行make_context_cache_binary_dlc.py脚本的工具
支持多种机型的并行处理
"""

import subprocess
import os
import sys
import time
import threading
from queue import Queue

DEVICE_MATRIX = {
    "8gen3": "SM8650",
    "8elite": "SM8750",
    "8sgen3": "SM8635",
    "8gen2": "SM8550",
    # "778": "SM7325",
}

NUM_PARALLEL = 16

def execute_command_subprocess(device_name, soc_code, base_cmd, result_queue):
    cmd = base_cmd.replace("SM8750", soc_code).replace("8elite", device_name)

    print(f"[{device_name}] 开始执行: {cmd}")

    try:
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            env=os.environ.copy()
        )

        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(f"[{device_name}] 执行成功")
            result_queue.put(f"[{device_name}] 成功")
        else:
            print(f"[{device_name}] 执行失败: {stderr}")
            result_queue.put(f"[{device_name}] 失败: {stderr}")

    except Exception as e:
        print(f"[{device_name}] 执行异常: {str(e)}")
        result_queue.put(f"[{device_name}] 异常: {str(e)}")

def worker_thread(task_queue, result_queue):
    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:
                break

            device_name, soc_code, base_cmd = task
            execute_command_subprocess(device_name, soc_code, base_cmd, result_queue)
            task_queue.task_done()

        except:
            break

def main():
    global NUM_PARALLEL
    base_commands = [
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk1of4 onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk1of4/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk1of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk1of4/rwkv7-g1a2-1.5b-20251005-ctx8192_prefill_chunk1of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk1of4/rwkv7-g1a2-1.5b-20251005-ctx8192_bsz2_chunk1of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk1of4/rwkv7-g1a2-1.5b-20251005-ctx8192_bsz4_chunk1of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk1of4/rwkv7-g1a2-1.5b-20251005-ctx8192_bsz6_chunk1of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk1of4/rwkv7-g1a2-1.5b-20251005-ctx8192_bsz8_chunk1of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk2of4 onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk2of4/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk2of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk2of4/rwkv7-g1a2-1.5b-20251005-ctx8192_prefill_chunk2of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk2of4/rwkv7-g1a2-1.5b-20251005-ctx8192_bsz2_chunk2of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk2of4/rwkv7-g1a2-1.5b-20251005-ctx8192_bsz4_chunk2of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk2of4/rwkv7-g1a2-1.5b-20251005-ctx8192_bsz6_chunk2of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk2of4/rwkv7-g1a2-1.5b-20251005-ctx8192_bsz8_chunk2of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk3of4 onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk3of4/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk3of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk3of4/rwkv7-g1a2-1.5b-20251005-ctx8192_prefill_chunk3of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk3of4/rwkv7-g1a2-1.5b-20251005-ctx8192_bsz2_chunk3of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk3of4/rwkv7-g1a2-1.5b-20251005-ctx8192_bsz4_chunk3of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk3of4/rwkv7-g1a2-1.5b-20251005-ctx8192_bsz6_chunk3of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk3of4/rwkv7-g1a2-1.5b-20251005-ctx8192_bsz8_chunk3of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk4of4 onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk4of4/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk4of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk4of4/rwkv7-g1a2-1.5b-20251005-ctx8192_prefill_chunk4of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk4of4/rwkv7-g1a2-1.5b-20251005-ctx8192_bsz2_chunk4of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk4of4/rwkv7-g1a2-1.5b-20251005-ctx8192_bsz4_chunk4of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk4of4/rwkv7-g1a2-1.5b-20251005-ctx8192_bsz6_chunk4of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk4of4/rwkv7-g1a2-1.5b-20251005-ctx8192_bsz8_chunk4of4.dlc output/ SM8750",

        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk1of4 onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk1of4/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk1of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk1of4/rwkv7-g1a2-1.5b-20251005-ctx8192_prefill_chunk1of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk2of4 onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk2of4/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk2of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk2of4/rwkv7-g1a2-1.5b-20251005-ctx8192_prefill_chunk2of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk3of4 onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk3of4/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk3of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk3of4/rwkv7-g1a2-1.5b-20251005-ctx8192_prefill_chunk3of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk4of4 onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk4of4/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk4of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk4of4/rwkv7-g1a2-1.5b-20251005-ctx8192_prefill_chunk4of4.dlc output/ SM8750",

        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk1of4 onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk1of4/rwkv7-g1a2-1.5b-20251005-ctx8192_embedding_chunk1of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk1of4/rwkv7-g1a2-1.5b-20251005-ctx8192_embedding_prefill_chunk1of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk2of4 onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk2of4/rwkv7-g1a2-1.5b-20251005-ctx8192_embedding_chunk2of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk2of4/rwkv7-g1a2-1.5b-20251005-ctx8192_embedding_prefill_chunk2of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk3of4 onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk3of4/rwkv7-g1a2-1.5b-20251005-ctx8192_embedding_chunk3of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk3of4/rwkv7-g1a2-1.5b-20251005-ctx8192_embedding_prefill_chunk3of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk4of4 onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk4of4/rwkv7-g1a2-1.5b-20251005-ctx8192_embedding_chunk4of4.dlc,onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk4of4/rwkv7-g1a2-1.5b-20251005-ctx8192_embedding_prefill_chunk4of4.dlc output/ SM8750",

        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk1of4 onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk1of4/rwkv7-g1a2-1.5b-20251005-ctx8192_embedding_chunk1of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk2of4 onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk2of4/rwkv7-g1a2-1.5b-20251005-ctx8192_embedding_chunk2of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk3of4 onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk3of4/rwkv7-g1a2-1.5b-20251005-ctx8192_embedding_chunk3of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk4of4 onnx/rwkv7-g1a2-1.5b-20251005-ctx8192_chunk4of4/rwkv7-g1a2-1.5b-20251005-ctx8192_embedding_chunk4of4.dlc output/ SM8750",

        "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite onnx/rwkv7-g1a2-1.5b-20251005-ctx8192/rwkv7-g1a2-1.5b-20251005-ctx8192_ext_embedding.dlc output/ SM8750",

        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a-0.1b-20250728-ctx4096-a16w8-8elite onnx/rwkv7-g1a-0.1b-20250728-ctx4096/rwkv7-g1a-0.1b-20250728-ctx4096.dlc,onnx/rwkv7-g1a-0.1b-20250728-ctx4096/rwkv7-g1a-0.1b-20250728-ctx4096_prefill.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7a-g1b-0.1b-20250819-ctx4096-a16w8-8elite onnx/rwkv7a-g1b-0.1b-20250819-ctx4096/rwkv7a-g1b-0.1b-20250819-ctx4096.dlc,onnx/rwkv7a-g1b-0.1b-20250819-ctx4096/rwkv7a-g1b-0.1b-20250819-ctx4096_prefill.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a-0.4b-20250905-ctx4096-a16w8-8elite onnx/rwkv7-g1a-0.4b-20250905-ctx4096/rwkv7-g1a-0.4b-20250905-ctx4096.dlc,onnx/rwkv7-g1a-0.4b-20250905-ctx4096/rwkv7-g1a-0.4b-20250905-ctx4096_prefill.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1-1.5b-Lonely-Neko-a16w8-8elite onnx/rwkv7-g1-1.5b-Lonely-Neko/rwkv7-g1-1.5b-Lonely-Neko.dlc,onnx/rwkv7-g1-1.5b-Lonely-Neko/rwkv7-g1-1.5b-Lonely-Neko_prefill.dlc output/ SM8750",

        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a-2.9b-20250924-ctx4096-a16w4-8elite onnx/rwkv7-g1a-2.9b-20250924-ctx4096/rwkv7-g1a-2.9b-20250924-ctx4096.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096/rwkv7-g1a-2.9b-20250924-ctx4096_prefill.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a-2.9b-20250924-ctx4096-a16w4-8elite_chunk1of4 onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk1of4/rwkv7-g1a-2.9b-20250924-ctx4096_chunk1of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk1of4/rwkv7-g1a-2.9b-20250924-ctx4096_prefill_chunk1of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk1of4/rwkv7-g1a-2.9b-20250924-ctx4096_bsz2_chunk1of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk1of4/rwkv7-g1a-2.9b-20250924-ctx4096_bsz4_chunk1of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk1of4/rwkv7-g1a-2.9b-20250924-ctx4096_bsz6_chunk1of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk1of4/rwkv7-g1a-2.9b-20250924-ctx4096_bsz8_chunk1of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a-2.9b-20250924-ctx4096-a16w4-8elite_chunk2of4 onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk2of4/rwkv7-g1a-2.9b-20250924-ctx4096_chunk2of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk2of4/rwkv7-g1a-2.9b-20250924-ctx4096_prefill_chunk2of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk2of4/rwkv7-g1a-2.9b-20250924-ctx4096_bsz2_chunk2of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk2of4/rwkv7-g1a-2.9b-20250924-ctx4096_bsz4_chunk2of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk2of4/rwkv7-g1a-2.9b-20250924-ctx4096_bsz6_chunk2of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk2of4/rwkv7-g1a-2.9b-20250924-ctx4096_bsz8_chunk2of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a-2.9b-20250924-ctx4096-a16w4-8elite_chunk3of4 onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk3of4/rwkv7-g1a-2.9b-20250924-ctx4096_chunk3of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk3of4/rwkv7-g1a-2.9b-20250924-ctx4096_prefill_chunk3of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk3of4/rwkv7-g1a-2.9b-20250924-ctx4096_bsz2_chunk3of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk3of4/rwkv7-g1a-2.9b-20250924-ctx4096_bsz4_chunk3of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk3of4/rwkv7-g1a-2.9b-20250924-ctx4096_bsz6_chunk3of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk3of4/rwkv7-g1a-2.9b-20250924-ctx4096_bsz8_chunk3of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a-2.9b-20250924-ctx4096-a16w4-8elite_chunk4of4 onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk4of4/rwkv7-g1a-2.9b-20250924-ctx4096_chunk4of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk4of4/rwkv7-g1a-2.9b-20250924-ctx4096_prefill_chunk4of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk4of4/rwkv7-g1a-2.9b-20250924-ctx4096_bsz2_chunk4of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk4of4/rwkv7-g1a-2.9b-20250924-ctx4096_bsz4_chunk4of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk4of4/rwkv7-g1a-2.9b-20250924-ctx4096_bsz6_chunk4of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk4of4/rwkv7-g1a-2.9b-20250924-ctx4096_bsz8_chunk4of4.dlc output/ SM8750",

        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a-2.9b-20250924-ctx4096-a16w4-8elite_chunk1of4 onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk1of4/rwkv7-g1a-2.9b-20250924-ctx4096_chunk1of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk1of4/rwkv7-g1a-2.9b-20250924-ctx4096_prefill_chunk1of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a-2.9b-20250924-ctx4096-a16w4-8elite_chunk2of4 onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk2of4/rwkv7-g1a-2.9b-20250924-ctx4096_chunk2of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk2of4/rwkv7-g1a-2.9b-20250924-ctx4096_prefill_chunk2of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a-2.9b-20250924-ctx4096-a16w4-8elite_chunk3of4 onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk3of4/rwkv7-g1a-2.9b-20250924-ctx4096_chunk3of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk3of4/rwkv7-g1a-2.9b-20250924-ctx4096_prefill_chunk3of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1a-2.9b-20250924-ctx4096-a16w4-8elite_chunk4of4 onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk4of4/rwkv7-g1a-2.9b-20250924-ctx4096_chunk4of4.dlc,onnx/rwkv7-g1a-2.9b-20250924-ctx4096_chunk4of4/rwkv7-g1a-2.9b-20250924-ctx4096_prefill_chunk4of4.dlc output/ SM8750",

        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1-2.9b-20250519-ctx4096-a16w4-8elite_chunk1of4 onnx/rwkv7-g1-2.9b-20250519-ctx4096_chunk1of4/rwkv7-g1-2.9b-20250519-ctx4096_chunk1of4.dlc,onnx/rwkv7-g1-2.9b-20250519-ctx4096_chunk1of4/rwkv7-g1-2.9b-20250519-ctx4096_prefill_chunk1of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1-2.9b-20250519-ctx4096-a16w4-8elite_chunk2of4 onnx/rwkv7-g1-2.9b-20250519-ctx4096_chunk2of4/rwkv7-g1-2.9b-20250519-ctx4096_chunk2of4.dlc,onnx/rwkv7-g1-2.9b-20250519-ctx4096_chunk2of4/rwkv7-g1-2.9b-20250519-ctx4096_prefill_chunk2of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1-2.9b-20250519-ctx4096-a16w4-8elite_chunk3of4 onnx/rwkv7-g1-2.9b-20250519-ctx4096_chunk3of4/rwkv7-g1-2.9b-20250519-ctx4096_chunk3of4.dlc,onnx/rwkv7-g1-2.9b-20250519-ctx4096_chunk3of4/rwkv7-g1-2.9b-20250519-ctx4096_prefill_chunk3of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g1-2.9b-20250519-ctx4096-a16w4-8elite_chunk4of4 onnx/rwkv7-g1-2.9b-20250519-ctx4096_chunk4of4/rwkv7-g1-2.9b-20250519-ctx4096_chunk4of4.dlc,onnx/rwkv7-g1-2.9b-20250519-ctx4096_chunk4of4/rwkv7-g1-2.9b-20250519-ctx4096_prefill_chunk4of4.dlc output/ SM8750",

        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-0.1B-g1-respark-voice-tunable-ipa-a16w8-8elite onnx/rwkv7-0.1B-g1-respark-voice-tunable-ipa/rwkv7-0.1B-g1-respark-voice-tunable-ipa.dlc,onnx/rwkv7-0.1B-g1-respark-voice-tunable-ipa/rwkv7-0.1B-g1-respark-voice-tunable-ipa_prefill.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-0.4B-g1-respark-voice-tunable-ipa-a16w8-8elite onnx/rwkv7-0.4B-g1-respark-voice-tunable_ipa/rwkv7-0.4B-g1-respark-voice-tunable_ipa.dlc,onnx/rwkv7-0.4B-g1-respark-voice-tunable_ipa/rwkv7-0.4B-g1-respark-voice-tunable_ipa_prefill.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name RWKV_v7_G1a_0.4B_Translate_ctx4096_20250915-a16w8-8elite onnx/RWKV_v7_G1a_0.4B_Translate_ctx4096_20250915_latest/RWKV_v7_G1a_0.4B_Translate_ctx4096_20250915_latest.dlc,onnx/RWKV_v7_G1a_0.4B_Translate_ctx4096_20250915_latest/RWKV_v7_G1a_0.4B_Translate_ctx4096_20250915_latest_prefill.dlc output/ SM8750",

        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g0-7.2b-20250722-ctx4096-a16w4-8elite_chunk1of4 onnx/rwkv7-g0-7.2b-20250722-ctx4096_chunk1of4/rwkv7-g0-7.2b-20250722-ctx4096_embedding_chunk1of4.dlc,onnx/rwkv7-g0-7.2b-20250722-ctx4096_chunk1of4/rwkv7-g0-7.2b-20250722-ctx4096_embedding_prefill_chunk1of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g0-7.2b-20250722-ctx4096-a16w4-8elite_chunk2of4 onnx/rwkv7-g0-7.2b-20250722-ctx4096_chunk2of4/rwkv7-g0-7.2b-20250722-ctx4096_embedding_chunk2of4.dlc,onnx/rwkv7-g0-7.2b-20250722-ctx4096_chunk2of4/rwkv7-g0-7.2b-20250722-ctx4096_embedding_prefill_chunk2of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g0-7.2b-20250722-ctx4096-a16w4-8elite_chunk3of4 onnx/rwkv7-g0-7.2b-20250722-ctx4096_chunk3of4/rwkv7-g0-7.2b-20250722-ctx4096_embedding_chunk3of4.dlc,onnx/rwkv7-g0-7.2b-20250722-ctx4096_chunk3of4/rwkv7-g0-7.2b-20250722-ctx4096_embedding_prefill_chunk3of4.dlc output/ SM8750",
        # "python make_context_cache_binary_dlc.py --wkv_customop --output_name rwkv7-g0-7.2b-20250722-ctx4096-a16w4-8elite_chunk4of4 onnx/rwkv7-g0-7.2b-20250722-ctx4096_chunk4of4/rwkv7-g0-7.2b-20250722-ctx4096_embedding_chunk4of4.dlc,onnx/rwkv7-g0-7.2b-20250722-ctx4096_chunk4of4/rwkv7-g0-7.2b-20250722-ctx4096_embedding_prefill_chunk4of4.dlc output/ SM8750",
    ]

    task_queue = Queue()
    result_queue = Queue()

    total_tasks = 0
    for device_name, soc_code in DEVICE_MATRIX.items():
        for cmd in base_commands:
            task_queue.put((device_name, soc_code, cmd))
            total_tasks += 1

    threads = []
    for i in range(NUM_PARALLEL):
        t = threading.Thread(target=worker_thread, args=(task_queue, result_queue))
        t.daemon = True
        t.start()
        threads.append(t)

    task_queue.join()

    for _ in range(NUM_PARALLEL):
        task_queue.put(None)

    for t in threads:
        t.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    print("\n=== 执行结果统计 ===")
    success_count = 0
    fail_count = 0
    
    for result in results:
        if "成功" in result:
            success_count += 1
        else:
            fail_count += 1
            print(result)

    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总计: {len(results)}")
    print(f"预期任务数: {total_tasks}")

    packing_commands = [
        # 1.5B
        # "python pack_model_file.py --hidden_size 2048 --vocab_size 65536 --quant_type a16w8 --target_platform SM8750 --model_files output/rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk1of4.bin,output/rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk2of4.bin,output/rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk3of4.bin,output/rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite_chunk4of4.bin --output output/rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite-batch.rmpack && rm -rf output/rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite*.bin",

        # 1.5B ext_embedding
        # "python pack_model_file.py --hidden_size 2048 --vocab_size 65536 --quant_type a16w8 --target_platform SM8750 --model_files output/rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite.bin --external_embedding_dtype uint16 --external_embedding_file onnx/rwkv7-g1a2-1.5b-20251005-ctx8192/rwkv7-g1a2-1.5b-20251005-ctx8192.uint16.emb --output output/rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite-mmap_embedding.rmpack && rm -rf output/rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite*.bin",
        "python pack_model_file.py --hidden_size 2048 --vocab_size 65536 --quant_type a16w4 --target_platform SM8750 --model_files output/rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite.bin --external_embedding_dtype uint16 --external_embedding_file onnx/rwkv7-g1a2-1.5b-20251005-ctx8192/rwkv7-g1a2-1.5b-20251005-ctx8192.uint16.emb --output output/rwkv7-g1a2-1.5b-20251005-ctx8192-a16w4-8elite-mmap_embedding.rmpack && rm -rf output/rwkv7-g1a2-1.5b-20251005-ctx8192-a16w8-8elite*.bin",

        # "python pack_model_file.py --hidden_size 2560 --vocab_size 65536 --quant_type a16w4 --target_platform SM8750 --model_files output/rwkv7-g1a-2.9b-20250924-ctx4096-a16w4-8elite_chunk1of4.bin,output/rwkv7-g1a-2.9b-20250924-ctx4096-a16w4-8elite_chunk2of4.bin,output/rwkv7-g1a-2.9b-20250924-ctx4096-a16w4-8elite_chunk3of4.bin,output/rwkv7-g1a-2.9b-20250924-ctx4096-a16w4-8elite_chunk4of4.bin --output output/rwkv7-g1a-2.9b-20250924-ctx4096-a16w4-8elite-batch.rmpack && rm -rf output/rwkv7-g1a-2.9b-20250924-ctx4096-a16w4-8elite*.bin",
        # "python pack_model_file.py --hidden_size 4096 --vocab_size 65536 --quant_type a16w4 --target_platform SM8750 --model_files output/rwkv7-g0-7.2b-20250722-ctx4096-a16w4-8elite_chunk1of4.bin,output/rwkv7-g0-7.2b-20250722-ctx4096-a16w4-8elite_chunk2of4.bin,output/rwkv7-g0-7.2b-20250722-ctx4096-a16w4-8elite_chunk3of4.bin,output/rwkv7-g0-7.2b-20250722-ctx4096-a16w4-8elite_chunk4of4.bin --spill_fill_buffer_size 320000000 --external_embedding_file onnx/rwkv7-g0-7.2b-20250722-ctx4096_chunk1of4.uint16.emb --external_embedding_dtype uint16 --output output/rwkv7-g0-7.2b-20250722-ctx4096-a16w4-8elite.rmpack"# && rm -rf output/rwkv7-g0-7.2b-20250722-ctx4096-a16w4-8elite*.bin",
        # "python pack_model_file.py --hidden_size 768 --vocab_size 65536 --quant_type a16w8 --target_platform SM8750 --model_files output/rwkv7a-g1b-0.1b-20250819-ctx4096-a16w8-8elite.bin --output output/rwkv7a-g1b-0.1b-20250819-ctx4096-a16w8-8elite.rmpack --external_deep_embedding_file onnx/rwkv7a-g1b-0.1b-20250819-ctx4096/rwkv7a-g1b-0.1b-20250819-ctx4096.uint16.deepemb --external_deep_embedding_dtype uint16 --deep_emb_size 1024 && rm -rf output/rwkv7a-g1b-0.1b-20250819-ctx4096-a16w8-8elite.bin"
    ]

    if len(packing_commands) == 0:
        return
    
    NUM_PARALLEL = 1
    print("\n=== 开始执行打包命令 ===")
    # 创建任务队列和结果队列
    task_queue = Queue()
    result_queue = Queue()
    
    # 添加所有任务到队列
    total_tasks = 0
    for device_name, soc_code in DEVICE_MATRIX.items():
        for cmd in packing_commands:
            task_queue.put((device_name, soc_code, cmd))
            total_tasks += 1

    # 创建工作线程
    threads = []
    for i in range(NUM_PARALLEL):
        t = threading.Thread(target=worker_thread, args=(task_queue, result_queue))
        t.daemon = True
        t.start()
        threads.append(t)
    
    print(f"已启动 {len(threads)} 个工作线程")
    
    # 等待所有任务完成
    task_queue.join()
    
    # 发送结束信号给所有线程
    for _ in range(NUM_PARALLEL):
        task_queue.put(None)
    
    # 等待所有线程结束
    for t in threads:
        t.join()
    
    # 收集结果
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # 输出结果统计
    print("\n=== 执行结果统计 ===")
    success_count = 0
    fail_count = 0
    
    for result in results:
        if "成功" in result:
            success_count += 1
        else:
            fail_count += 1
            print(result)
    
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总计: {len(results)}")
    print(f"预期任务数: {total_tasks}")

if __name__ == "__main__":
    main()
