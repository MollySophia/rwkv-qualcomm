#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import struct
import os
import json
from typing import Dict, List, Tuple, BinaryIO
from pathlib import Path
import argparse

class RWKVModelPacker:
    MAGIC_HEADER = b"RWKVMBLE"
    ALIGNMENT = 4096  # 4KB对齐

    def __init__(self):
        self.config: Dict[str, int] = {}
        self.files: List[Tuple[str, int, int]] = []  # (filename, size, offset)
        self.binary_data: List[bytes] = []

    def add_config(self, key: str, value: int):
        self.config[key] = value

    def add_file(self, file_path: str, file_name: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with open(file_path, 'rb') as f:
            data = f.read()
            size = len(data)
            self.binary_data.append(data)
            self.files.append((file_name, size, 0))  # offset稍后计算

    def _calculate_offsets(self):
        """计算所有文件的偏移量"""
        # 文件头大小
        offset = len(self.MAGIC_HEADER)
        
        # 配置项大小
        config_json = json.dumps(self.config, ensure_ascii=False).encode('utf-8')
        offset += 4 + len(config_json)  # 4字节存储配置项长度

        # 文件信息大小
        offset += 4  # 文件数量
        for filename, size, _ in self.files:
            offset += 4 + len(filename.encode('utf-8')) + 8  # 文件名长度 + 文件名 + size(8字节) + offset(8字节)

        # 对齐到4KB边界
        padding_size = (self.ALIGNMENT - (offset % self.ALIGNMENT)) % self.ALIGNMENT
        offset += padding_size

        # 更新文件偏移量,每个文件都4KB对齐
        for i, (filename, size, _) in enumerate(self.files):
            # 确保每个文件的offset都是4KB对齐的
            padding_size = (self.ALIGNMENT - (offset % self.ALIGNMENT)) % self.ALIGNMENT
            offset += padding_size
            self.files[i] = (filename, size, offset)
            offset += size

    def pack(self, output_path: str):
        """打包所有文件到输出路径"""
        self._calculate_offsets()

        with open(output_path, 'wb') as f:
            # 写入文件头
            f.write(self.MAGIC_HEADER)

            # 写入配置项
            config_json = json.dumps(self.config, ensure_ascii=False).encode('utf-8')
            f.write(struct.pack('<I', len(config_json)))  # 配置项长度
            f.write(config_json)

            # 写入文件信息
            f.write(struct.pack('<I', len(self.files)))  # 文件数量
            for filename, size, offset in self.files:
                filename_bytes = filename.encode('utf-8')
                f.write(struct.pack('<I', len(filename_bytes)))  # 文件名长度
                f.write(filename_bytes)  # 文件名
                f.write(struct.pack('<Q', size))  # 文件大小 (8字节)
                f.write(struct.pack('<Q', offset))  # 文件偏移量 (8字节)

            # 写入padding以对齐到4KB边界
            current_pos = f.tell()
            padding_size = (self.ALIGNMENT - (current_pos % self.ALIGNMENT)) % self.ALIGNMENT
            f.write(b'\0' * padding_size)

            # 写入二进制文件内容,每个文件都4KB对齐
            for i, data in enumerate(self.binary_data):
                # 确保当前位置是4KB对齐的
                current_pos = f.tell()
                padding_size = (self.ALIGNMENT - (current_pos % self.ALIGNMENT)) % self.ALIGNMENT
                f.write(b'\0' * padding_size)
                # 写入文件内容
                f.write(data)
        
        print(f"模型文件已打包到: {output_path}")
        print(f"总大小: {os.path.getsize(output_path)} 字节")
        print(f"包含文件数: {len(self.files)}")
        print(f"配置项: {self.config}")
        print(f"文件信息: {self.files}")

# [文件头] [配置项长度] [配置项JSON] [文件数量] [文件信息...] [padding] [二进制内容...]
#  8字节     4字节      N字节        4字节      M字节         P字节    ...

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hidden_size', type=int, required=True, help='hidden size')
    parser.add_argument('--vocab_size', type=int, required=True, help='vocab size')
    parser.add_argument('--quant_type', type=str, default='a16w4', choices=['a16w4', 'a16w8', 'a8w8', 'fp16'], help='quant type')
    parser.add_argument('--target_platform', type=str, required=True, help='target platform')
    parser.add_argument('--output', type=str, required=True, help='output file')
    parser.add_argument('--spill_fill_buffer_size', type=int, default=0, help='spill fill buffer size')
    parser.add_argument('--external_embedding_file', type=Path, default=None, help='external embedding file')
    parser.add_argument('--external_embedding_dtype', type=str, default='uint16', choices=['uint16', 'fp16', 'fp32'], help='external embedding dtype')
    parser.add_argument('--external_deep_embedding_file', type=Path, default=None, help='external deep embedding file')
    parser.add_argument('--external_deep_embedding_dtype', type=str, default='uint16', choices=['uint16', 'fp16', 'fp32'], help='external deep embedding dtype')
    parser.add_argument('--deep_emb_size', type=int, default=0, help='deep embedding size')
    parser.add_argument('--external_lmhead_file', type=Path, default=None, help='external lmhead file')
    parser.add_argument('--external_lmhead_filetype', type=str, default='mnn', choices=['mnn', 'raw_fp32', 'raw_fp16'], help='external lmhead filetype')
    parser.add_argument('--model_files', type=str, required=True, help='model files')
    args = parser.parse_args()

    packer = RWKVModelPacker()
    packer.add_config("hidden_size", args.hidden_size)
    packer.add_config("vocab_size", args.vocab_size)
    packer.add_config("quant_type", args.quant_type)
    packer.add_config("backend", "qnn")
    packer.add_config("target_platform", args.target_platform)
    packer.add_config("spill_fill_buffer_size", args.spill_fill_buffer_size)
    packer.add_config("use_external_embedding", 1 if args.external_embedding_file is not None else 0)
    packer.add_config("external_embedding_dtype", args.external_embedding_dtype if args.external_embedding_file is not None else 'None')
    packer.add_config("use_external_lmhead", 1 if args.external_lmhead_file is not None else 0)
    packer.add_config("external_lmhead_filetype", args.external_lmhead_filetype if args.external_lmhead_file is not None else 'None')
    packer.add_config("use_external_deep_embedding", 1 if args.external_deep_embedding_file is not None else 0)
    packer.add_config("external_deep_embedding_dtype", args.external_deep_embedding_dtype if args.external_deep_embedding_file is not None else 'None')
    packer.add_config("deep_embedding_size", args.deep_emb_size)

    if args.external_embedding_file is not None:
        packer.add_file(args.external_embedding_file, "embedding")

    if args.external_lmhead_file is not None:
        packer.add_file(args.external_lmhead_file, "lmhead")

    if args.external_deep_embedding_file is not None:
        packer.add_file(args.external_deep_embedding_file, "deep_embedding")

    models = args.model_files.split(',')
    if type(models) == str:
        models = [models]
    packer.add_config("n_chunks", len(models))
    for i, model in enumerate(models):
        packer.add_file(model, f"model_{i}")

    packer.pack(args.output)


if __name__ == "__main__":
    main()