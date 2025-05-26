# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import numpy as np
import os

@dataclass
class EmbeddingSplitter:
    # 输入参数
    bin_file: str = field(
        metadata={"help": "Path to the embedding binary file"}
    )
    numb: int = field(
        default=16,
        metadata={"help": "Number of splits to create", "gt": 0}
    )
    output_dir: str = field(
        default="split_embeddings",
        metadata={"help": "Output directory for split files"}
    )

    def __post_init__(self):
        self._verify_inputs()
        self._setup_output()

    def _verify_inputs(self):
        """验证输入参数有效性"""
        if not os.path.exists(self.bin_file):
            raise FileNotFoundError(f"Input file not found: {self.bin_file}")
        if self.numb <= 0:
            raise ValueError("Number of splits must be positive")

    def _setup_output(self):
        """创建输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)

    def read_bin(self):
        """保持原始read_bin实现并增强功能"""
        # 读取二进制数据
        embeddings = np.fromfile(self.bin_file, dtype=np.uint16)
        
        # 计算分块参数
        total_length = len(embeddings)
        block_size = total_length // self.numb
        
        print(f"Total elements: {total_length}, Block size: {block_size}")
        
        # 分割并保存
        for i in range(self.numb):
            self._save_chunk(embeddings, i, block_size)

    def _save_chunk(self, data, index, block_size):
        """保存单个分块"""
        start = index * block_size
        end = (index + 1) * block_size
        chunk = data[start:end]
        
        filename = os.path.join(
            self.output_dir,
            f"Embedding_{index+1:02d}-of-{self.numb:02d}.bin"
        )
        
        print(f"Writing {filename}")
        chunk.tofile(filename)

    @classmethod
    def from_args(cls):
        """命令行参数解析"""
        import argparse
        parser = argparse.ArgumentParser(description="Embedding File Splitter")
        parser.add_argument("--bin_file", required=True)
        parser.add_argument("--numb", type=int, default=16)
        parser.add_argument("--output_dir", default="split_embeddings")
        args = parser.parse_args()
        return cls(**vars(args))

if __name__ == "__main__":
    splitter = EmbeddingSplitter.from_args()
    splitter.read_bin()