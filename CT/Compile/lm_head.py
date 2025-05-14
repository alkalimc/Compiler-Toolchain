# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class ModelQuantizer:
    # 参数字段定义
    model_path: str = field(
        default='/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq',
        metadata={"help": "Path to the pretrained model"}
    )
    output_dir: str = field(
        default='/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq/compile/qwen2_qweight_bin',
        metadata={"help": "Output directory for quantized files"}
    )
    quant_bits: int = field(
        default=4,
        metadata={"help": "Number of bits for quantization", "choices": [4, 8]}
    )
    group_size: int = field(
        default=128,
        metadata={"help": "Group size for quantization"}
    )

    def __post_init__(self):
        self._setup_directories()
        self._verify_paths()
        self.model, self.tokenizer = self.get_model()

    def _setup_directories(self):
        """创建输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)

    def _verify_paths(self):
        """验证模型路径"""
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path not found: {self.model_path}")

    # 保留原始函数实现
    def int32_to_int4(self, tensor):
        """保持原始int32_to_int4实现"""
        mask = torch.tensor([0xF], dtype=torch.int32, device=tensor.device)
        shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], 
                            dtype=torch.int32, device=tensor.device)
        shifts = shifts.view(1, 1, -1)
        tensor_expanded = tensor.unsqueeze(-1)
        return (tensor_expanded >> shifts) & mask

    def out_txt_1(self, data_pt, txt_name):
        """保持原始out_txt_1实现"""
        data = data_pt.detach().cpu().numpy()
        filename = os.path.join('out_txt_glm3', txt_name)
        with open(filename, 'w') as f:
            f.write(' '.join(map(str, data.shape)) + '\n')
            np.savetxt(f, data.flatten(), delimiter=',', newline='\n')

    def out_bin(self, data_pt, bin_name):
        """保持原始out_bin实现"""
        if data_pt.dtype == torch.float32:
            data_pt = data_pt.to(torch.float16)
        with open(bin_name, 'wb') as f:
            f.write(data_pt.numpy().flatten().tobytes())

    def out_txt(self, data_pt_src, txt_name, is_out=False, is_transpose=False):
        """保持原始out_txt实现"""
        data_pt = data_pt_src.T if is_transpose else data_pt_src
        if is_out:
            self.out_bin(data_pt, txt_name.replace('.txt', '.bin'))

    def in_pickle(self, pkl_name, is_weight=True):
        """保持原始in_pickle实现"""
        with open(pkl_name, 'rb') as f:
            weight = pickle.load(f)
        return weight.detach().cpu() if is_weight else weight

    def quantize_group(self, tensor, num_bits):
        """保持原始quantize_group实现（V0911版本）"""
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        qmin, qmax = -(2**(num_bits - 1)), (2**(num_bits - 1)) - 1
        
        if max_val >= torch.abs(min_val):
            scale = 2 * max_val / (2*qmax)
        else:
            max_val = torch.abs(min_val)
            scale = 2 * max_val / (-2*qmin)
            
        return (tensor / scale).to(torch.int), scale

    def quant_minmax(self, weight, _bits, _gs):
        """保持原始quant_minmax实现"""
        h, w = weight.shape
        qweight = torch.zeros(h, w).half()
        qscales = torch.zeros(h//_gs, w).half()
        
        for i in range(0, h, _gs):
            for j in range(w):
                block = weight[i:i+_gs, j]
                qblock, scale = self.quantize_group(block, _bits)
                qweight[i:i+_gs, j] = qblock
                qscales[i//_gs, j] = scale.item()
        return qweight, qscales

    def read_bin(self, bin_name, _dtype=np.float16):
        """保持原始read_bin实现"""
        with open(bin_name, 'rb') as f:
            return torch.tensor(np.frombuffer(f.read(), dtype=_dtype))

    def get_model(self):
        """保持原始get_model实现"""
        print("============  Loading model ============")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return model, tokenizer

    def quantize_output_layer(self):
        """量化输出层主逻辑"""
        output_layer = self.model.lm_head.weight
        Wt, scales = self.quant_minmax(output_layer.transpose(0, 1).to(torch.float32), 
                                     self.quant_bits, self.group_size)
        
        # 保存结果
        self.out_bin(Wt.transpose(0,1).int(), os.path.join(self.output_dir, 'lm_head_qweight.bin'))
        self.out_bin(scales.transpose(0,1).half(), os.path.join(self.output_dir, 'lm_head_scales.bin'))
        '''选择性添加
        output_dir = '/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq/compile/qwen2_qweight_bin'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_bin(Wt.transpose(0,1).to(torch.int32)      , os.path.join(output_dir, f'lm_head_qweight.bin'))
        out_bin(scales.transpose(0,1).to(torch.float16), os.path.join(output_dir, f'lm_head_scales.bin'))
        '''
    @classmethod
    def from_args(cls):
        """命令行参数解析"""
        parser = argparse.ArgumentParser(description="Model Quantization Tool")
        parser.add_argument("--model_path", default='/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq')
        parser.add_argument("--output_dir", default='/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq/compile/qwen2_qweight_bin')
        parser.add_argument("--quant_bits", type=int, default=4)
        parser.add_argument("--group_size", type=int, default=128)
        args = parser.parse_args()
        return cls(**vars(args))

if __name__ == "__main__":
    quantizer = ModelQuantizer.from_args()
    quantizer.quantize_output_layer()