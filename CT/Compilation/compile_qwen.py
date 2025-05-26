# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
import logging
import torch
import torch.nn as nn
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptqmodel import GPTQModel
from gptqmodel.utils.backend import BACKEND
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

@dataclass
class QwenCompiler:
    # Compilation parameters
    quantize_model: str = field(
        metadata={"help": "Path to the quantized model directory"}
    )
    compiled_model: str = field(
        default="",
        metadata={"help": "Path to the compiled model directory"}
    )
    wbits: int = field(
        default=4,
        metadata={"help": "Number of bits used for quantization", "choices": [4, 8]}
    )
    group_size: int = field(
        default=128,
        metadata={"help": "Group size used for quantization"}
    )
    device: str = field(
        default="cuda:0",
        metadata={"help": "Device to use for compilation"}
    )

    def __post_init__(self):
        self._setup_directories()
        self._verify_parameters()

    def _setup_directories(self):
        """Create required output directories"""
        self.qweight_dir = os.path.join(self.compiled_model, 'compile/qwen2_qweight_bin')
        self.fp16_weight_dir = os.path.join(self.compiled_model, 'compile/qwen2_fp16_weight_bin')
        os.makedirs(self.qweight_dir, exist_ok=True)
        os.makedirs(self.fp16_weight_dir, exist_ok=True)

    def _verify_parameters(self):
        """Validate input parameters"""
        if not os.path.exists(self.quantize_model):
            raise ValueError(f"Quantized model directory not found: {self.quantize_model}")

    def _out_bin(self, data_pt, bin_name):
        """Save tensor data to binary file"""
        if data_pt.dtype == torch.float32:
            data_pt = data_pt.to(torch.float16)

        logging.info(f"Saving {bin_name} with shape {data_pt.shape}")
        
        with open(bin_name, 'wb') as f:
            f.write(data_pt.detach().cpu().numpy().flatten().tobytes())

    def process_qweight(self, name, param):
        """Process quantized weights"""
        logging.info(f"Processing qweight: {name}")
        
        qweight = param
        qweight_int4 = torch.empty(qweight.shape[0], 8, qweight.shape[1], 
                                device=qweight.device, dtype=torch.int8)
        
        for i in range(8):
            qweight_int4[:, i, :] = ((qweight >> (4 * i)) & 0xF).to(torch.int8)
        
        qweight_int4 = qweight_int4 - 8
        qweight_int4 = qweight_int4.reshape(qweight.shape[0]*8, qweight.shape[1])
        qweight_trp = qweight_int4.transpose(0, 1).to(torch.int32)
        
        self._out_bin(qweight_trp, os.path.join(self.qweight_dir, f'{name}.bin'))

    def process_scales(self, name, param):
        """Process scale parameters"""
        logging.info(f"Processing scales: {name}")
        scale_trp = param.transpose(0, 1)
        self._out_bin(scale_trp, os.path.join(self.qweight_dir, f'{name}.bin'))

    def process_bias(self, name, param):
        """Process bias parameters"""
        logging.info(f"Processing bias: {name}")
        self._out_bin(param, os.path.join(self.qweight_dir, f'{name}.bin'))

    def process_norm_layers(self, name, param):
        """Process normalization layers"""
        logging.info(f"Processing norm layer: {name}")
        weights = param
        bias = torch.zeros(weights.shape, device=param.device, dtype=torch.float16)
        weights_bias = torch.cat([weights, bias], dim=0)
        self._out_bin(weights_bias, os.path.join(self.fp16_weight_dir, f'{name}.bin'))

    def process_embeddings(self, name, param):
        """Process embedding layers"""
        logging.info(f"Processing embeddings: {name}")
        self._out_bin(param.to(torch.float16), os.path.join(self.fp16_weight_dir, f'{name}.bin'))

    def compile_model(self):
        """Main compilation pipeline"""
        logging.info("Starting model compilation")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.quantize_model,
            torch_dtype="auto",
            device_map=self.device
        )
        
        for name, param in model.state_dict().items():
            logging.debug(f"Processing parameter: {name}")
            
            
            if "qweight" in name:
                self.process_qweight(name, param)
            elif "scales" in name:
                self.process_scales(name, param)
                self.handle_missing_bias(name, model)
            
            elif "bias" in name:
                self.process_bias(name, param)
            elif "norm" in name:
                self.process_norm_layers(name, param)
            elif "embed_tokens" in name:
                self.process_embeddings(name, param)

    def handle_missing_bias(self, name, model):
        """Handle missing bias parameters"""
        bias_name = name.replace("scales", "bias")
        if bias_name not in model.state_dict():
            logging.info(f"Generating zero bias for {bias_name}")
            scales = model.state_dict()[name]
            bias = torch.zeros(1, scales.shape[1], device=scales.device, dtype=torch.float16)
            self._out_bin(bias, os.path.join(self.qweight_dir, f'{bias_name}.bin'))

    @classmethod
    def from_args(cls):
        """参数解析适配器"""
        parser = argparse.ArgumentParser(description="Model Compiler")
        parser.add_argument("--compiled_model", required=True)
        parser.add_argument("--quantize_model", default="")
        parser.add_argument("--wbits", type=int, default=4)
        parser.add_argument("--group_size", type=int, default=128)
        parser.add_argument("--device", default="cuda:0")
        args = parser.parse_args()
        return cls(**vars(args))

if __name__ == "__main__":
    compiler = QwenCompiler.from_args()
    compiler.compile_model()