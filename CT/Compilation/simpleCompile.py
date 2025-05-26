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
from .compile_qwen import QwenCompiler
from .compile_glm import GlmCompiler

@dataclass
class simpleCompiler():
    model_type: str = field(default="Qwen", metadata={"choices": [
        "Qwen",
        "Glm"
        ]})
    input_datapt: str = field(
        default="",
        metadata={"help": "Path to the quantized model directory"}
    )
    output_datapt: str = field(
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
        if self.model_type == "Qwen":
            self.compiler=QwenCompiler(
                quantize_model=self.input_datapt,
                compiled_model=self.output_datapt,
                wbits=self.wbits,
                group_size=self.group_size,
                device=self.device 
            )
        elif self.model_type == "Glm":
            self.compiler=GlmCompiler(
                model_name=self.input_datapt,
                quantize_model=self.output_datapt,
                wbits=self.wbits,
                group_size=self.group_size
                 
            )
        self._run_compilation()

    def _run_compilation(self):
        """执行实际编译流程"""
        if hasattr(self.compiler, 'compile_model'):
            self.compiler.compile_model()
        else:
            raise NotImplementedError(f"Compiler {type(self.compiler)} has no compile_model method")