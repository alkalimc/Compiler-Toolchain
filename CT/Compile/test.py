from dataclasses import dataclass, field
import os
import logging
import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.utils.eval import EVAL
from gptqmodel.utils.backend import BACKEND
from compile_qwen import *
from GPTQModel import GPTQQuantizer
from compile_qwen import QwenCompiler
compiler = QwenCompiler(
    quantize_model="/data/disk0/Workspace/Compiler-Toolchain/Models/Quanted/Qwen2-7B-Instruct-W4A16-gptq",
    wbits=4,
    group_size=128
)
compiler.compile_model()