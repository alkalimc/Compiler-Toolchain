# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

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

@dataclass
class GPTQQuantizer:
    # Model parameters
    model: str = field(
        metadata={"help": "Path or identifier of the base model to be quantized."}
    )
    quantize_model: str = field(
        default="",
        metadata={"help": "Path to store the quantized model. If empty, auto-generate."}
    )
    
    # Quantization parameters
    wbits: int = field(
        default=4,
        metadata={"help": "Number of bits for weight quantization", "choices": [4, 8]}
    )
    group_size: int = field(
        default=128,
        metadata={"help": "Group size for GPTQ quantization"}
    )
    desc_act: bool = field(
        default=False,
        metadata={"help": "Use descending activation in GPTQ (advanced)"}
    )
    sym: bool = field(
        default=False,
        metadata={"help": "Enforce symmetric quantization"}
    )
    use_triton: bool = field(
        default=False,
        metadata={"help": "Use Triton-based kernels if available"}
    )
    
    # Device settings
    device: str = field(
        default="cuda:0",
        metadata={"help": "Device to run quantization/inference"}
    )
    
    # Evaluation parameters
    eval: bool = field(
        default=False,
        metadata={"help": "Run evaluation after quantization"}
    )
    eval_framework: str = field(
        default="lmeval",
        metadata={"help": "Evaluation framework", "choices": ["lmeval", "evalplus"]}
    )
    eval_tasks: str = field(
        default="ARC_CHALLENGE",
        metadata={"help": "Comma-separated tasks for evaluation"}
    )
    output_file: str = field(
        default="eval_result.json",
        metadata={"help": "Output file for evaluation results"}
    )
    
    # Compilation flag
    compile: bool = field(
        default=False,
        metadata={"help": "Run compile step after quantization"}
    )

    def __post_init__(self):
        self._setup_paths()
        self._setup_logging()
        self.quantize_config = QuantizeConfig(
            bits=self.wbits,
            group_size=self.group_size,
            desc_act=self.desc_act,
            sym=self.sym,
            format='gptq',
        )
        
    def _setup_paths(self):
        """Handle path configurations"""
        if not self.quantize_model:
            self.quantize_model = f"{self.model}-W{self.wbits}A16-gptq"

    def _setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout,
        )

    def get_wikitext2(self, tokenizer, nsamples=256, seqlen=1024):
        """Prepare wikitext2 dataset"""
        traindata = load_dataset(
            "/data/disk0/Dataset/wikitext",
            "wikitext-2-raw-v1",
            split="train"
        ).filter(lambda x: len(x["text"]) >= seqlen)

        return [tokenizer(example["text"]) for example in traindata.select(range(nsamples))]

    def gptqmodel(self):
        """Quantize the model"""
        logging.info(f"Loading and quantizing model: {self.model}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True, trust_remote_code=True)
        traindataset = self.get_wikitext2(tokenizer)
        
        model = GPTQModel.load(
            self.model, 
            self.quantize_config, 
            trust_remote_code=True
        ).to(self.device)
        
        model.quantize(traindataset, tokenizer=tokenizer, backend=BACKEND.TORCH)
        model.save(self.quantize_model)
        tokenizer.save_pretrained(self.quantize_model)
        
        return model, tokenizer

    def eval_model(self, model):
        """Evaluate the model"""
        logging.info("Running evaluation...")
        
        if self.eval_framework == 'lmeval':
            # 这里示例任务写死成 ARC_CHALLENGE，可在此自定义需要评测的任务
            tasks = [getattr(EVAL.LM_EVAL, t.strip()) for t in self.eval_tasks.split(",")]
            GPTQModel.eval(
                model,
                framework=EVAL.LM_EVAL,
                tasks=tasks,
                trust_remote_code=True,
                output_file=self.output_file
            )
        elif self.eval_framework == 'evalplus':
            tasks = [getattr(EVAL.EVALPLUS, t.strip()) for t in self.eval_tasks.split(",")]
            evaluate(
                dataset=tasks,
                model=self.quantize_model,
                output_file=self.output_file
            )
            
        logging.info(f"Evaluation results saved to {self.output_file}")

    def compile_model(self, model):
        """Compile the model"""
        logging.info("Compiling the model...")
        compile(self.quantize_model)

    def run(self):
        """Main execution flow"""
        logging.info(f"Starting process with arguments: {self.__dict__}")
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        
        # Quantization
        quantized_model, quant_tokenizer = self.gptqmodel()
        
        # Evaluation
        if self.eval:
            self.eval_model(self.quantized_model)
            
        # Compilation
        if self.compile:
            self.compile_model()

    @classmethod
    def from_args(cls):
        """Create instance from command line arguments"""
        # This method would contain the original argparse logic
        # Simplified for demonstration purposes
        return cls(
            model="path/to/model",
            wbits=4,
            # ... other parameters from command line
        )

if __name__ == "__main__":
    quantizer = GPTQQuantizer.from_args()
    quantizer.run()