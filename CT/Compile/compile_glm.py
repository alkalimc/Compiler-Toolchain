# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
import logging
import time
import torch
import torch.nn as nn
from gptqmodel import GPTQModel
from gptqmodel.utils.backend import BACKEND
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

@dataclass
class GlmCompiler:
    # 原始参数对应字段
    model_name: str = field(
        metadata={"help": "Path to the quantized model"}
    )
    quantize_model: str = field(
        default="",
        metadata={"help": "Output directory for compiled model"}
    )
    wbits: int = field(
        default=4,
        metadata={"help": "Quantization bits", "choices": [4, 8]}
    )
    group_size: int = field(
        default=128,
        metadata={"help": "Group size for quantization"}
    )

    def __post_init__(self):
        self._setup()
        self.model = get_llm(self.model_name)
        self.layers = self.model.transformer.encoder.layers

    def _setup(self):
        """初始化配置"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )
        os.makedirs(self.quantize_model, exist_ok=True)

    # 保留原始函数作为类方法
    def get_llm(self, model_name, seqlen=2048):
        return GPTQModel.load(
            model_name,
            trust_remote_code=True,
            backend=BACKEND.TORCH
        )

    def find_layers(self, module, layers=[nn.Conv2d, nn.Linear, TorchQuantLinear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(self.find_layers(
                child, layers=layers, 
                name=f"{name}.{name1}" if name else name1
            ))
        return res

    def dequantize(self, sublayer):
        if isinstance(sublayer, TorchQuantLinear):    
            identity = torch.eye(
                sublayer.infeatures,
                device=sublayer.qweight.device,
                dtype=sublayer.scales.dtype
            )
            
            with torch.no_grad():
                output = sublayer(identity.to(sublayer.scales.dtype))
            
            weight = output.T[:sublayer.outfeatures, :sublayer.infeatures]
            max_values = weight.view(weight.shape[0], -1, self.group_size).abs().max(dim=2).values
            scales = max_values / (2 ** (self.wbits - 1) - 1)
            return weight, scales
        
        if isinstance(sublayer, nn.Linear):
            weight = sublayer.weight.data
            max_values = weight.view(weight.shape[0], -1, self.group_size).abs().max(dim=2).values
            scales = max_values / (2 ** (self.wbits - 1) - 1)
            return weight, scales

    def out_bin(self, data_pt, bin_name):
        if data_pt.dtype == torch.float32:
            data_pt = data_pt.to(torch.float16)
        
        os.makedirs(os.path.dirname(bin_name), exist_ok=True)
        with open(bin_name, 'wb') as f:
            f.write(data_pt.detach().cpu().numpy().flatten().tobytes())

    # 保留原始compile函数结构
    def compile(self):
        start_time = time.time() 

        for i in range(len(self.layers)):
            layer = self.layers[i]
            subset = self.find_layers(layer)
            logging.info(f"Compiling layer {i}")
            
            # 保持原始处理逻辑
            block_dir = os.path.join(self.quantize_model, f"BLOCK{str(i).zfill(2)}")
            
            # 原始QKV处理
            self._process_qkv(i, subset, block_dir)
            # 原始dense处理
            self._process_dense(i, subset, block_dir)

        # 输出层处理
        self._process_output_layer()
        
        logging.info(f"Total time: {time.time()-start_time:.2f}s")

    def _process_qkv(self, layer_idx, subset, block_dir):
        """保持原始QKV处理逻辑"""
        query_key_value = subset['self_attention.query_key_value']
        Wt, scales = self.dequantize(query_key_value)
        bias = query_key_value.bias.detach()
        
        # 保持原始拆分逻辑
        MVM_BN_Wq = Wt[:4096, :]
        MVM_BN_Wk = Wt[4096:4096+256, :]
        MVM_BN_Wv = Wt[4096+256:, :]
        
        # 保持原始保存逻辑
        step_dir = os.path.join(block_dir, "step1_6")
        os.makedirs(step_dir, exist_ok=True)
        self.out_bin(MVM_BN_Wq.int(), os.path.join(step_dir, 'MVM_BN_Wq.bin'))
        self.out_bin(MVM_BN_Wk.int(), os.path.join(step_dir, 'MVM_BN_Wk.bin'))
        self.out_bin(MVM_BN_Wv.int(), os.path.join(step_dir, 'MVM_BN_Wv.bin'))

    def _process_dense(self, layer_idx, subset, block_dir):
        """保持原始dense处理逻辑"""
        # 原始dense处理代码
        dense = subset['self_attention.dense']
        Wt, scales = self.dequantize(dense)
        step_dir = os.path.join(block_dir, "step7_12")
        os.makedirs(step_dir, exist_ok=True)
        self.out_bin(Wt.int(), os.path.join(step_dir, 'MVM_BN_RES_weight.bin'))

    def _process_output_layer(self):
        """保持原始输出层处理逻辑"""
        output_layer = self.model.transformer.output_layer
        Wt, scales = self.dequantize(output_layer)
        out_dir = os.path.join(self.quantize_model, "output_layer")
        os.makedirs(out_dir, exist_ok=True)
        self.out_bin(Wt.int(), os.path.join(out_dir, 'MVM_weight.bin'))

    @classmethod
    def from_args(cls):
        """参数解析适配器"""
        parser = argparse.ArgumentParser(description="Model Compiler")
        parser.add_argument("--model_name", required=True)
        parser.add_argument("--quantize_model", default="")
        parser.add_argument("--wbits", type=int, default=4)
        parser.add_argument("--group_size", type=int, default=128)
        args = parser.parse_args()
        return cls(**vars(args))

if __name__ == "__main__":
    compiler = GlmCompiler.from_args()
    compiler.compile()