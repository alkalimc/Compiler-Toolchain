# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
import logging
import numpy as np
import torch
from typing import List

@dataclass
class BlockDataGenerator:
    # 原始参数对应字段
    base_model_path: str = field(
        metadata={"help": "Base path to the compiled model files"}
    )
    output_root: str = field(
        default="QWEN_BLOCK_write_data",
        metadata={"help": "Root directory for output files"}
    )
    port_num: int = field(
        default=32,
        metadata={"help": "Number of HBM ports"}
    )
    layer_num: int = field(
        default=28,
        metadata={"help": "Number of model layers"}
    )

    def __post_init__(self):
        self._setup_directories()
        self._verify_paths()

    def _setup_directories(self):
        """创建基础目录结构"""
        os.makedirs(self.output_root, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )

    def _verify_paths(self):
        """验证基础路径有效性"""
        if not os.path.exists(self.base_model_path):
            raise ValueError(f"Base model path not found: {self.base_model_path}")

    # 保留原始工具函数
    def read_bin(self, bin_name, _dtype=np.float16):
        """保持原始read_bin实现"""
        with open(bin_name, 'rb') as f:
            return np.frombuffer(f.read(), dtype=_dtype)

    def out_bin(self, data_pt, bin_name):
        """保持原始out_bin实现"""
        if data_pt.dtype == torch.float32:
            data_pt = data_pt.to(torch.float16)
        with open(bin_name, 'wb') as f:
            f.write(data_pt.detach().cpu().numpy().flatten().tobytes())

    # 保留核心功能函数
    def gen_block_para(self, layer: int, step: str, chin: int, chout: int, dir: str, file: str):
        """保持原始gen_block_para实现"""
        # 原始实现逻辑
        q_weight_bin = read_bin(f"/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq/compile/qwen2_qweight_bin/model.layers.{layer}.{step}.qweight.bin",_dtype=np.int32)
        q_weight_bin = torch.tensor(q_weight_bin,dtype=torch.int32)
        q_weight_bin = q_weight_bin & 0xF
        q_weight     = q_weight_bin.reshape(len(q_weight_bin)//8,8)
        for col in range(8):
            q_weight[:,col] = q_weight[:,col] * 2**(4*col)
        q_weight = torch.sum(q_weight,dim=1)
        q_weight = q_weight.reshape(chout,chin//8)
        print(q_weight,q_weight.shape)

        q_scale_bin  = read_bin(f"/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq/compile/qwen2_qweight_bin/model.layers.{layer}.{step}.scales.bin",_dtype=np.int16)
        q_scale_bin  = torch.tensor(q_scale_bin,dtype=torch.int32)
        print(f"checkpos1 {layer} {step} q_scale: {q_scale_bin} {q_scale_bin.shape}")
        q_scale_bin  = q_scale_bin & 0xFFFF
        print(f"checkpos2 {layer} {step} q_scale: {q_scale_bin} {q_scale_bin.shape}")
        q_scale      = q_scale_bin.reshape(len(q_scale_bin)//2,2)
        print(f"checkpos3 {layer} {step} q_scale: {q_scale_bin} {q_scale_bin.shape}")
        for col in range(2):
            q_scale[:,col] = q_scale[:,col] * 2**(16*col)
        print(f"checkpos4 {layer} {step} q_scale: {q_scale_bin} {q_scale_bin.shape}")
        q_scale      = torch.sum(q_scale,dim=1)
        print(f"{layer} {step} q_scale: {q_scale} {q_scale.shape}")
        q_scale      = q_scale.reshape(chout,chin//2//128)
        q_scale_add0 = torch.zeros((chout,(chin//2//128+8-1)//8*8),dtype=torch.int32)
        q_scale_add0[:,:chin//128//2] = q_scale
        
        scale_weight = torch.tensor([],dtype=torch.int32)
        for i in range(0,q_scale_add0.shape[1]//8):
            scale_weight = torch.cat((scale_weight,q_scale_add0[:,i*8:(i+1)*8]),dim=1)
            if((i+1)*256>q_weight.shape[1]):
                scale_weight = torch.cat((scale_weight,q_weight[:,i*256:]),dim=1)
            else:
                scale_weight = torch.cat((scale_weight,q_weight[:,i*256:(i+1)*256]),dim=1)        
        scale_weight = torch.tensor(scale_weight,dtype=torch.int32)
        print(scale_weight,scale_weight.dtype)
        
            

        # 输出路径调整
        output_dir = f"{self.output_root}_port{self.port_num}/BLOCK{str(layer).zfill(2)}/{dir}"
        os.makedirs(output_dir, exist_ok=True)
        for port in range(self.port_num):
            self.out_bin(
                scale_weight[port::self.port_num,:],
                f"{output_dir}/{file}_HBM_DDR_{str(port).zfill(2)}.bin"
            )

    def gen_outlayer_para(self, chin: int, chout: int, dir: str, file: str):
        """保持原始gen_outlayer_para实现"""
        # 原始实现逻辑
        q_weight_bin = torch.tensor(
            self.read_bin(f"{self.base_model_path}/qwen2_qweight_bin/lm_head_qweight.bin", np.int32),
            dtype=torch.int32
        )
        q_weight_bin = torch.tensor(q_weight_bin,dtype=torch.int32)
        q_weight_bin = q_weight_bin & 0xF
        q_weight     = q_weight_bin.reshape(len(q_weight_bin)//8,8)
        for col in range(8):
            q_weight[:,col] = q_weight[:,col] * 2**(4*col)
        q_weight = torch.sum(q_weight,dim=1)
        q_weight = q_weight.reshape(chout,chin//8)
        print(q_weight,q_weight.shape)

        q_scale_bin  = read_bin(f"/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq/compile/qwen2_qweight_bin/lm_head_scales.bin",_dtype=np.int16)
        q_scale_bin  = torch.tensor(q_scale_bin,dtype=torch.int32)
        q_scale_bin  = q_scale_bin & 0xFFFF
        q_scale      = q_scale_bin.reshape(len(q_scale_bin)//2,2)
        for col in range(2):
            q_scale[:,col] = q_scale[:,col] * 2**(16*col)
        q_scale      = torch.sum(q_scale,dim=1)
        q_scale      = q_scale.reshape(chout,chin//2//128)
        q_scale_add0 = torch.zeros((chout,(chin//2//128+8-1)//8*8),dtype=torch.int32)
        q_scale_add0[:,:chin//128//2] = q_scale
        
        scale_weight = torch.tensor([],dtype=torch.int32)
        for i in range(0,q_scale_add0.shape[1]//8):
            scale_weight = torch.cat((scale_weight,q_scale_add0[:,i*8:(i+1)*8]),dim=1)
            if((i+1)*256>q_weight.shape[1]):
                scale_weight = torch.cat((scale_weight,q_weight[:,i*256:]),dim=1)
            else:
                scale_weight = torch.cat((scale_weight,q_weight[:,i*256:(i+1)*256]),dim=1)        
        scale_weight = torch.tensor(scale_weight,dtype=torch.int32)
        print(scale_weight,scale_weight.dtype)

        # 输出路径调整
        output_dir = f"{self.output_root}_port{self.port_num}/OutLayer/{dir}"
        os.makedirs(output_dir, exist_ok=True)
        for port in range(self.port_num):
            self.out_bin(
                scale_weight[port::self.port_num,:],
                f"{output_dir}/{file}_HBM_DDR_{str(port).zfill(2)}.bin"
            )

    # 保持其他原始函数实现
    def gen_block_ln(self, layer: int, step: str, dir: str, file: str):
        """保持原始gen_block_ln实现"""
        source = f"{self.base_model_path}/qwen2_fp16_weight_bin/model.layers.{layer}.{step}.weight.bin"
        target = f"{self.output_root}_port{self.port_num}/BLOCK{str(layer).zfill(2)}/{dir}/{file}_wt_in_DDR.bin"
        os.system(f"cp -rf {source} {target}")

    def gen_outlayer_ln(self, dir: str, file: str):
        """保持原始gen_outlayer_ln实现"""
        source = f"{self.base_model_path}/qwen2_fp16_weight_bin/model.norm.weight.bin"
        target = f"{self.output_root}_port{self.port_num}/OutLayer/{dir}/{file}_wt_in_DDR.bin"
        os.system(f"cp -rf {source} {target}")

    def gen_block_bias(self, layer: int, step: str, dir: str, file: str):
        """保持原始gen_block_bias实现"""
        bias_bin = read_bin(f"/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq/compile/qwen2_qweight_bin/model.layers.{layer}.{step}.bias.bin",_dtype=np.int16)
        bias_bin  = torch.tensor(bias_bin,dtype=torch.int32)
        bias_bin  = bias_bin & 0xFFFF
        bias_bin  = bias_bin | 0x3C000000
        self.out_bin(bias_bin, f"{self.output_root}_port{self.port_num}/BLOCK{str(layer).zfill(2)}/{dir}/{file}_wt_and_bias_in_DDR.bin")

    def gen_outlayer_bias(self, chout: int, dir: str, file: str):
        """保持原始gen_outlayer_bias实现"""
        bias_bin  = torch.zeros(chout,dtype=torch.int32)
        bias_bin  = bias_bin | 0x3C000000
        self.out_bin(bias_bin, f"{self.output_root}_port{self.port_num}/OutLayer/{dir}/{file}_wt_and_bias_in_DDR.bin")

    def generate_all_blocks(self):
        """保持原始主生成逻辑"""
        for layer in range(self.layer_num):
            # 创建目录结构
            block_dir = f"{self.output_root}_port{self.port_num}/BLOCK{str(layer).zfill(2)}"
            os.makedirs(f"{block_dir}/LN_DDR_bin", exist_ok=True)
            os.makedirs(f"{block_dir}/MVM_BN_DDR_bin", exist_ok=True)
            os.makedirs(f"{block_dir}/MVM_BN_RES_DDR_bin", exist_ok=True)
            os.makedirs(f"{block_dir}/MVM_BN_RES_write_to_HBM_bin", exist_ok=True)
            os.makedirs(f"{block_dir}/MVM_BN_write_to_HBM_bin", exist_ok=True)
            
            # 生成各组件
            self.gen_block_para(layer, "self_attn.q_proj", 3584, 3584, "MVM_BN_write_to_HBM_bin", "MVMBN0_q")
            self.gen_block_para(layer,"self_attn.k_proj",3584,512 ,"MVM_BN_write_to_HBM_bin","MVMBN0_k")
            self.gen_block_para(layer,"self_attn.v_proj",3584,512 ,"MVM_BN_write_to_HBM_bin","MVMBN0_v")
            self.gen_block_para(layer,"mlp.gate_proj"   ,3584,18944,"MVM_BN_write_to_HBM_bin","MVMBN1")
            
            self.gen_block_para(layer,"self_attn.o_proj",3584,3584 ,"MVM_BN_RES_write_to_HBM_bin","MVMBNRES0")
            self.gen_block_para(layer,"mlp.up_proj"     ,3584,18944 ,"MVM_BN_RES_write_to_HBM_bin","MVMBNRES1")
            self.gen_block_para(layer,"mlp.down_proj"   ,18944,3584 ,"MVM_BN_RES_write_to_HBM_bin","MVMBNRES2")

            ##GEN BIAS
            self.gen_block_bias(layer,"self_attn.q_proj","MVM_BN_DDR_bin","MVMBN0_q")
            self.gen_block_bias(layer,"self_attn.k_proj","MVM_BN_DDR_bin","MVMBN0_k")
            self.gen_block_bias(layer,"self_attn.v_proj","MVM_BN_DDR_bin","MVMBN0_v")
            self.gen_block_bias(layer,"mlp.gate_proj"   ,"MVM_BN_DDR_bin","MVMBN1")
            
            self.gen_block_bias(layer,"self_attn.o_proj","MVM_BN_RES_DDR_bin","MVMBNRES0")
            self.gen_block_bias(layer,"mlp.up_proj"     ,"MVM_BN_RES_DDR_bin","MVMBNRES1")
            self.gen_block_bias(layer,"mlp.down_proj"   ,"MVM_BN_RES_DDR_bin","MVMBNRES2")   
            
            ##GEN LN
            self.gen_block_ln(layer,"input_layernorm","LN_DDR_bin","LN0")
            self.gen_block_ln(layer,"post_attention_layernorm","LN_DDR_bin","LN1")

    def generate_output_layer(self):
        """保持原始输出层生成逻辑"""
        os.makedirs(f"{self.output_root}_port{self.port_num}/OutLayer", exist_ok=True)
        self.gen_outlayer_para(3584, 152064, "MVM_BN_write_to_HBM_bin", "MVMBN_Argmax")
        self.gen_outlayer_bias(152064, "MVM_BN_DDR_bin", "MVMBN_Argmax")
        self.gen_outlayer_ln("LN_DDR_bin", "LN")

    @classmethod
    def from_args(cls):
        """参数解析适配器"""
        parser = argparse.ArgumentParser(description="Block Data Generator")
        parser.add_argument("--base_model_path", required=True)
        parser.add_argument("--output_root", default="QWEN_BLOCK_write_data")
        parser.add_argument("--port_num", type=int, default=32)
        parser.add_argument("--layer_num", type=int, default=28)
        args = parser.parse_args()
        return cls(**vars(args))

if __name__ == "__main__":
    generator = BlockDataGenerator.from_args()
    generator.generate_all_blocks()
    generator.generate_output_layer()