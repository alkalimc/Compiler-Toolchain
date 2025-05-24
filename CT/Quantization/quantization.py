# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
import torch
from typing import Optional, Union
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

@dataclass
class Quantization():
    username: str = field(default="Compiler-Toolchain")

    model_id: str = field(default="Qwen2.5-7B-Instruct")

    data_id: str = field(default="allenai-c4")
    data_file: str = field(default=os.path.join("en", "c4-train.00001-of-01024.json.gz"))
    data_split: str = field(default="train")
    data_range: int = field(default=1024)
    data_tag: str = field(default="text")

    quantization_bits: int = field(default=4, metadata={"choices": [
        2,
        3,
        4,
        8
        ]})
    quantization_group_size: int = field(default=128)
    quantization_desc_act: bool = field(default=False)
    quantization_sym: bool = field(default=True)
    quantization_batch_size: int = field(default=16, metadata={"min_value": 1})
    quantization_device: Optional[Union[str, torch.device]] = field(default=torch.device("cuda:0"))

    trust_remote_code: bool = field(default=True)

    def __post_init__(self):
        workspace: str = os.path.join("/data/disk0/Workspace", self.username)
        model_path: str = os.path.join(workspace, "Models", self.model_id)
        data_path: str = os.path.join(workspace, "Datasets", self.data_id)
        quantization_path: str = os.path.join(workspace, "Models", "Quanted", f"{self.model_id}-W{self.quantization_bits}A16-gptq")

        quantizationConfig = QuantizeConfig(
            bits=self.quantization_bits,
            group_size=self.quantization_group_size,
            desc_act=self.quantization_desc_act,
            sym=self.quantization_sym
            )

        data_calibration_dataset = load_dataset(
            path=data_path,
            data_files=self.data_file,
            split=self.data_split,
            trust_remote_code=self.trust_remote_code
          ).select(range(self.data_range))[self.data_tag]
    
        model = GPTQModel.load(
            model_id_or_path=model_path,
            quantize_config=quantizationConfig,
            trust_remote_code=self.trust_remote_code,
            device=self.quantization_device
            )
        model.quantize(
            calibration_dataset=data_calibration_dataset,
            batch_size=self.quantization_batch_size
            )
        model.save(save_dir=quantization_path)