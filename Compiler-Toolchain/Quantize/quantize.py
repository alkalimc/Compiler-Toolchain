# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import field
import os
from typing import Optional, Union
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

class quantize():
    workspace: str = field(default=f"/data/disk0/Workspace/{os.getlogin()}")

    model_id: str = field(default="Qwen2.5-7B-Instruct")
    model_path: str = field(default=f"{workspace}/Models/{model_id}")

    data_id: str = field(default="allenai-c4")
    data_path: str = field(default=f"{workspace}/Dataset/{model_id}")
    data_files: str = field(default="en/c4-train.00001-of-01024.json.gz")
    data_split: str = field(default="train")
    data_range: int = field(default=1024)
    data_tag: str = field(default="text")

    quantize_bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8]})
    quantize_group_size: int = field(default=128)
    quantize_desc_act: bool = field(default=False)
    quantize_sym: bool = field(default=True)
    quantize_v2: bool = field(default=True)
    quantize_path: str = field(default=f"{workspace}/Models/Quanted/{model_id}-W{quantize_bits}A16-gptq")
    quantize_batch_size: int = field(default=16, metadata={"min_value": 1})

    quantize_config: QuantizeConfig = field(default=QuantizeConfig(quantize_bits, quantize_group_size, quantize_desc_act, quantize_sym, quantize_v2))

    data_calibration_dataset: load_dataset = field(default=load_dataset(
        data_path,
        data_files,
        data_split
      ).select(range(data_range))[data_tag])
    
    model: GPTQModel.load = field(default=GPTQModel.load(model_path, quantize_config))
    
    model.quantize(data_calibration_dataset, quantize_batch_size)
    model.save(quantize_path)