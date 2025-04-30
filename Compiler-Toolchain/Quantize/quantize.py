# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

@dataclass
class Quantize():
    workspace: str = field(default=f"/data/disk0/Workspace/{os.getlogin()}")

    model_id: str = field(default="Qwen2.5-7B-Instruct")
    model_path: str = field(default=f"{workspace}/Models/{model_id}")

    data_id: str = field(default="allenai-c4")
    data_path: str = field(default=f"{workspace}/Datasets/{model_id}")
    data_files: str = field(default="en/c4-train.00001-of-01024.json.gz")
    data_split: str = field(default="train")
    data_range: int = field(default=1024)
    data_tag: str = field(default="text")

    quantize_bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8]})
    quantize_group_size: int = field(default=128)
    quantize_desc_act: bool = field(default=False)
    quantize_sym: bool = field(default=True)
    quantize_path: str = field(default=f"{workspace}/Models/Quanted/{model_id}-W{quantize_bits}A16-gptq")
    quantize_batch_size: int = field(default=16, metadata={"min_value": 1})

    trust_remote_code: bool = field(default=True)

    def __post_init__(self):
        quantizeConfig = QuantizeConfig(
            bits=self.quantize_bits,
            group_size=self.quantize_group_size,
            desc_act=self.quantize_desc_act,
            sym=self.quantize_sym
            )

        data_calibration_dataset = load_dataset(
            data_dir=self.data_path,
            data_files=self.data_files,
            split=self.data_split,
            trust_remote_code=self.trust_remote_code
          ).select(range(self.data_range))[self.data_tag]
    
        model = GPTQModel.load(
            model_id_or_path=self.model_path,
            quantize_config=quantizeConfig,
            trust_remote_code=self.trust_remote_code
            )
        model.quantize(
            calibration_dataset=data_calibration_dataset,
            batch_size=self.quantize_batch_size
            )
        model.save(save_dir=self.quantize_path)