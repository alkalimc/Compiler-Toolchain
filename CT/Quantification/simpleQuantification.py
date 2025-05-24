# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
import torch
from typing import Optional, Union
from CT.Quantification.quantification import Quantification
from CT.Quantification.qwenVLQuantification import QwenVLQuantification

@dataclass
class SimpleQuantification():
    model_type: str = field(default="Qwen", metadata={"choices": [
        "Qwen",
        "Qwen_VL"
        ]})
    model_id: str = field(default="Qwen2.5-7B-Instruct")
    data_id: str = field(default="allenai-c4")
    data_file: str = field(default=os.path.join("en", "c4-train.00001-of-01024.json.gz"))
    quantize_batch_size: int = field(default=16, metadata={"min_value": 1})
    quantize_device: Optional[Union[str, torch.device]] = field(default=torch.device("cuda:0"))

    def __post_init__(self):
        if self.model_type == "Qwen":
            quantification = Quantification(
                model_id=self.model_id,
                data_id=self.data_id,
                data_file=self.data_file,
                quantize_batch_size=self.quantize_batch_size,
                quantize_device=self.quantize_device
                )
        elif self.model_type == "Qwen_VL":
            qwenVLQuantification = QwenVLQuantification(
                model_id=self.model_id,
                data_id=self.data_id,
                data_file=self.data_file,
                quantize_batch_size=self.quantize_batch_size,
                quantize_device=self.quantize_device
                )