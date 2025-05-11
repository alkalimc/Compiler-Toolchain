# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
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
    quantize_batch_size: int = field(default=16, metadata={"min_value": 1})
    quantize_device: Optional[Union[str, torch.device]] = field(default=None)

    def __post_init__(self):
        if self.model_type == "Qwen":
            quantification = Quantification(
                model_id=self.model_id,
                quantize_batch_size=self.quantize_batch_size,
                quantize_device=self.quantize_device
                )
        elif self.model_type == "Qwen_VL":
            qwenVLQuantification = QwenVLQuantification(
                model_id=self.model_id,
                quantize_batch_size=self.quantize_batch_size,
                quantize_device=self.quantize_device
                )