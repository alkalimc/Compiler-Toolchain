# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
from quantize import Quantize

@dataclass
class SimpleQuantize():
    model_id: str = field(default="Qwen2.5-7B-Instruct")
    quantize_batch_size: int = field(default=16, metadata={"min_value": 1})

    def __post_init__(self):
        quantize = Quantize(
            model_id=self.model_id,
            quantize_batch_size=self.quantize_batch_size
            )