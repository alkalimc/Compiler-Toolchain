# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import quantize

@dataclass
class simpleQuantize():
    model_id: str = field(default="Qwen2.5-7B-Instruct")
    quantize_v2: bool = field(default=True)
    quantize_batch_size: int = field(default=16, metadata={"min_value": 1})

    def __post_init__(self):
        Quantize = quantize(self.model_id, self.quantize_v2, self.quantize_batch_size)