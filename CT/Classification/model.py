# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field

@dataclass
class ModelClassification():
    model_id: str = field(default="Qwen2.5-7B-Instruct")
    model_type: str = "FP16"
    model_classification: str = "Qwen"

    def __post_init__(self):
        if 'gptq' in self.model_id:
            self.model_type = "GPTQ"
        if 'Qwen' in self.model_id:
            if 'VL' in self.model_id:
                self.model_classification = "QwenVL"
            else:
                self.model_classification = "Qwen"
        elif 'chatglm' in self.model_id:
            self.model_classification = "GLM"

    def modelType(self) -> str:
        return self.model_type
    def modelClassification(self) -> str:
        return self.model_classification