# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
import torch
import subprocess
from typing import Optional, Union
from CT.Evaluation.lmEvaluationHarness import LmEvaluationHarness
from CT.Evaluation.evalPlus import EvalPlus

@dataclass
class SimpleEvaluation():
    model_id: str = field(default="Qwen2.5-7B-Instruct")

    evaluation_framework: str = field(default="lm-evaluation-harness", metadata={"choices": [
        "lm-evaluation-harness",
        "EvalPlus"
        ]})
    evaluation_task: str = field(default="arc_easy", metadata={"choices": [
        "arc_easy",
        "arc_challenge",
        "gsm8k_cot",
        "gsm8k_platinum_cot",
        "hellaswag",
        "mmlu",
        "gpqa",
        "boolq",
        "openbookqa",
        "humaneval",
        "mbpp"
        ]})
    evaluation_device: Optional[Union[str, torch.device]] = field(default=torch.device("cuda:0"))
    torch.device("cuda:0")
    evaluation_batch_size: int = field(default=4, metadata={"min_value": 1})

    def __post_init__(self):
        if self.evaluation_framework == "lm-evaluation-harness":
            LmEvaluationHarness(
                model_id=self.model_id,
                evaluation_task=self.evaluation_task,
                evaluation_device=self.evaluation_device,
                evaluation_batch_size=self.evaluation_batch_size
            )
        elif self.evaluation_framework == "EvalPlus":
            EvalPlus(
                model_id=self.model_id,
                evaluation_task=self.evaluation_task,
                evaluation_device=self.evaluation_device,
                evaluation_batch_size=self.evaluation_batch_size
            )