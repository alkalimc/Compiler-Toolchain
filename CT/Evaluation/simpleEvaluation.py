# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
import torch
import subprocess
from typing import Optional, Union

@dataclass
class EvalPlus():
    username: str = field(default="Compiler-Toolchain")
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
        workspace: str = os.path.join("/data/disk0/Workspace", self.username)
        model_path: str = os.path.join(workspace, "Models", self.model_id)
        evaluation_path: str = os.path.join(workspace, "Evaluations", self.model_id, self.evaluation_framework)
        evaluation_output_path: str = os.path.join(evaluation_path, f"{self.evaluation_task}.log")
