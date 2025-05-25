# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
import subprocess

@dataclass
class LmEvaluationHarness():
    username: str = field(default="Compiler-Toolchain")
    model_id: str = field(default="Qwen2.5-7B-Instruct")

    evaluation_framework: str = field(default="lm-evaluation-harness")
    evaluation_task: str = field(default="arc_easy", metadata={"choices": [
        "arc_easy",
        "arc_challenge",
        "gsm8k_cot",
        "gsm8k_platinum_cot",
        "hellaswag",
        "mmlu",
        "gpqa",
        "boolq",
        "openbookqa"
        ]})
    evaluation_device: str = field(default="cuda:0")
    evaluation_batch_size: int = field(default=4, metadata={"min_value": 1})

    trust_remote_code: bool = field(default=True)

    def __post_init__(self):
        workspace: str = os.path.join("/data/disk0/Workspace", self.username)
        model_path: str = os.path.join(workspace, "Models", self.model_id)
        evaluation_path: str = os.path.join(workspace, "Evaluations", self.model_id, self.evaluation_framework)
        evaluation_output_path: str = os.path.join(evaluation_path, f"{self.evaluation_task}.log")

        if not os.path.exists(evaluation_path):
            os.makedirs(evaluation_path)

        process = subprocess.run(
            f"lm-eval "
            f"--model hf "
            f"--model_args pretrained='{model_path}' "
            f"--tasks {self.evaluation_task} "
            f"--device {self.evaluation_device} "
            f"--batch_size {self.evaluation_batch_size} "
            f"--trust_remote_code "
            f"--output_path {evaluation_output_path} 2>&1",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        for line in process.stdout.splitlines():
            print(line)