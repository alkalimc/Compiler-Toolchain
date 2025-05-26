# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
import subprocess

@dataclass
class EvalPlus():
    username: str = field(default="Compiler-Toolchain")
    model_id: str = field(default="Qwen2.5-7B-Instruct")

    evaluation_framework: str = field(default="EvalPlus")
    evaluation_task: str = field(default="humaneval", metadata={"choices": [
        "humaneval",
        "mbpp"
        ]})
    evaluation_device: str = field(default="cuda:0")
    evaluation_batch_size: int = field(default=4, metadata={"min_value": 1})

    trust_remote_code: bool = field(default=True)

    def __post_init__(self):
        workspace: str = os.path.join("/data/disk0/Workspace", self.username)
        model_path: str = os.path.join(workspace, "Models", self.model_id)
        evaluation_path: str = os.path.join(workspace, "Evaluations", self.model_id, self.evaluation_framework)
        evaluation_output_path: str = os.path.join(evaluation_path, f"{self.evaluation_task}.json")

        if not os.path.exists(evaluation_path):
            os.makedirs(evaluation_path)

        process = subprocess.Popen(
            f"evalplus.evaluate "
            f"--model \"{model_path}\" "
            f"--dataset {self.evaluation_task}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        for line in process.stdout:
            print(line)