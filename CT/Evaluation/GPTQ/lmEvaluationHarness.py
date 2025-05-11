# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL

@dataclass
class LmEvaluationHarness():
    username: str = field(default="Compiler-Toolchain")
    model_id: str = field(default="Qwen2.5-7B-Instruct-W4A16-GPTQ")
    evaluation_batch_size: int = field(default=1)

    evaluation_path: str = field(init=False)
    evaluation_framework: EVAL = field(init=False, default=EVAL.LM_EVAL)
    evaluation_task: EVAL.LM_EVAL = field(default=EVAL.LM_EVAL.ARC_CHALLENGE)
    evaluation_batch_size: int = field(default=4, metadata={"min_value": 1})
    evaluation_output_path: str = field(init=False)

    trust_remote_code: bool = field(default=True)

    def __post_init__(self):
        workspace: str = os.path.join("/data/disk0/Workspace", self.username)
        model_path: str = os.path.join(workspace, "Models", "Quanted", self.model_id)
        evaluation_path: str = os.path.join(workspace, "Evaluations", "Quanted", self.model_id, self.evaluation_framework.__name__)
        evaluation_id: str = os.path.join(evaluation_path, f"{self.evaluation_task.name}.json")

        if not os.path.exists(evaluation_path):
            os.makedirs(evaluation_path)

        evalplus_result = GPTQModel.eval(
            model_or_id_or_path=model_path,
            framework=self.evaluation_framework,
            tasks=self.evaluation_task,
            batch_size=self.evaluation_batch_size,
            trust_remote_code=self.trust_remote_code,
            output_path=evaluation_id
            )