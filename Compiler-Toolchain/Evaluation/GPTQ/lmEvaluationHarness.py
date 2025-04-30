# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL

@dataclass
class EvalPlus():
    workspace: str = field(default=f"/data/disk0/Workspace/{os.getlogin()}")

    model_id: str = field(default="Qwen2.5-7B-Instruct")
    model_path: str = field(default=f"{workspace}/Models/{model_id}")

    evaluation_path: str = field(default=f"{workspace}/Evaluations/{model_id}")
    evaluation_framework = EVAL.LM_EVAL
    evaluation_tasks: EVAL.LM_EVAL = field(default=EVAL.LM_EVAL.ARC_EASY)
    evaluation_batch_size: int = field(default=1, metadata={"min_value": 1})
    evaluation_output_path: str = field(default=f"{evaluation_path}/{evaluation_framework.name}/{model_id}{evaluation_framework.name}{evaluation_tasks.name}.json")

    trust_remote_code: bool = field(default=True)

    def __post_init__(self):
        evalplus_results = GPTQModel.eval(
            model_or_id_or_path=self.model_path,
            framework=self.evaluation_framework,
            tasks=self.evaluation_tasks,
            batch_size=self.evaluation_batch_size,
            trust_remote_code=self.trust_remote_code,
            output_path=self.evaluation_output_path
            )