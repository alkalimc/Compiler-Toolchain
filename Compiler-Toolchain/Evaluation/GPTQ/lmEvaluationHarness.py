# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL

###
from gptqmodel import QuantizeConfig

@dataclass
class LmEvaluationHarness():
    '''更换数据生成逻辑，一部分变量放到postinit动态生成
    username: str = field(default="Compiler-Toolchain")
    workspace: str = field(default=f"/data/disk0/Workspace/{username}")

    model_id: str = field(default="Qwen2.5-7B-Instruct")
    model_path: str = field(default=f"{workspace}/Models/{model_id}")

    evaluation_path: str = field(default=f"{workspace}/Evaluations/Quanted/{model_id}")
    evaluation_framework = EVAL.LM_EVAL
    evaluation_tasks: EVAL.LM_EVAL = field(default=EVAL.LM_EVAL.ARC_EASY)
    evaluation_batch_size: int = field(default=1, metadata={"min_value": 1})
    evaluation_output_path: str = field(default=f"{evaluation_path}/{evaluation_framework.name}/{model_id}{evaluation_framework.name}{evaluation_tasks.name}.json")

    trust_remote_code: bool = field(default=True)
    '''
    ###
    username: str = field(default="Compiler-Toolchain")
    model_id: str = field(default="Qwen2.5-7B-Instruct")
    evaluation_tasks: EVAL.LM_EVAL = field(default=EVAL.LM_EVAL.ARC_CHALLENGE)
    evaluation_batch_size: int = field(default=1)
    trust_remote_code: bool = field(default=True)

    workspace: str = field(init=False)
    model_path: str = field(init=False)
    evaluation_path: str = field(init=False)
    evaluation_output_path: str = field(init=False)
    evaluation_framework: EVAL = field(init=False, default=EVAL.LM_EVAL)

    
    def __post_init__(self):
        ###
        self.workspace = f"/data/disk0/Workspace/{self.username}"
        self.model_path = f"{self.workspace}/Models/{self.model_id}"
        self.evaluation_path = f"{self.workspace}/Evaluations/Quanted/{self.model_id}"
        self.evaluation_output_path = (
            f"{self.evaluation_path}/{self.evaluation_framework}/"
            f"{self.model_id}{self.evaluation_framework}{self.evaluation_tasks}.json"
        )

        evalplus_results = GPTQModel.eval(
            model_or_id_or_path=self.model_path,
            framework=self.evaluation_framework,
            tasks=self.evaluation_tasks,
            batch_size=self.evaluation_batch_size,
            trust_remote_code=self.trust_remote_code,
            output_path=self.evaluation_output_path
            )