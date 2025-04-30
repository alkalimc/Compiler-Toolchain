# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
from lmEvaluationHarness import LmEvaluationHarness
from gptqmodel.utils.eval import EVAL
from evalPlus import EvalPlus

@dataclass
class SimpleEvaluation():
    model_id: str = field(default="Qwen2.5-7B-Instruct")
    evaluation_framework: str = field(default="LM_EVAL", metadata={"choices": [
        "LM_EVAL",
        "EVALPLUS"
        ]})
    evaluation_batch_size: int = field(default=1, metadata={"min_value": 1})

    if evaluation_framework == "LM_EVAL":
        evaluation_tasks: str = field(default="ARC_EASY", metadata={"choices": [
            "ARC_EASY",
            "ARC_CHALLENGE",
            "GSM8K_COT",
            "GSM8K_PLATINUM_COT",
            "HELLASWAG",
            "MMLU",
            "GPQA",
            "BOOLQ",
            "OPENBOOKQA"
            ]})
    else:
        evaluation_tasks: str = field(default="HUMAN", metadata={"choices": [
            "HUMAN",
            "MBPP"
            ]})
        
    def __post_init__(self):
        if self.evaluation_framework == "LM_EVAL":
            for lm_eval in EVAL.LM_EVAL:
                if self.evaluation_tasks == lm_eval.name:
                    evaluation_tasks = lm_eval
            lm_evaluation_harness = LmEvaluationHarness(
                model_id=self.model_id,
                evaluation_batch_size=self.evaluation_batch_size,
                evaluation_tasks=evaluation_tasks)
        else:
            for evalPlus in EVAL.EVALPLUS:
                if self.evaluation_tasks == evalPlus.name:
                    evaluation_tasks = evalPlus
            evalPlus = EvalPlus(
                model_id=self.model_id,
                evaluation_batch_size=self.evaluation_batch_size,
                evaluation_tasks=evaluation_tasks)