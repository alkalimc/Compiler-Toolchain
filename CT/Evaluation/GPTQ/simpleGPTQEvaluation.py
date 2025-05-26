# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
from gptqmodel.utils.eval import EVAL
from CT.Evaluation.GPTQ.lmEvaluationHarness import LmEvaluationHarness
from CT.Evaluation.GPTQ.evalPlus import EvalPlus

@dataclass
class SimpleGPTQEvaluation():
    model_id: str = field(default="Qwen2.5-7B-Instruct-W4A16-gptq")
    evaluation_framework: str = field(default="LM_EVAL", metadata={"choices": [
        "LM_EVAL",
        "EVALPLUS"
        ]})
    evaluation_task: str = field(default="ARC_EASY", metadata={"choices": [
        "ARC_EASY",
        "ARC_CHALLENGE",
        "GSM8K_COT",
        "GSM8K_PLATINUM_COT",
        "HELLASWAG",
        "MMLU",
        "GPQA",
        "BOOLQ",
        "OPENBOOKQA",
        "HUMAN",
        "MBPP"
        ]})
    evaluation_batch_size: int = field(default=1, metadata={"min_value": 1})
        
    def __post_init__(self):
        if self.evaluation_framework == "LM_EVAL":
            for lm_eval in EVAL.LM_EVAL:
                if self.evaluation_task == lm_eval.name:
                    evaluation_task = lm_eval
            lm_evaluation_harness = LmEvaluationHarness(
                model_id=self.model_id,
                evaluation_batch_size=self.evaluation_batch_size,
                evaluation_task=evaluation_task)
        elif self.evaluation_framework == "EVALPLUS":
            for evalPlus in EVAL.EVALPLUS:
                if self.evaluation_task == evalPlus.name:
                    evaluation_task = evalPlus
            evalPlus = EvalPlus(
                model_id=self.model_id,
                evaluation_batch_size=self.evaluation_batch_size,
                evaluation_task=evaluation_task)