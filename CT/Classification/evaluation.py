# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field

@dataclass
class EvaluationClassification():
    evaluation_task: str = field(default="ARC_EASY", metadata={"choices": [
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
    evaluation_framework: str = "LM_EVAL"

    def __post_init__(self):
        if self.evaluation_task in [
            "arc_easy",
            "arc_challenge",
            "gsm8k_cot",
            "gsm8k_platinum_cot",
            "hellaswag",
            "mmlu",
            "gpqa",
            "boolq",
            "openbookqa"
            ]:
            self.evaluation_framework = "LM_EVAL"
        elif self.evaluation_task in [
            "humaneval",
            "mbpp"
            ]:
            self.evaluation_framework = "EVALPLUS"

    def evaluationFramework(self) -> str:
        return self.evaluation_framework