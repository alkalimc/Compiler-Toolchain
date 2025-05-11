# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
from lmEvaluationHarness import LmEvaluationHarness
from gptqmodel.utils.eval import EVAL
from evalPlus import EvalPlus

###
import argparse
from enum import Enum

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

###
def main():
    parser = argparse.ArgumentParser(description="运行模型评估任务")
    parser.add_argument("--framework", type=str, required=True, choices=["LM_EVAL", "EVALPLUS"], 
                       help="评估框架（LM_EVAL 或 EVALPLUS）")
    parser.add_argument("--task", type=str, required=True, 
                       help="任务名称（LM_EVAL:ARC_EASY,ARC_CHALLENGE,GSM8K_COT,GSM8K_PLATINUM_COT,HELLASWAG,MMLU,GPQA,BOOLQ,OPENBOOKQA 或 EVALPLUS:HUMAN,MBPP）")
    parser.add_argument("--model_id", type=str, default="Qwen2.5-7B-Instruct",
                       help="模型名称")
    parser.add_argument("--batch_size", type=int, default=1,)
    parser.add_argument("--username", type=str, default="Compiler-Toolchain",
                       help="用户名")
    args = parser.parse_args()

    if args.framework == "LM_EVAL":
        framework_class = LmEvaluationHarness
        task_enum = getattr(EVAL.LM_EVAL, args.task, None)
    else:
        framework_class = EvalPlus
        task_enum = getattr(EVAL.EVALPLUS, args.task, None)

    if not task_enum:
        valid_tasks = [e.name for e in (EVAL.LM_EVAL if args.framework == "LM_EVAL" else EVAL.EVALPLUS)]
        raise ValueError(f"任务 '{args.task}' 无效！可选值：{valid_tasks}")

    framework_class(
        username=args.username,
        model_id=args.model_id,
        evaluation_tasks=task_enum,
        evaluation_batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()