# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

import os
import sys
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/data', 'disk0', 'Workspace', 'Compiler-Toolchain', 'Compiler-Toolchain')))
from CT.Classification.evaluation import EvaluationClassification

evaluation_tasks: list[str] = [
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
]

def evaluationClassification(evaluation_task: str):
    try:
        evaluationClassification = EvaluationClassification(
            evaluation_task=evaluation_task
        )
        print(f"{evaluation_task} Evaluation Framework:{evaluationClassification.evaluationFramework()}\n")
    except Exception as e:
        print(f"{evaluation_task} Classification Error, Reason: {e}")
        return
    
def main():
    multiprocessing.set_start_method("spawn")
    processes = []

    for evaluation_task in evaluation_tasks:
        process = multiprocessing.Process(target=evaluationClassification, args=(evaluation_task,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()