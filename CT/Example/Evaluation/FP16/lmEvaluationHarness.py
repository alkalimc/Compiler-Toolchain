# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

import os
import sys
import time
import torch
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/data', 'disk0', 'Workspace', 'Compiler-Toolchain', 'Compiler-Toolchain')))
from CT.Scheduler.GPU.simpleScheduler import SimpleScheduler
from CT.Evaluation.simpleEvaluation import SimpleEvaluation

model_ids: list[str] = [
    "Qwen2.5-7B-Instruct",
    "Qwen2-7B-Instruct",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Qwen2-VL-7B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "chatglm3-6b"
]
evaluation_framework: str = "lm-evaluation-harness"
evaluation_tasks:  list[str] = [
    "arc_easy",
    "arc_challenge",
    "gsm8k_cot",
    "gsm8k_platinum_cot",
    "hellaswag",
    "mmlu",
#    "gpqa",
    "boolq",
    "openbookqa"
]
evaluation_batch_size: int = 4

process_cycle: int = 16

def simpleEvaluation(model_id: str, evaluation_task: str):
    try: 
        simpleScheduler = SimpleScheduler()
        torch.cuda.set_device(simpleScheduler.gpuSelected())
        simpleEvaluation = SimpleEvaluation(
            model_id=model_id,
            evaluation_framework=evaluation_framework,
            evaluation_task=evaluation_task,
            evaluation_device=torch.device(f"cuda:{simpleScheduler.gpuSelected()}"),
            evaluation_batch_size=evaluation_batch_size
        )
    except Exception as e:
        print(f"{model_id} Evaluation Error, Reason: {e}")
        return
    
def main():
    multiprocessing.set_start_method("spawn")
    processes = []

    for model_id in model_ids:
        for evaluation_task in evaluation_tasks:
            process = multiprocessing.Process(target=simpleEvaluation, args=(model_id, evaluation_task))
            processes.append(process)
            process.start()
            time.sleep(process_cycle)

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()