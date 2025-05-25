import os
import sys
from pathlib import Path
import multiprocessing
import torch

# 添加 Compiler-Toolchain 到 Python 路径
toolchain_path = Path("/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain")
sys.path.append(str(toolchain_path))

# 导入 evalPlus 的评估函数
from Example.Evaluation.evalPlus import simpleEvaluation

def check_gpu_available():
    """检查可用的 GPU 并返回数量"""
    return torch.cuda.device_count()

def run_custom_evaluation(
    model_ids: list[str],
    evaluation_framework: str,
    evaluation_tasks: list[str],
):
    """直接调用评估函数，自定义参数"""
    # 检查 GPU 可用性
    num_gpus = check_gpu_available()
    if num_gpus == 0:
        raise RuntimeError("No GPU available! Please check CUDA installation.")

    print(f"Available GPUs: {num_gpus}")

    # 启动多进程评估
    multiprocessing.set_start_method("spawn")
    processes = []

    for model_id in model_ids:
        for task in evaluation_tasks:
            p = multiprocessing.Process(
                target=simpleEvaluation,
                args=(model_id, task),
            )
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    # === 在这里指定你的参数 ===
    custom_models = ["Qwen2.5-7B-Instruct"]  # 自定义模型
    custom_framework = "EvalPlus"  # 评估框架
    custom_tasks = ["humaneval", "mbpp"]  # 评估任务

    # 运行评估
    run_custom_evaluation(
        model_ids=custom_models,
        evaluation_framework=custom_framework,
        evaluation_tasks=custom_tasks,
    )