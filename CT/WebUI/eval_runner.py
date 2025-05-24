#!/usr/bin/env python3
# eval_runner.py - 简化的大模型评估调用脚本

import os
import sys
import argparse
from multiprocessing import Process

# 设置正确的路径
CT_ROOT = "/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain"
sys.path.insert(0, os.path.join(CT_ROOT, "CT/Example/Evaluation"))

def validate_model(model_name):
    """验证模型是否在支持列表中"""
    SUPPORTED_MODELS = [
        "Qwen2.5-7B-Instruct",
        "Qwen2-7B-Instruct",
        "DeepSeek-R1-Distill-Qwen-7B",
        "Qwen2-VL-7B-Instruct",
        "Qwen2.5-VL-7B-Instruct"
    ]
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型: {model_name}\n支持的模型有: {SUPPORTED_MODELS}")

def run_evaluation(model_name, eval_framework="EvalPlus"):
    """运行评估的主函数"""
    try:
        from evalPlus import simpleEvaluation
        
        # 支持的评估任务
        TASKS = ["humaneval", "mbpp"]
        
        processes = []
        for task in TASKS:
            p = Process(target=simpleEvaluation, args=(model_name, task))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
            
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="大模型评估调用脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-m", "--model", required=True, help="要评估的模型名称")
    parser.add_argument("-f", "--framework", default="EvalPlus", 
                       choices=["EvalPlus", "lm-evaluation-harness"],
                       help="使用的评估框架")
    
    args = parser.parse_args()
    
    try:
        validate_model(args.model)
        print(f"开始评估 - 模型: {args.model}, 框架: {args.framework}")
        run_evaluation(args.model, args.framework)
    except ValueError as e:
        print(str(e))
        sys.exit(1)