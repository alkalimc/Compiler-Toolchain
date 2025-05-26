# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

import os
import sys
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/data', 'disk0', 'Workspace', 'Compiler-Toolchain', 'Compiler-Toolchain')))
from CT.Classification.model import ModelClassification

model_ids: list[str]  = [
    "Qwen2.5-7B-Instruct",
    "Qwen2-7B-Instruct",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Qwen2.5-VL-7B-Instruct",
    "Qwen2-VL-7B-Instruct",
    "Qwen2.5-7B-Instruct-W4A16-gptq",
    "Qwen2-7B-Instruct-W4A16-gptq",
    "DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq",
    "Qwen2.5-VL-7B-Instruct-W4A16-gptq",
    "Qwen2-VL-7B-Instruct-W4A16-gptq"
]

def modelClassification(model_id: str):
    try: 
        modelClassification = ModelClassification(model_id=model_id)
        print(f"{model_id} Type:{modelClassification.modelType()}\n")
        print(f"{model_id} Class:{modelClassification.modelClassification()}\n")
        
    except Exception as e:
        print(f"{model_id} Classification Error, Reason: {e}")
        return
    
def main():
    multiprocessing.set_start_method("spawn")
    processes = []

    for model_id in model_ids:
        process = multiprocessing.Process(target=modelClassification, args=(model_id,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()