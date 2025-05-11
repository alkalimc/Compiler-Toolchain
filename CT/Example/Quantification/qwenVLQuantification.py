# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

import os
import sys
import torch
import subprocess
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/data', 'disk0', 'Workspace', 'Compiler-Toolchain', 'Compiler-Toolchain')))
from CT.Quantification.simpleQuantification import SimpleQuantification
from CT.Scheduler.simpleScheduler import SimpleScheduler

model_type = "Qwen_VL"
model_ids = [
    "Qwen2-VL-7B-Instruct",
#    "Qwen2.5-VL-7B-Instruct"
]
quantize_batch_size: int = 4

def simpleQuantification(model_id: str, quantize_batch_size: int):
    try: 
        simpleScheduler = SimpleScheduler()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(simpleScheduler.gpu_selected)
        print(f"\nCUDA_VISIBLE_DEVICE = {subprocess.run("echo $CUDA_VISIBLE_DEVICES", shell=True, capture_output=True, text=True).stdout}")
        simpleQuantification = SimpleQuantification(
            model_type=model_type,
            model_id=model_id,
            quantize_batch_size=quantize_batch_size,
            quantize_device=torch.device(f"cuda:{simpleScheduler.gpu_selected}")
        )
    except Exception as e:
        print(f"{model_id} Quantification Error, Reason: {e}")
        return
    
def main():
    multiprocessing.set_start_method("spawn")
    processes = []

    for model_id in model_ids:
        process = multiprocessing.Process(target=simpleQuantification, args=(model_id, quantize_batch_size))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()