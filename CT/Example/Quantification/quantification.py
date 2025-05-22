# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

import os
import sys
import torch
import subprocess
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/data', 'disk0', 'Workspace', 'Compiler-Toolchain', 'Compiler-Toolchain')))
from CT.Scheduler.simpleScheduler import SimpleScheduler
from CT.Quantification.simpleQuantification import SimpleQuantification

model_type: str = "Qwen"
model_ids: list[str]  = [
    "Qwen2.5-7B-Instruct",
    "Qwen2-7B-Instruct",
    "DeepSeek-R1-Distill-Qwen-7B"
]
data_id: str = "allenai-c4"
data_file: str = os.path.join("en", "c4-train.00001-of-01024.json.gz")
quantize_batch_size: int = 4

def simpleQuantification(model_id: str):
    try: 
        simpleScheduler = SimpleScheduler()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(simpleScheduler.gpu_selected)
        torch.cuda.set_device(simpleScheduler.gpu_selected)
        print(f"\nCUDA_VISIBLE_DEVICE = {subprocess.run("echo $CUDA_VISIBLE_DEVICES", shell=True, capture_output=True, text=True).stdout}")
        simpleQuantification = SimpleQuantification(
            model_type=model_type,
            model_id=model_id,
            data_id=data_id,
            data_file=data_file,
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
        process = multiprocessing.Process(target=simpleQuantification, args=(model_id,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()