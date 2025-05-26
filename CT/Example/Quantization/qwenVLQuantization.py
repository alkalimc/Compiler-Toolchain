# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

import os
import sys
import torch
import subprocess
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/data', 'disk0', 'Workspace', 'Compiler-Toolchain', 'Compiler-Toolchain')))
from CT.Scheduler.GPU.simpleScheduler import SimpleScheduler
from CT.Quantization.simpleQuantization import SimpleQuantization

model_type: str = "QwenVL"
model_ids: list[str]  = [
    "Qwen2.5-VL-7B-Instruct",
    "Qwen2-VL-7B-Instruct"
]
data_id: str = "allenai-c4"
data_file: str = os.path.join("en", "c4-train.00001-of-01024.json.gz")
quantization_batch_size: int = 4

def simpleQuantization(model_id: str):
    try: 
        simpleScheduler = SimpleScheduler()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(simpleScheduler.gpuSelected())
        torch.cuda.set_device(simpleScheduler.gpuSelected())
        print(f"\nCUDA_VISIBLE_DEVICE = {subprocess.run("echo $CUDA_VISIBLE_DEVICES", shell=True, capture_output=True, text=True).stdout}")
        simpleQuantization = SimpleQuantization(
            model_type=model_type,
            model_id=model_id,
            data_id=data_id,
            data_file=data_file,
            quantization_batch_size=quantization_batch_size,
            quantization_device=torch.device(f"cuda:{simpleScheduler.gpuSelected()}")
        )
    except Exception as e:
        print(f"{model_id} Quantization Error, Reason: {e}")
        return
    
def main():
    multiprocessing.set_start_method("spawn")
    processes = []

    for model_id in model_ids:
        process = multiprocessing.Process(target=simpleQuantization, args=(model_id,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()