# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

import os
import sys
import subprocess
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/data', 'disk0', 'Workspace', 'Compiler-Toolchain', 'Compiler-Toolchain')))
from CT.Deployment.simpleDeployment import SimpleDeployment
from CT.Scheduler.GPU.simpleScheduler import SimpleScheduler as GPUSimpleScheduler
from CT.Scheduler.Port.simpleScheduler import SimpleScheduler as PortSimpleScheduler

model_ids: list[str] = [
    "Qwen2.5-7B-Instruct",
    "Qwen2-7B-Instruct",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Qwen2-VL-7B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "chatglm3-6b"
]

model_type: str = "FP16"

deployment_gpu_memory_utilization: float = 0.95
deployment_api_key: str = "yuhaolab"

def simpleDeployment(model_id: str):
    try:
        gpuSimpleScheduler = GPUSimpleScheduler()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuSimpleScheduler.gpuSelected())
        print(f"\nCUDA_VISIBLE_DEVICE = {subprocess.run("echo $CUDA_VISIBLE_DEVICES", shell=True, capture_output=True, text=True).stdout}")

        portSimpleScheduler = PortSimpleScheduler()
        deployment_port = portSimpleScheduler.portSelected()
        
        simpleDeployment = SimpleDeployment(
            model_id=model_id,
            model_type=model_type,
            deployment_gpu_memory_utilization=deployment_gpu_memory_utilization,
            deployment_port=deployment_port,
            deployment_api_key=deployment_api_key
        )
    except Exception as e:
        print(f"{model_id} Deployment Error, Reason: {e}")
        return
    
def main():
    multiprocessing.set_start_method("spawn")
    processes = []

    for model_id in model_ids:
        process = multiprocessing.Process(target=simpleDeployment, args=(model_id,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()