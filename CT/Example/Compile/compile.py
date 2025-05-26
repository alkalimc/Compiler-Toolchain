import os
import sys
import logging
import torch
import torch.nn as nn
import argparse
import subprocess
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/data', 'disk0', 'Workspace', 'Compiler-Toolchain', 'Compiler-Toolchain')))
from CT.Scheduler.GPU.simpleScheduler import SimpleScheduler
from CT.Compilation.simpleCompile import simpleCompiler

Model_type: str = "Qwen"
input_datapt: list[str] = [
    '/data/disk0/Workspace/Compiler-Toolchain/Models/Quanted/Qwen2-7B-Instruct-W4A16-gptq',
    '/data/disk0/Workspace/Compiler-Toolchain/Models/Quanted/Qwen2.5-7B-Instruct-W4A16-gptq'
]
output_datapt: str = "./temp"
wbits: int = 4
group_size: int = 128
device: str = "cuda:0"

def simpleCompile(datapt:str):
    try:
        simpleScheduler=SimpleScheduler()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(simpleScheduler.gpuSelected())
        torch.cuda.set_device(simpleScheduler.gpuSelected())
        print(f"\nCUDA_VISIBLE_DEVICE = {subprocess.run("echo $CUDA_VISIBLE_DEVICES", shell=True, capture_output=True, text=True).stdout}")
        simpleCompilation = simpleCompiler(
            model_type=Model_type,
            input_datapt=datapt,
            output_datapt=output_datapt,
            wbits=wbits,
            group_size=group_size,
            device=torch.device(f"cuda:{simpleScheduler.gpuSelected()}")
        )

    except Exception as e:
        print(f"{datapt} Compilation Error, Reason: {e}")
        return

def main():
    multiprocessing.set_start_method("spawn")
    processes = []

    for datapt in input_datapt:
        process = multiprocessing.Process(target=simpleCompile, args=(datapt,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()
    