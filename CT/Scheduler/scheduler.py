# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
import numpy
import time

@dataclass
class Scheduler():
    scheduler_gpu_type: str = field(default="4090", metadata={"choices": [
        "4090",
        "H100"
        ]})
    scheduler_exclude_gpu: int = field(default=4, metadata={"choices": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        ]})
    scheduler_minmum_free_vram: int = field(default=20, metadata={"min_value": 1})
    scheduler_check_cycle_time: int = field(default=60, metadata={"min_value": 1})

    def __post_init__(self):
        if self.scheduler_gpu_type == "4090":
            while True:
                os.system('nvidia-smi -q -d Memory | grep -A8 "GPU" | grep "Free" | awk \'{print $3}\' > temp')
                with open('temp', 'r') as temp:
                    lines: list[str] = temp.readlines()
                gpu_free_vram: list[int] = [int(x.strip()) for x in lines]
                gpu_free_vram_exclude: list[int] = gpu_free_vram[:self.scheduler_exclude_gpu] + gpu_free_vram[self.scheduler_exclude_gpu+1:]
                gpu_enough_vram: list[int] = [index for index, memory in enumerate(gpu_free_vram_exclude) if memory >= self.scheduler_minmum_free_vram * 1024]
                if gpu_enough_vram:
                    gpu_max_vram: int = gpu_enough_vram[numpy.argmax([gpu_free_vram_exclude[i] for i in gpu_enough_vram])]
                    if gpu_max_vram >= self.scheduler_exclude_gpu:
                        gpu_max_vram += 1
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_max_vram)
                    os.system('rm temp')
                    print(f"Use CUDA:{gpu_max_vram}")
                    break
                else:
                    print(f"No Free CUDA Device Available Now, Automatically Retry After {self.scheduler_check_cycle_time} Seconds")
                    time.sleep(self.scheduler_check_cycle_time)
        else:
            while True:
                os.system('nvidia-smi -q -d Memory | grep -A8 "GPU" | grep "Free" | awk \'{print $3}\' > temp')
                with open('temp', 'r') as temp:
                    lines: list[str] = temp.readlines()
                gpu_free_vram: list[int] = [int(x.strip()) for x in lines]
                if gpu_free_vram[4] >= self.scheduler_minmum_free_vram * 1024:
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(4)
                    os.system('rm temp')
                    print(f"Use CUDA:{self.scheduler_exclude_gpu}")
                    break
                else:
                    print(f"No Free CUDA Device Available Now, Automatically Retry After {self.scheduler_check_cycle_time} Seconds")
                    time.sleep(self.scheduler_check_cycle_time)