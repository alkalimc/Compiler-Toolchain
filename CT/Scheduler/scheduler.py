# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
import time
import threading

@dataclass
class Scheduler():
    username: str = field(default="Compiler-Toolchain")
    
    scheduler_device_map: dict = field(default_factory=lambda: {
        "4090": [0, 1, 2, 3, 5],
        "H100": [4, 6]
    })
    scheduler_gpu_type: str = field(default="4090", metadata={"choices": [
        "4090",
        "H100"
        ]})
    scheduler_minmum_free_vram: int = field(default=22, metadata={"min_value": 1})
    scheduler_check_cycle_time: int = field(default=64, metadata={"min_value": 1})
    scheduler_remove_lock_time: int = field(default=60, metadata={"min_value": 1})
    scheduler_gpu: int
    
    def gpu_status(self, gpu: int, scheduler_lock_path: str) -> bool:
        lock_file: str = f"{scheduler_lock_path}/CUDA:{gpu}.lock"
        if os.path.exists(lock_file):
            with open(lock_file, "r") as lock:
                lock_time = float(lock.read().strip())
            if time.time() - lock_time >= self.scheduler_remove_lock_time:
                os.remove(lock_file)
                return True
            return False
        return True
    def gpu_lock(self, gpu: int, scheduler_lock_path: str):
        lock_file: str = f"{scheduler_lock_path}/CUDA:{gpu}.lock"
        with open(lock_file, "w") as lock:
            lock.write(str(time.time()))

    def scheduler_gpu(self, scheduler_temp_path: str, scheduler_available_gpu, scheduler_lock_path: str) -> int:
        while True:
            os.system("nvidia-smi -q -d Memory | grep -A8 'GPU' | grep 'Free' | awk \'{print $3}\' > /data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain/CT/Scheduler/runtime/temp")
            with open(scheduler_temp_path, "r") as temp:
                lines: list[str] = temp.readlines()
            gpu_free_vram: list[int] = [int(x.strip()) for x in lines]

            for i in scheduler_available_gpu:
                if gpu_free_vram[i] >= self.scheduler_minmum_free_vram * 1024:
                    if self.gpu_status(i, scheduler_lock_path):
                        self.gpu_lock(gpu = i, scheduler_lock_path=scheduler_lock_path)
                        print(f"\nUse CUDA:{i}")
                        return i
            print(f"\nNo Free CUDA Device Available Now, Automatically Retry After {self.scheduler_check_cycle_time} Seconds")
            time.sleep(self.scheduler_check_cycle_time)

    def __post_init__(self):
        workspace: str = os.path.join("/data/disk0/Workspace", self.username)
        scheduler_runtime_path: str = os.path.join(workspace, "Compiler-Toolchain", "CT", "Scheduler", "runtime")
        scheduler_temp_path = os.path.join(scheduler_runtime_path, "temp")
        scheduler_lock_path = os.path.join(scheduler_runtime_path, "lock")
        scheduler_available_gpu = self.scheduler_device_map[self.scheduler_gpu_type]

        if not os.path.exists(scheduler_lock_path):
            os.makedirs(scheduler_lock_path)

        self.gpu_selected = self.scheduler_gpu(self,
            scheduler_temp_path=scheduler_temp_path,
            scheduler_available_gpu=scheduler_available_gpu,
            scheduler_lock_path=scheduler_lock_path
            )
        
    def gpu_selected(self) -> int:
        return self.gpu_selected