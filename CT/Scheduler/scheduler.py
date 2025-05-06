# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
import time
import threading

@dataclass
class Scheduler():
    username: str = field(default="Compiler-Toolchain")

    scheduler_gpu_type: str = field(default="4090", metadata={"choices": [
        "4090",
        "H100"
        ]})
    scheduler_exclude_gpu: int = field(default=4, metadata={"min_value": 0,
        "max_value": 6
        })
    scheduler_minmum_free_vram: int = field(default=20, metadata={"min_value": 1})
    scheduler_check_cycle_time: int = field(default=60, metadata={"min_value": 1})
    scheduler_remove_lock_time: int = field(default=60, metadata={"min_value": 1})

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

    def __post_init__(self):
        workspace: str = f"/data/disk0/Workspace/{self.username}"
        scheduler_runtime_path: str = os.path.join(workspace, "Compiler-Toolchain", "CT", "Scheduler", "runtime")
        scheduler_temp_path = os.path.join(scheduler_runtime_path, "temp")
        scheduler_lock_path = os.path.join(scheduler_runtime_path, "lock")

        threads = []

        if self.scheduler_gpu_type == "4090":
            while True:
                os.system("nvidia-smi -q -d Memory | grep -A8 'GPU' | grep 'Free' | awk \'{print $3}\' > CT/Scheduler/runtime/temp")
                with open(scheduler_temp_path, "r") as temp:
                    lines: list[str] = temp.readlines()

                gpu_free_vram: list[int] = [int(x.strip()) for x in lines]
                for i in range(len(gpu_free_vram)):
                    if i == self.scheduler_exclude_gpu:
                        gpu_free_vram[i] = 0
                    if not self.gpu_status(i, scheduler_lock_path):
                        gpu_free_vram[i] = 0

                if gpu_free_vram:
                    gpu_max_vram = gpu_free_vram.index(max(gpu_free_vram))

                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_max_vram)
                    os.system(f"rm {scheduler_temp_path}")

                    thread = threading.Thread(target=self.gpu_lock, args=(gpu_max_vram, scheduler_lock_path))
                    threads.append(thread)
                    thread.start()

                    print(f"Use CUDA:{gpu_max_vram}")
                    break
                else:
                    timer = threading.Timer(self.scheduler_check_cycle_time)
                    timer.start()
                    print(f"No Free CUDA Device Available Now, Automatically Retry After {self.scheduler_check_cycle_time} Seconds")
        else:
            while True:
                os.system("nvidia-smi -q -d Memory | grep -A8 'GPU' | grep 'Free' | awk \'{print $3}\' > temp")
                with open(scheduler_temp_path, "r") as temp:
                    lines: list[str] = temp.readlines()
                gpu_free_vram: list[int] = [int(x.strip()) for x in lines]
                if gpu_free_vram[self.scheduler_exclude_gpu] >= self.scheduler_minmum_free_vram * 1024:
                    if self.gpu_status(gpu=self.scheduler_exclude_gpu, scheduler_lock_path=scheduler_lock_path):
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.scheduler_exclude_gpu)
                        os.system(f"rm {scheduler_temp_path}")

                        thread = threading.Thread(target=self.gpu_lock, args=(self.scheduler_exclude_gpu, scheduler_lock_path))
                        threads.append(thread)
                        thread.start()

                        print(f"Use CUDA:{self.scheduler_exclude_gpu}")
                        break
                    else:
                        timer = threading.Timer(self.scheduler_remove_lock_time)
                        timer.start()
                        print(f"No Unlock CUDA Device Available Now, Automatically Retry After {self.scheduler_remove_lock_time} Seconds")
                else:
                    timer = threading.Timer(self.scheduler_check_cycle_time)
                    timer.start()
                    print(f"No Free CUDA Device Available Now, Automatically Retry After {self.scheduler_check_cycle_time} Seconds")