# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
import time
import random
import psutil

@dataclass
class Scheduler():
    username: str = field(default="Compiler-Toolchain")
    
    scheduler_port: list[int] = field(default_factory=lambda: list(range(1024, 49151)))
    scheduler_check_cycle_time: int = field(default=64, metadata={"min_value": 1})
    scheduler_remove_lock_time: int = field(default=60, metadata={"min_value": 1})
    port_selected: int = 0
    
    def port_status(self, port: int, scheduler_lock_path: str) -> bool:
        lock_file: str = f"{scheduler_lock_path}/{port}.lock"
        if os.path.exists(lock_file):
            with open(lock_file, "r") as lock:
                lock_time = float(lock.read().strip())
            if time.time() - lock_time >= self.scheduler_remove_lock_time:
                os.remove(lock_file)
                return True
            return False
        return True
    def port_lock(self, port: int, scheduler_lock_path: str):
        lock_file: str = f"{scheduler_lock_path}/{port}.lock"
        with open(lock_file, "w") as lock:
            lock.write(str(time.time()))
    def portUsed(self) -> set:
        port_used = set()
        for connection in psutil.net_connections(kind='inet'):
            port_used.add(connection.laddr.port)
        return port_used

    def schedulerPort(self, scheduler_lock_path: str) -> int:
        while True:
            random.shuffle(self.scheduler_port)
            port_used: set = self.portUsed()
            for port in self.scheduler_port:
                if port not in port_used:
                    if self.port_status(port=port, scheduler_lock_path=scheduler_lock_path):
                        self.port_lock(port=port, scheduler_lock_path=scheduler_lock_path)
                        print(f"\nUse Port:{port}")
                        return port
            print(f"\nNo Free Port Available Now, Automatically Retry After {self.scheduler_check_cycle_time} Seconds")
            time.sleep(self.scheduler_check_cycle_time)

    def __post_init__(self):
        workspace: str = os.path.join("/data/disk0/Workspace", self.username)
        scheduler_runtime_path: str = os.path.join(workspace, "Compiler-Toolchain", "CT", "Scheduler", "Port", "runtime")
        scheduler_temp_path = os.path.join(scheduler_runtime_path, "temp")
        scheduler_lock_path = os.path.join(scheduler_runtime_path, "lock")

        if not os.path.exists(scheduler_lock_path):
            os.makedirs(scheduler_lock_path)

        self.port_selected = self.schedulerPort(scheduler_lock_path=scheduler_lock_path)
        
    def portSelected(self) -> int:
        return self.port_selected