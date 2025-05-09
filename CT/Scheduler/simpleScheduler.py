# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
from CT.Scheduler.scheduler import Scheduler

@dataclass
class SimpleScheduler():
    scheduler_gpu_type: str = field(default="4090", metadata={"choices": [
        "4090",
        "H100"
        ]})
    gpu_selected: int
    
    def __post_init__(self):
        scheduler = Scheduler(scheduler_gpu_type=self.scheduler_gpu_type)
        self.gpu_selected = scheduler.gpu_selected

    def gpu_selected(self) -> int:
        return self.gpu_selected