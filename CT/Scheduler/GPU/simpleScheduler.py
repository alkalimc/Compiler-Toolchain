# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
from CT.Scheduler.GPU.scheduler import Scheduler

@dataclass
class SimpleScheduler():
    scheduler_gpu_type: str = field(default="4090", metadata={"choices": [
        "4090",
        "H100"
        ]})
    scheduler_minmum_free_vram: float = field(default=22, metadata={"min_value": 0.01})
    gpu_selected: int = 0
    
    def __post_init__(self):
        scheduler = Scheduler(
            scheduler_gpu_type=self.scheduler_gpu_type,
            scheduler_minmum_free_vram=self.scheduler_minmum_free_vram
            )
        self.gpu_selected = scheduler.gpuSelected()

    def gpuSelected(self) -> int:
        return self.gpu_selected