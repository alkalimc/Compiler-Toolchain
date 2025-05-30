# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
from CT.Scheduler.Port.scheduler import Scheduler

@dataclass
class SimpleScheduler():
    scheduler_port: list[int] = field(default_factory=lambda: list(range(1024, 49151)))
    port_selected: int = 0

    def __post_init__(self):
        scheduler = Scheduler(scheduler_port=self.scheduler_port)
        self.port_selected = scheduler.portSelected()

    def portSelected(self) -> int:
        return self.port_selected