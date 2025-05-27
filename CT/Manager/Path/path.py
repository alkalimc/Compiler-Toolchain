# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os

@dataclass
class Path:
    username: str = field(default="Compiler-Toolchain")
    workspace: str = field(default="/data/disk0/Workspace")
    path: str = os.path.join("/data/disk0/Workspace", "Compiler-Toolchain")

    def __post_init__(self):
        self.path: str = os.path.join(self.workspace, self.username)
        
    def path_selected(self) -> str:
        return self.path