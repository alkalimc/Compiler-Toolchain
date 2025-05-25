# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import secrets

@dataclass
class Key:
    nbytes: int = field(default=6, metadata={"min_value": 1})
    key: str = "yuhaolab"

    def __post_init__(self):
        self.key = secrets.token_urlsafe(self.nbytes)
        
    def keySelected(self) -> str:
        return self.key