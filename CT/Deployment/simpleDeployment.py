# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
import subprocess
from CT.Deployment.deployment import Deployment

@dataclass
class SimpleDeployment():
    model_id: str = field(default="Qwen2.5-7B-Instruct")
    model_type: str = field(default="FP16", metadata={"choices": [
        "FP16"
        "gptq"
        ]})

    deployment_max_model_len: int = field(default=32768, metadata={"min_value": 1})
    deployment_gpu_memory_utilization: float = field(default=0.95, metadata={"min_value": 0.01})
    deployment_enforce_eager: bool = field(default=True)
    deployment_host: str = field(default="0.0.0.0")
    deployment_port: str = field(default="2570")
    deployment_api_key: str = field(default="yuhaolab")

    def __post_init__(self):
        Deployment(
            model_id=self.model_id,
            model_type=self.model_type,
            deployment_max_model_len=self.deployment_max_model_len,
            deployment_gpu_memory_utilization=self.deployment_gpu_memory_utilization,
            deployment_enforce_eager=self.deployment_enforce_eager,
            deployment_host=self.deployment_host,
            deployment_port=self.deployment_port,
            deployment_api_key=self.deployment_api_key
        )