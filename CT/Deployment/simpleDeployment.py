# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
from CT.Deployment.FP16.simpleFP16Deployment import SimpleFP16Deployment
from CT.Deployment.GPTQ.simpleGPTQDeployment import SimpleGPTQDeployment

@dataclass
class SimpleDeployment():
    model_id: str = field(default="Qwen2.5-7B-Instruct")
    model_type: str = field(default="FP16", metadata={"choices": [
        "FP16",
        "GPTQ"
        ]})

    deployment_max_model_len: int = field(default=32768, metadata={"min_value": 1})
    deployment_gpu_memory_utilization: float = field(default=0.95, metadata={
        "min_value": 0.01,
        "max_value": 0.99
        })
    deployment_enforce_eager: bool = field(default=True)
    deployment_host: str = field(default="0.0.0.0")
    deployment_port: int = field(default=2570, metadata={
        "min_value": 1024,
        "max_value": 49151
        })
    deployment_api_key: str = field(default="yuhaolab")

    def __post_init__(self):
        if self.model_type == "FP16":
            simpleFP16Deployment = SimpleFP16Deployment(
                model_id=self.model_id,
                deployment_max_model_len=self.deployment_max_model_len,
                deployment_gpu_memory_utilization=self.deployment_gpu_memory_utilization,
                deployment_enforce_eager=self.deployment_enforce_eager,
                deployment_host=self.deployment_host,
                deployment_port=self.deployment_port,
                deployment_api_key=self.deployment_api_key
            )
        elif self.model_type == "GPTQ":
            simpleGPTQDeployment = SimpleGPTQDeployment(
                model_id=self.model_id,
                deployment_max_model_len=self.deployment_max_model_len,
                deployment_gpu_memory_utilization=self.deployment_gpu_memory_utilization,
                deployment_enforce_eager=self.deployment_enforce_eager,
                deployment_host=self.deployment_host,
                deployment_port=self.deployment_port,
                deployment_api_key=self.deployment_api_key
            )