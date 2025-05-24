# powered by alkali
# Copyright 2024- alkali. All Rights Reserved.

from dataclasses import dataclass, field
import os
import subprocess

@dataclass
class Deployment():
    username: str = field(default="Compiler-Toolchain")
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
        workspace: str = os.path.join("/data/disk0/Workspace", self.username)
        deployment_served_model_name: str = self.model_id

        if self.model_type == "FP16":
            model_path: str = os.path.join(workspace, "Models", self.model_id)
            
            if self.deployment_enforce_eager == True:
                print(subprocess.run(f"vllm serve {model_path} --max_model_len {self.deployment_max_model_len} --gpu-memory-utilization {self.deployment_gpu_memory_utilization} --enforce-eager --host {self.deployment_host} --port {self.deployment_port} --served-model-name {deployment_served_model_name} --api-key {self.deployment_api_key}", shell=True, capture_output=True, text=True).stdout)
            else:
                model_path: str = os.path.join(workspace, "Models", self.model_id)
                print(subprocess.run(f"vllm serve {model_path} --max_model_len {self.deployment_max_model_len} --gpu-memory-utilization {self.deployment_gpu_memory_utilization} --host {self.deployment_host} --port {self.deployment_port} --served-model-name {deployment_served_model_name} --api-key {self.deployment_api_key}", shell=True, capture_output=True, text=True).stdout)
        elif self.model_type == "gptq":
            model_path: str = os.path.join(workspace, "Models", "Quanted", self.model_id)

            if self.deployment_enforce_eager == True:
                model_path: str = os.path.join(workspace, "Models", self.model_id)
                print(subprocess.run(f"vllm serve {model_path} --max_model_len {self.deployment_max_model_len} --gpu-memory-utilization {self.deployment_gpu_memory_utilization} --enforce-eager --host {self.deployment_host} --port {self.deployment_port} --served-model-name {deployment_served_model_name} --api-key {self.deployment_api_key} --quantization gptq_marlin", shell=True, capture_output=True, text=True).stdout)
            else:
                model_path: str = os.path.join(workspace, "Models", self.model_id)
                print(subprocess.run(f"vllm serve {model_path} --max_model_len {self.deployment_max_model_len} --gpu-memory-utilization {self.deployment_gpu_memory_utilization} --host {self.deployment_host} --port {self.deployment_port} --served-model-name {deployment_served_model_name} --api-key {self.deployment_api_key} --quantization gptq_marlin", shell=True, capture_output=True, text=True).stdout)