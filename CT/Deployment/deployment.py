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
        workspace: str = os.path.join("/data/disk0/Workspace", self.username)
        deployment_served_model_name: str = self.model_id

        if self.model_type == "FP16":
            model_path: str = os.path.join(workspace, "Models", self.model_id)
            
            if self.deployment_enforce_eager:
                print(subprocess.run(
                    f"vllm serve {model_path} "
                    f"--max_model_len {self.deployment_max_model_len} "
                    f"--model_args pretrained='{model_path}' "
                    f"--gpu-memory-utilization {self.deployment_gpu_memory_utilization} "
                    f"--enforce-eager "
                    f"--host {self.deployment_host} "
                    f"--port {self.deployment_port} "
                    f"--served-model-name {deployment_served_model_name} "
                    f"--api-key {self.deployment_api_key}",
                    shell=True,
                    capture_output=True,
                    text=True
                ).stdout)
            else:
                print(subprocess.run(
                    f"vllm serve {model_path} "
                    f"--max_model_len {self.deployment_max_model_len} "
                    f"--model_args pretrained='{model_path}' "
                    f"--gpu-memory-utilization {self.deployment_gpu_memory_utilization} "
                    f"--host {self.deployment_host} "
                    f"--port {self.deployment_port} "
                    f"--served-model-name {deployment_served_model_name} "
                    f"--api-key {self.deployment_api_key}",
                    shell=True,
                    capture_output=True,
                    text=True
                ).stdout)
        elif self.model_type == "gptq":
            model_path: str = os.path.join(workspace, "Models", "Quanted", self.model_id)

            if self.deployment_enforce_eager:
                print(subprocess.run(
                    f"vllm serve {model_path} "
                    f"--max_model_len {self.deployment_max_model_len} "
                    f"--model_args pretrained='{model_path}' "
                    f"--gpu-memory-utilization {self.deployment_gpu_memory_utilization} "
                    f"--enforce-eager "
                    f"--host {self.deployment_host} "
                    f"--port {self.deployment_port} "
                    f"--served-model-name {deployment_served_model_name} "
                    f"--api-key {self.deployment_api_key} "
                    f"--quantization gptq_marlin",
                    shell=True,
                    capture_output=True,
                    text=True
                ).stdout)
            else:
                print(subprocess.run(
                    f"vllm serve {model_path} "
                    f"--max_model_len {self.deployment_max_model_len} "
                    f"--model_args pretrained='{model_path}' "
                    f"--gpu-memory-utilization {self.deployment_gpu_memory_utilization} "
                    f"--host {self.deployment_host} "
                    f"--port {self.deployment_port} "
                    f"--served-model-name {deployment_served_model_name} "
                    f"--api-key {self.deployment_api_key} "
                    f"--quantization gptq_marlin",
                    shell=True,
                    capture_output=True,
                    text=True
                ).stdout)