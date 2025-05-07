from typing import Dict, Callable
from app.models.function import FunctionDescriptor

class FunctionManager:
    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self.descriptors: Dict[str, FunctionDescriptor] = {}

    def register(self, descriptor: FunctionDescriptor, func: Callable):
        self.functions[descriptor.name] = func
        self.descriptors[descriptor.name] = descriptor
        return func

    def get_function(self, name: str) -> Callable:
        return self.functions.get(name)

    def get_descriptors(self) -> list:
        return list(self.descriptors.values())

function_manager = FunctionManager()  # 全局单例实例