import os
from pathlib import Path

# 基础配置
HOST = '10.20.108.87'
PORT = 7678

# 日志配置
LOG_DIR = "/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain/CT/WebUI/logs"
ERROR_LOG = os.path.join(LOG_DIR, "quantization_errors.log")
PROGRESS_LOG = os.path.join(LOG_DIR, "quantization_progress.log") 
EVALUATION_LOG = os.path.join(LOG_DIR, "evaluation_progress.log")

# 路径配置
GPTQ_LOG_DIR = "/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain/CT/WebUI/gptq_log"
QUANTIFICATION_PATH = "/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain/CT/Example/Quantification"
EVALUATION_PATH = "/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain/CT/Example/Evaluation"

# 默认参数
DEFAULT_QUANTIZATION_PARAMS = {
    "model_name": None,
    "precision": 4  # 默认精度
}