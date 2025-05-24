import os
from datetime import datetime
from pathlib import Path
from .config import LOG_DIR, ERROR_LOG, PROGRESS_LOG, EVALUATION_LOG

def setup_logging():
    """初始化日志目录和文件"""
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(ERROR_LOG):
        with open(ERROR_LOG, 'w') as f:
            f.write("====== Quantization Error Log ======\n")
    Path(PROGRESS_LOG).touch(exist_ok=True)
    Path(EVALUATION_LOG).touch(exist_ok=True)

def log_error(error_msg, source="backend"):
    """
    记录错误到日志文件
    :param error_msg: 错误信息
    :param source: 错误来源 (backend/frontend/quant)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{source.upper()}] {error_msg}\n"
    
    try:
        with open(ERROR_LOG, 'a') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"无法写入日志文件: {str(e)}")