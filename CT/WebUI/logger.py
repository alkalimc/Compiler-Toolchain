import os
from datetime import datetime

LOG_DIR = "/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain/CT/WebUI/ERROR_logs"
LOG_FILE = os.path.join(LOG_DIR, "quantization_errors.log")
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB 
MAX_BACKUP_COUNT = 5  # 最大保留的旧日志文件数

def _rotate_logs():
    """执行日志轮换"""
    if not os.path.exists(LOG_FILE):
        return

    # 检查当前日志大小
    if os.path.getsize(LOG_FILE) <= MAX_LOG_SIZE:
        return

    # 删除最旧的日志文件（如果超过最大数量）
    existing_logs = sorted(
        [f for f in os.listdir(LOG_DIR) if f.startswith("quantization_errors") and f != "quantization_errors.log"],
        key=lambda x: os.path.getmtime(os.path.join(LOG_DIR, x))
    )
    while len(existing_logs) >= MAX_BACKUP_COUNT:
        oldest_log = existing_logs.pop(0)
        os.remove(os.path.join(LOG_DIR, oldest_log))

    # 重命名当前日志文件（添加时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_log = os.path.join(LOG_DIR, f"quantization_errors_{timestamp}.log")
    os.rename(LOG_FILE, archived_log)

def log_error(error_msg, source="backend"):
    """记录错误到日志文件（带自动轮换功能）"""
    try:
        # 确保日志目录存在
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # 检查并执行日志轮换
        _rotate_logs()
        
        # 写入日志
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] [{source.upper()}] {error_msg}\n")
            
    except Exception as e:
        print(f"⚠️ 无法写入日志文件: {str(e)}")  # 作为最后兜底输出到控制台

# 初始化时创建空日志文件
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("=== Quantization Error Log ===\n")