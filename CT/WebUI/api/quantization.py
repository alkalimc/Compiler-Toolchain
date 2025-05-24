import multiprocessing
import sys
import os
import contextlib
from pathlib import Path
import traceback
import time
from .config import PROGRESS_LOG, GPTQ_LOG_DIR, QUANTIFICATION_PATH
from .logging_utils import log_error

current_quant_process = None

def quantification_entrypoint(model_id, log_path, is_vl_model):
    """
    后端包装函数，用于捕获quantification输出并写入日志。
    """
    try:
        os.makedirs(GPTQ_LOG_DIR, exist_ok=True)
        os.chdir(GPTQ_LOG_DIR)
    
        with open(log_path, 'a') as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                # 动态导入，避免多进程冲突
                sys.path.append(QUANTIFICATION_PATH)
                if is_vl_model:
                    from qwenVLQuantification import simpleQuantification
                    print(f"[INFO] 启动VL模型量化: {model_id}")
                else:
                    from quantification import simpleQuantification
                    print(f"[INFO] 启动普通模型量化: {model_id}")
                    
                simpleQuantification(model_id)

                print(f"[INFO] 模型量化完成: {model_id}")

    except Exception as e:
        with open(log_path, 'a') as f:
            f.write(f"[ERROR] 模型量化异常: {e}\n")

def is_quant_running():
    global current_quant_process
    return current_quant_process is not None and current_quant_process.is_alive()

def run_quantification(model_name):
    """
    封装量化进程，添加错误处理
    """
    global current_quant_process

    try:
        # 清空进度日志
        with open(PROGRESS_LOG, 'w') as f:
            f.write("")
        
        is_vl_model = "VL" in model_name.upper()
        log_error(f"开始量化模型: {model_name} (类型: {'VL' if is_vl_model else '普通'})", "quant")

        # 创建并启动量化进程
        current_quant_process = multiprocessing.Process(
            target=quantification_entrypoint,
            args=(model_name, PROGRESS_LOG, is_vl_model,)
        )
        current_quant_process.start()

        return current_quant_process.pid
    
    except Exception as e:
        error_msg = f"量化失败 - 模型:{model_name} 错误:{traceback.format_exc()}"
        log_error(error_msg, "quant")
        current_quant_process = None
        raise

def cancel_quantization():
    global current_quant_process
    
    try:
        if not is_quant_running():
            return False, '没有正在运行的量化进程'
        
        # 先写入日志再终止进程
        with open(PROGRESS_LOG, 'a') as f:
            f.write("[INFO] 正在取消量化进程...\n")
        
        current_quant_process.terminate()
        current_quant_process.join(timeout=2)  # 设置超时
        
        # 确认进程已终止后再记录
        with open(PROGRESS_LOG, 'a') as f:
            f.write("[INFO] 量化进程已被用户取消\n")
        
        current_quant_process = None
        
        return True, '量化进程已成功取消'
    except Exception as e:
        error_msg = f"取消量化失败: {str(e)}"
        log_error(error_msg, "backend")
        return False, error_msg