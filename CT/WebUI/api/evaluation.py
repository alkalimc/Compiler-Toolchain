import multiprocessing
import sys
import contextlib
import traceback
import time
from .config import EVALUATION_LOG, EVALUATION_PATH
from .logging_utils import log_error

current_eval_process = None

def evaluation_entrypoint(model_id, eval_method, log_path):
    """
    评估任务入口函数
    """
    try:
        with open(log_path, 'a') as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                sys.path.append(EVALUATION_PATH)
                
                print(f"[INFO] 开始评估: {model_id} ({eval_method})")
                
                if eval_method in ["humaneval", "mbpp"]:
                    from evalPlus import simpleEvaluation
                else:
                    from lmEvaluationHarness import simpleEvaluation
                
                simpleEvaluation(model_id=model_id, evaluation_task=eval_method)
                
                print(f"[INFO] 评估完成: {model_id} ({eval_method})")

    except Exception as e:
        with open(log_path, 'a') as f:
            f.write(f"[ERROR] 评估异常: {e}\n")

def is_eval_running():
    global current_eval_process
    return current_eval_process is not None and current_eval_process.is_alive()

def run_evaluation(model_name, eval_method):
    """
    启动评估进程
    """
    global current_eval_process

    try:
        # 清空评估日志
        with open(EVALUATION_LOG, 'w') as f:
            f.write("")
        
        log_error(f"开始评估模型: {model_name} (方法: {eval_method})", "eval")

        current_eval_process = multiprocessing.Process(
            target=evaluation_entrypoint,
            args=(model_name, eval_method, EVALUATION_LOG)
        )
        current_eval_process.start()

        return current_eval_process.pid
    
    except Exception as e:
        error_msg = f"评估失败 - 模型:{model_name} 方法:{eval_method} 错误:{traceback.format_exc()}"
        log_error(error_msg, "eval")
        current_eval_process = None
        raise

def cancel_evaluation():
    global current_eval_process
    
    try:
        if not is_eval_running():
            return False, '没有正在运行的评估进程'
        
        with open(EVALUATION_LOG, 'a') as f:
            f.write("[INFO] 正在取消评估进程...\n")
        
        current_eval_process.terminate()
        current_eval_process.join(timeout=2)
        
        with open(EVALUATION_LOG, 'a') as f:
            f.write("[INFO] 评估进程已被用户取消\n")
        
        current_eval_process = None
        
        return True, '评估进程已成功取消'
    except Exception as e:
        error_msg = f"取消评估失败: {str(e)}"
        log_error(error_msg, "backend")
        return False, error_msg