from flask import jsonify, request
import re
from .auth import auth
from .quantization import (
    is_quant_running, run_quantification, cancel_quantization,
    current_quant_process
)
from .evaluation import (
    is_eval_running, run_evaluation, cancel_evaluation,
    current_eval_process
)
from .config import (
    PROGRESS_LOG, EVALUATION_LOG, DEFAULT_QUANTIZATION_PARAMS
)
from .logging_utils import log_error

quantization_params = DEFAULT_QUANTIZATION_PARAMS.copy()

def remove_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def get_progress_log(log_file, is_running_func):
    """通用进度日志获取函数"""
    try:
        if not os.path.exists(log_file):
            return {
                'success': False, 
                'message': '进度文件不存在',
                'is_running': False
            }
        
        # 读取最后50行进度日志
        with open(log_file, 'r') as f:
            lines = f.readlines()[-50:]  # 获取最后50行
        
        # 过滤 ANSI 转义字符
        clean_lines = [remove_ansi_codes(line.strip()) for line in lines if line.strip()]
        
        return {
            'success': True,
            'progress': clean_lines,
            'is_running': is_running_func()
        }
    except Exception as e:
        log_error(f"获取进度失败: {str(e)}", "backend")
        return {
            'success': False, 
            'message': '获取进度失败',
            'is_running': False
        }

def handle_get_api():
    """GET接口返回当前状态"""
    try:
        return f'''
        <h1>API 服务已运行</h1>
        <p>模型: {quantization_params["model_name"] or "未设置"}</p>
        <p>精度: {quantization_params["precision"]}bit</p>
        <p>用户: {auth.current_user()}</p>
        '''
    except Exception as e:
        log_error(f"GET接口错误: {str(e)}", "backend")
        return jsonify({'success': False, 'message': '服务异常'}), 500

def handle_post_api():
    global quantization_params
    
    try:
        data = request.get_json()
        log_error(f"收到请求数据: {str(data)}", "backend")

        # 1. 处理评估启动请求
        if data.get("start_evaluation"):
            if not all(k in data for k in ["model_name", "eval_method"]):
                error_msg = "缺少模型名称或评估方法参数"
                log_error(error_msg, "backend")
                return {'success': False, 'message': error_msg}, 400
            
            try:
                # 如果已有评估进程在运行，先终止
                if is_eval_running():
                    current_eval_process.terminate()
                    time.sleep(1)
                
                # 启动新评估进程
                pid = run_evaluation(data["model_name"], data["eval_method"])
                
                return {
                    'success': True,
                    'message': '评估进程已启动',
                    'pid': pid,
                    'eval_method': data["eval_method"]
                }
            except Exception as e:
                error_msg = f"评估进程启动失败: {str(e)}"
                log_error(error_msg, "backend")
                return {'success': False, 'message': error_msg}, 500

        # 2. 处理参数获取请求
        if data.get("action") == "get_quantization_params":
            return quantization_params

        # 3. 处理量化启动请求
        if data.get("start_quantization"):
            if "model_name" not in data:
                error_msg = "缺少模型名称参数"
                log_error(error_msg, "backend")
                return {'success': False, 'message': error_msg}, 400
            
            try:
                # 如果已有量化进程在运行，先终止
                if is_quant_running():
                    current_quant_process.terminate()
                    time.sleep(1)  # 等待进程终止
                
                # 启动新进程
                pid = run_quantification(data["model_name"])
                
                log_error(f"已启动量化进程 PID: {pid}", "backend")
                return {
                    'success': True,
                    'message': '量化进程已启动',
                    'pid': pid,
                    'current_params': {
                        'model_name': data["model_name"]
                    }
                }
            except Exception as e:
                error_msg = f"量化进程启动失败: {str(e)}"
                log_error(error_msg, "backend")
                return {'success': False, 'message': error_msg}, 500
    
        # 4. 处理普通参数更新
        if "model_name" in data:
            quantization_params["model_name"] = data["model_name"]
            log_error(f"更新模型名称: {data['model_name']}", "backend")

        return {
            'success': True,
            'message': '参数更新成功',
            'current_params': {
                'model_name': quantization_params["model_name"],
            }
        }

    except Exception as e:
        error_msg = f"POST接口处理异常: {traceback.format_exc()}"
        log_error(error_msg, "backend")
        return {'success': False, 'message': '服务器内部错误'}, 500

def handle_log_client_error():
    """接收前端错误日志"""
    try:
        data = request.get_json()
        log_error(data.get("message", "未知前端错误"), "frontend")
        return {'success': True}
    except Exception as e:
        log_error(f"前端日志接口错误: {str(e)}", "backend")
        return {'success': False}, 500

def handle_verify_auth():
    """验证用户身份（前端登录用）"""
    try:
        return {
            'success': True,
            'message': '认证成功',
            'user': auth.current_user()
        }
    except Exception as e:
        log_error(f"认证接口错误: {str(e)}", "backend")
        return {'success': False, 'message': '认证失败'}, 401