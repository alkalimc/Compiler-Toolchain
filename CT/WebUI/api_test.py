import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import sys
import os
from datetime import datetime
import traceback
import time
from pathlib import Path
import contextlib
import re

LOG_DIR = "/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain/CT/WebUI/logs"
ERROR_LOG = os.path.join(LOG_DIR, "quantization_errors.log")
PROGRESS_LOG = os.path.join(LOG_DIR, "quantization_progress.log") 
EVALUATION_LOG = os.path.join(LOG_DIR, "evaluation_progress.log")
current_eval_process = None
PORT = 7678
HOST = '10.20.108.87'
current_quant_process = None

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

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Authorization", "Content-Type"]
    }
})
auth = HTTPBasicAuth()

setup_logging()

users = {
    "admin": generate_password_hash("yuhaolab.CT"),  
    "user": generate_password_hash("yuhaolab.CT")
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

quantization_params = {
    "model_name": None,
    "precision": 4  # 默认精度
}

def quantification_entrypoint(model_id, log_path, is_vl_model):
    """
    后端包装函数，用于捕获quantification输出并写入日志。
    """
    try:
        os.makedirs("/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain/CT/WebUI/gptq_log", exist_ok=True)
        os.chdir("/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain/CT/WebUI/gptq_log")
    
        with open(log_path, 'a') as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                # 动态导入，避免多进程冲突
                sys.path.append("/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain/CT/Example/Quantification")
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

def evaluation_entrypoint(model_name, eval_method, log_path, is_quantized=False):
    """
    评估任务入口：根据是否量化，调用不同的路径与模块。
    """
    try:
        with open(log_path, 'a') as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):

                if is_quantized:
                    sys.path.insert(0, "/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain/CT/Example/Evaluation/GPTQ")
                    print(f"[INFO] 使用 GPTQ 评估模块评估量化模型: {model_name}")
                else:
                    sys.path.insert(0, "/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain/CT/Example/Evaluation")
                    print(f"[INFO] 使用原始评估模块评估模型: {model_name}")

                # 模块导入根据评估方法切换
                if eval_method in ["humaneval", "mbpp"]:
                    from evalPlus import simpleEvaluation
                else:
                    from lmEvaluationHarness import simpleEvaluation

                # 调用评估函数
                simpleEvaluation(model_id=model_name, evaluation_task=eval_method)
                print(f"[INFO] 评估完成: {model_name} ({eval_method})")

    except Exception as e:
        with open(log_path, 'a') as f:
            f.write(f"[ERROR] 评估异常: {e}\n")


def is_eval_running():
    global current_eval_process
    return current_eval_process is not None and current_eval_process.is_alive()

def run_evaluation(model_name, eval_method, is_quantized=False):
    """
    启动评估进程，根据是否量化，选择正确路径和 model_id。
    """
    global current_eval_process

    try:
        with open(EVALUATION_LOG, 'w') as f:
            f.write("")

        log_error(f"准备评估: {model_name}, 量化模型: {is_quantized}", "eval")

        # 对量化模型进行 ID 补全
        full_model_id = model_name
        if is_quantized:
            full_model_id = f"{model_name}-W4A16-gptq"

        current_eval_process = multiprocessing.Process(
            target=evaluation_entrypoint,
            args=(full_model_id, eval_method, EVALUATION_LOG, is_quantized)
        )
        current_eval_process.start()

        return current_eval_process.pid

    except Exception as e:
        error_msg = f"评估失败 - 模型:{model_name} 方法:{eval_method} 错误:{traceback.format_exc()}"
        log_error(error_msg, "eval")
        current_eval_process = None
        raise


@app.route('/api', methods=['GET'])
@auth.login_required
def get_api():
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

@app.route('/api', methods=['POST'])
@auth.login_required
def post_api():
    global quantization_params, current_eval_process
    
    try:
        data = request.get_json()
        log_error(f"收到请求数据: {str(data)}", "backend")

        # 1. 处理评估启动请求
        if data.get("start_evaluation"):
            if not all(k in data for k in ["model_name", "eval_method"]):
                error_msg = "缺少模型名称或评估方法参数"
                log_error(error_msg, "backend")
                return jsonify({'success': False, 'message': error_msg}), 400
            
            try:
                # 如果已有评估进程在运行，先终止
                if is_eval_running():
                    current_eval_process.terminate()
                    time.sleep(1)
                
                # 启动新评估进程
                is_quantized = data.get("is_quantized", False)
                pid = run_evaluation(data["model_name"], data["eval_method"], is_quantized=is_quantized)
                
                return jsonify({
                    'success': True,
                    'message': '评估进程已启动',
                    'pid': pid,
                    'eval_method': data["eval_method"]
                })
            except Exception as e:
                error_msg = f"评估进程启动失败: {str(e)}"
                log_error(error_msg, "backend")
                return jsonify({'success': False, 'message': error_msg}), 500

        # 1. 处理参数获取请求
        if data.get("action") == "get_quantization_params":
            return jsonify(quantization_params)

        # 2. 处理量化启动请求
        if data.get("start_quantization"):
            if "model_name" not in data:
                error_msg = "缺少模型名称参数"
                log_error(error_msg, "backend")
                return jsonify({'success': False, 'message': error_msg}), 400
            
            try:
                # 如果已有量化进程在运行，先终止
                if is_quant_running():
                    current_quant_process.terminate()
                    time.sleep(1)  # 等待进程终止
                
                # 启动新进程
                pid = run_quantification(data["model_name"])
                
                log_error(f"已启动量化进程 PID: {pid}", "backend")
                return jsonify({
                    'success': True,
                    'message': '量化进程已启动',
                    'pid': pid,
                    'current_params': {
                        'model_name': data["model_name"]
                    }
                })
            except Exception as e:
                error_msg = f"量化进程启动失败: {str(e)}"
                log_error(error_msg, "backend")
                return jsonify({'success': False, 'message': error_msg}), 500
    
        # 3. 处理普通参数更新
        if "model_name" in data:
            quantization_params["model_name"] = data["model_name"]
            log_error(f"更新模型名称: {data['model_name']}", "backend")
        
        # if "precision" in data:
        #     quantization_params["precision"] = int(data["precision"])
        #     log_error(f"更新量化精度: {data['precision']}bit", "backend")

        return jsonify({
            'success': True,
            'message': '参数更新成功',
            'current_params': {
                'model_name': quantization_params["model_name"],
                # 'precision': quantization_params["precision"]  # 
            }
        })

    except Exception as e:
        error_msg = f"POST接口处理异常: {traceback.format_exc()}"
        log_error(error_msg, "backend")
        return jsonify({'success': False, 'message': '服务器内部错误'}), 500

@app.route('/api/log_client_error', methods=['POST'])
def log_client_error():
    """接收前端错误日志"""
    try:
        data = request.get_json()
        log_error(data.get("message", "未知前端错误"), "frontend")
        return jsonify({'success': True})
    except Exception as e:
        log_error(f"前端日志接口错误: {str(e)}", "backend")
        return jsonify({'success': False}), 500
    
@app.route('/api/verify', methods=['GET'])
@auth.login_required
def verify_auth():
    """验证用户身份（前端登录用）"""
    try:
        return jsonify({
            'success': True,
            'message': '认证成功',
            'user': auth.current_user()
        })
    except Exception as e:
        log_error(f"认证接口错误: {str(e)}", "backend")
        return jsonify({'success': False, 'message': '认证失败'}), 401

def remove_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

@app.route('/api/progress', methods=['GET'])
@auth.login_required
def get_progress():
    """获取当前量化进度"""
    try:
        if not os.path.exists(PROGRESS_LOG):
            return jsonify({
                'success': False, 
                'message': '进度文件不存在',
                'is_running': False
            }), 404
        
        # 读取最后50行进度日志
        with open(PROGRESS_LOG, 'r') as f:
            lines = f.readlines()[-50:]  # 获取最后50行
        
        # 过滤 ANSI 转义字符
        clean_lines = [remove_ansi_codes(line.strip()) for line in lines if line.strip()]
        
        return jsonify({
            'success': True,
            'progress': clean_lines,
            'is_running': is_quant_running()
        })
    except Exception as e:
        log_error(f"获取进度失败: {str(e)}", "backend")
        return jsonify({
            'success': False, 
            'message': '获取进度失败',
            'is_running': False
        }), 500
    
@app.route('/api/eval_progress', methods=['GET'])
@auth.login_required
def get_eval_progress():
    """获取评估进度（类似get_progress）"""
    try:
        if not os.path.exists(EVALUATION_LOG):
            return jsonify({
                'success': False, 
                'message': '评估日志不存在',
                'is_running': False
            }), 404
        
        with open(EVALUATION_LOG, 'r') as f:
            lines = f.readlines()[-50:]
        
        clean_lines = [remove_ansi_codes(line.strip()) for line in lines if line.strip()]
        
        return jsonify({
            'success': True,
            'progress': clean_lines,
            'is_running': is_eval_running()
        })
    except Exception as e:
        log_error(f"获取评估进度失败: {str(e)}", "backend")
        return jsonify({
            'success': False, 
            'message': '获取评估进度失败',
            'is_running': False
        }), 500

@app.route('/api/cancel_quant', methods=['POST'])
@auth.login_required
def cancel_quantization():
    global current_quant_process
    
    try:
        if not is_quant_running():
            return jsonify({
                'success': False,
                'message': '没有正在运行的量化进程'
            }), 400
        
        # 先写入日志再终止进程
        with open(PROGRESS_LOG, 'a') as f:
            f.write("[INFO] 正在取消量化进程...\n")
        
        current_quant_process.terminate()
        current_quant_process.join(timeout=2)  # 设置超时
        
        # 确认进程已终止后再记录
        with open(PROGRESS_LOG, 'a') as f:
            f.write("[INFO] 量化进程已被用户取消\n")
        
        current_quant_process = None
        
        return jsonify({
            'success': True,
            'message': '量化进程已成功取消'
        })
    except Exception as e:
        error_msg = f"取消量化失败: {str(e)}"
        log_error(error_msg, "backend")
        return jsonify({
            'success': False,
            'message': error_msg
        }), 500
    
@app.route('/api/cancel_eval', methods=['POST'])
@auth.login_required
def cancel_evaluation():
    global current_eval_process
    
    try:
        if not is_eval_running():
            return jsonify({
                'success': False,
                'message': '没有正在运行的评估进程'
            }), 400
        
        with open(EVALUATION_LOG, 'a') as f:
            f.write("[INFO] 正在取消评估进程...\n")
        
        current_eval_process.terminate()
        current_eval_process.join(timeout=2)
        
        with open(EVALUATION_LOG, 'a') as f:
            f.write("[INFO] 评估进程已被用户取消\n")
        
        current_eval_process = None
        
        return jsonify({
            'success': True,
            'message': '评估进程已成功取消'
        })
    except Exception as e:
        error_msg = f"取消评估失败: {str(e)}"
        log_error(error_msg, "backend")
        return jsonify({
            'success': False,
            'message': error_msg
        }), 500

if __name__ == '__main__':
    print(f'''
    服务器已启动！
    - GET 测试: http://{HOST}:{PORT}/api
    - 进度获取: http://{HOST}:{PORT}/api/progress
    - POST 测试需使用 Postman 或前端调用
    - 错误日志路径: {ERROR_LOG}
    - 进度日志路径: {PROGRESS_LOG}
    ''')
    app.run(host=HOST, port=PORT)