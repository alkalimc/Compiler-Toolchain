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

WORKSPACE_ROOT = "/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain"
sys.path.insert(0, WORKSPACE_ROOT)
os.chdir(WORKSPACE_ROOT)
LOG_DIR = os.path.join(WORKSPACE_ROOT, "CT", "WebUI", "logs")
ERROR_LOG = os.path.join(LOG_DIR, "quantization_errors.log")
PROGRESS_LOG = os.path.join(LOG_DIR, "quantization_progress.log") 
EVALUATION_LOG = os.path.join(LOG_DIR, "evaluation_progress.log")
DEPLOYMENT_LOG = os.path.join(LOG_DIR, "deployment_progress.log")
COMPILATION_LOG = os.path.join(LOG_DIR, "compilation_progress.log")
QUANT_MODEL_DIR = os.path.join(WORKSPACE_ROOT, "..", "Models", "Quanted")
PORT = 7678
HOST = '10.20.108.87'
current_quant_process = None
current_eval_process = None
current_deploy_process = None
current_compile_process = None

def setup_logging():
    """初始化日志目录和文件"""
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(ERROR_LOG):
        with open(ERROR_LOG, 'w') as f:
            f.write("====== Error Log ======\n")
    Path(PROGRESS_LOG).touch(exist_ok=True)
    Path(EVALUATION_LOG).touch(exist_ok=True)
    Path(DEPLOYMENT_LOG).touch(exist_ok=True)
    Path(COMPILATION_LOG).touch(exist_ok=True)

def log_error(error_msg, source="backend"):
    """
    记录错误到日志文件
    :param error_msg: 错误信息
    :param source: 错误来源 
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
    "precision": 4 
}

def quantification_entrypoint(model_id, log_path, is_vl_model):
    try:
        gptq_log_dir = os.path.join(WORKSPACE_ROOT, "CT", "WebUI", "gptq_log")
        os.makedirs(gptq_log_dir, exist_ok=True)
        os.chdir(gptq_log_dir)
    
        with open(log_path, 'a') as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                if is_vl_model:
                    from CT.Example.Quantization.qwenVLQuantization import simpleQuantization
                    print(f"[INFO] 启动VL模型量化: {model_id}")
                else:
                    from CT.Example.Quantization.quantization import simpleQuantization
                    print(f"[INFO] 启动普通模型量化: {model_id}")
                    
                simpleQuantization(model_id)

                print(f"[INFO] 模型量化完成: {model_id}")

    except Exception as e:
        with open(log_path, 'a') as f:
            f.write(f"[ERROR] 模型量化异常: {e}\n")

def is_quant_running():
    global current_quant_process
    return current_quant_process is not None and current_quant_process.is_alive()

def run_quantification(model_name):
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

def evaluation_entrypoint(model_name, eval_method, eval_tasks, log_path, is_quantized=False):
    try:
        with open(log_path, 'a') as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                if is_quantized:
                    print(f"[INFO] 使用 GPTQ 评估模块评估量化模型: {model_name}")
                    if eval_method == "evalPlus":
                        from CT.Example.Evaluation.GPTQ.evalPlus import simpleEvaluation
                    else:  
                        from CT.Example.Evaluation.GPTQ.lmEvaluationHarness import simpleEvaluation
                else:
                    print(f"[INFO] 使用原始评估模块评估模型: {model_name}")
                    if eval_method == "evalPlus":
                        from CT.Example.Evaluation.FP16.evalPlus import simpleEvaluation
                    else: 
                        from CT.Example.Evaluation.FP16.lmEvaluationHarness import simpleEvaluation

                # 遍历执行所有选中的评估任务
                for task in eval_tasks:
                    print(f"[INFO] 开始评估任务: {task}")
                    simpleEvaluation(model_id=model_name, evaluation_task=task)
                    print(f"[INFO] 评估任务完成: {task}")

                print(f"[INFO] 所有评估任务完成: {model_name}")

    except Exception as e:
        with open(log_path, 'a') as f:
            f.write(f"[ERROR] 评估异常: {e}\n")

def is_eval_running():
    global current_eval_process
    return current_eval_process is not None and current_eval_process.is_alive()

def run_evaluation(model_name, eval_method, eval_tasks, is_quantized=False):
    global current_eval_process

    try:
        with open(EVALUATION_LOG, 'w') as f:
            f.write("")

        log_error(f"准备评估: {model_name}, 量化模型: {is_quantized}, 任务: {eval_tasks}", "eval")

        full_model_id = model_name
        if is_quantized:
            full_model_id = f"{model_name}-W4A16-gptq"
        current_eval_process = multiprocessing.Process(
            target=evaluation_entrypoint,
            args=(full_model_id, eval_method, eval_tasks, EVALUATION_LOG, is_quantized)
        )
        current_eval_process.start()

        return current_eval_process.pid

    except Exception as e:
        error_msg = f"评估失败 - 模型:{model_name} 方法:{eval_method} 错误:{traceback.format_exc()}"
        log_error(error_msg, "eval")
        current_eval_process = None
        raise

def deployment_entrypoint(model_name, log_path):
    try:
        with open(log_path, 'a') as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                print(f"[INFO] 启动模型部署: {model_name}")
                from CT.Example.Deployment.gptqDeployment import simpleDeployment
                simpleDeployment(model_name)
                print(f"[INFO] 模型部署完成: {model_name}")
    except Exception as e:
        with open(log_path, 'a') as f:
            f.write(f"[ERROR] 部署异常: {e}\n")

def is_deploy_running():
    global current_deploy_process
    return current_deploy_process is not None and current_deploy_process.is_alive()

def run_deployment(model_name):
    global current_deploy_process

    try:
        with open(DEPLOYMENT_LOG, 'w') as f:
            f.write("")

        log_error(f"开始部署模型: {model_name}", "deploy")

        current_deploy_process = multiprocessing.Process(
            target=deployment_entrypoint,
            args=(model_name, DEPLOYMENT_LOG)
        )
        current_deploy_process.start()

        return current_deploy_process.pid

    except Exception as e:
        error_msg = f"部署启动失败 - 模型:{model_name} 错误:{traceback.format_exc()}"
        log_error(error_msg, "deploy")
        current_deploy_process = None
        raise

def compile_entrypoint(model_name, log_path):
    try:
        from CT.Example.Compile.compile import simpleCompile

        with open(log_path, 'a') as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                print(f"[INFO] 启动模型编译流程: {model_name}")
                model_path = os.path.join(QUANT_MODEL_DIR, f"{model_name}-W4A16-gptq")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"找不到模型路径: {model_path}")

                simpleCompile(model_path)

                print(f"[INFO] 编译任务完成: {model_name}")

    except Exception as e:
        with open(log_path, 'a') as f:
            f.write(f"[ERROR] 编译异常: {e}\n")

def is_compile_running():
    global current_compile_process
    return current_compile_process is not None and current_compile_process.is_alive()

def run_compilation(model_name):
    global current_compile_process

    try:
        with open(COMPILATION_LOG, 'w') as f:
            f.write("")

        log_error(f"准备编译模型: {model_name}", "compile")

        current_compile_process = multiprocessing.Process(
            target=compile_entrypoint,
            args=(model_name, COMPILATION_LOG)
        )
        current_compile_process.start()

        return current_compile_process.pid

    except Exception as e:
        error_msg = f"编译失败 - 模型:{model_name} 错误:{traceback.format_exc()}"
        log_error(error_msg, "compile")
        current_compile_process = None
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

        if data.get("action") == "get_quantization_params":
            return jsonify(quantization_params)

        # 评估启动
        if data.get("start_evaluation"):
            required_fields = ["model_name", "eval_method", "eval_tasks"]
            if not all(k in data for k in required_fields):
                error_msg = f"缺少必要参数，需要: {required_fields}"
                log_error(error_msg, "backend")
                return jsonify({'success': False, 'message': error_msg}), 400
            
            try:
                if is_eval_running():
                    current_eval_process.terminate()
                    time.sleep(1)
                
                is_quantized = data.get("is_quantized", False)
                pid = run_evaluation(
                    data["model_name"], 
                    data["eval_method"],
                    data["eval_tasks"],
                    is_quantized=is_quantized
                )
                
                return jsonify({
                    'success': True,
                    'message': '评估进程已启动',
                    'pid': pid,
                    'eval_method': data["eval_method"],
                    'eval_tasks': data["eval_tasks"]  
                })
            except Exception as e:
                error_msg = f"评估进程启动失败: {str(e)}"
                log_error(error_msg, "backend")
                return jsonify({'success': False, 'message': error_msg}), 500

        # 量化启动
        if data.get("start_quantization"):
            if "model_name" not in data:
                error_msg = "缺少模型名称参数"
                log_error(error_msg, "backend")
                return jsonify({'success': False, 'message': error_msg}), 400
            
            try:
                if is_quant_running():
                    current_quant_process.terminate()
                    time.sleep(1) 
                
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
            
        # 部署启动
        if data.get("start_deployment"):
            if "model_name" not in data:
                error_msg = "缺少模型名称参数"
                log_error(error_msg, "backend")
                return jsonify({'success': False, 'message': error_msg}), 400

            try:
                if is_deploy_running():
                    current_deploy_process.terminate()
                    time.sleep(1)

                pid = run_deployment(data["model_name"])

                return jsonify({
                    'success': True,
                    'message': '部署进程已启动',
                    'pid': pid
                })
            except Exception as e:
                error_msg = f"部署进程启动失败: {str(e)}"
                log_error(error_msg, "backend")
                return jsonify({'success': False, 'message': error_msg}), 500
            
        # 编译启动
        if data.get("start_compilation"):
            if "model_name" not in data:
                error_msg = "缺少模型名称参数"
                log_error(error_msg, "backend")
                return jsonify({'success': False, 'message': error_msg}), 400

            try:
                if is_compile_running():
                    current_compile_process.terminate()
                    time.sleep(1)

                pid = run_compilation(data["model_name"])

                return jsonify({
                    'success': True,
                    'message': '编译进程已启动',
                    'pid': pid
                })
            except Exception as e:
                error_msg = f"编译进程启动失败: {str(e)}"
                log_error(error_msg, "backend")
                return jsonify({'success': False, 'message': error_msg}), 500
    
        if "model_name" in data:
            quantization_params["model_name"] = data["model_name"]
            log_error(f"更新模型名称: {data['model_name']}", "backend")
        
      # if "precision" in data:
      # quantization_params["precision"] = int(data["precision"])
      # log_error(f"更新量化精度: {data['precision']}bit", "backend")

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
    """获取量化进度"""
    try:
        if not os.path.exists(PROGRESS_LOG):
            return jsonify({
                'success': False, 
                'message': '进度文件不存在',
                'is_running': False
            }), 404
        
        with open(PROGRESS_LOG, 'r') as f:
            lines = f.readlines()[-150:]  
        
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
    """获取评估进度"""
    try:
        if not os.path.exists(EVALUATION_LOG):
            return jsonify({
                'success': False, 
                'message': '评估日志不存在',
                'is_running': False
            }), 404
        
        with open(EVALUATION_LOG, 'r') as f:
            lines = f.readlines()[-100:]
        
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

@app.route('/api/deploy_progress', methods=['GET'])
@auth.login_required
def get_deploy_progress():
    """获取部署进度"""
    try:
        if not os.path.exists(DEPLOYMENT_LOG):
            return jsonify({'success': False, 'message': '部署日志文件不存在'}), 404

        with open(DEPLOYMENT_LOG, 'r') as f:
            lines = f.readlines()
        last_lines = lines[-50:]  
        clean_lines = [remove_ansi_codes(line) for line in last_lines]

        running = is_deploy_running()
        return jsonify({
            'success': True,
            'is_running': running,
            'logs': clean_lines
        })

    except Exception as e:
        error_msg = f"获取部署日志失败: {str(e)}"
        log_error(error_msg, "deploy")
        return jsonify({'success': False, 'message': '服务器错误'}), 500
    
@app.route('/api/compile_progress', methods=['GET'])
@auth.login_required
def get_compile_progress():
    """获取编译进度"""
    try:
        if not os.path.exists(COMPILATION_LOG):
            return jsonify({
                'success': False,
                'message': '编译日志不存在',
                'is_running': False
            }), 404

        with open(COMPILATION_LOG, 'r') as f:
            lines = f.readlines()[-50:]

        clean_lines = [remove_ansi_codes(line.strip()) for line in lines if line.strip()]

        return jsonify({
            'success': True,
            'progress': clean_lines,
            'is_running': is_compile_running()
        })
    except Exception as e:
        log_error(f"获取编译进度失败: {str(e)}", "backend")
        return jsonify({
            'success': False,
            'message': '获取编译进度失败',
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
        
        with open(PROGRESS_LOG, 'a') as f:
            f.write("[INFO] 正在取消量化进程...\n")
        
        current_quant_process.terminate()
        current_quant_process.join(timeout=2) 
        
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

@app.route('/api/cancel_deployment', methods=['POST'])
@auth.login_required
def cancel_deployment():
    global current_deploy_process
    try:
        if is_deploy_running():
            current_deploy_process.terminate()
            current_deploy_process = None
            log_error("部署任务被用户终止", "deploy")
            return jsonify({'success': True, 'message': '部署进程已取消'})
        else:
            return jsonify({'success': False, 'message': '没有正在运行的部署任务'})
    except Exception as e:
        log_error(f"取消部署进程失败: {str(e)}", "deploy")
        return jsonify({'success': False, 'message': '取消部署失败'}), 500

@app.route('/api/cancel_compile', methods=['POST'])
@auth.login_required
def cancel_compilation():
    global current_compile_process

    try:
        if not is_compile_running():
            return jsonify({
                'success': False,
                'message': '没有正在运行的编译进程'
            }), 400

        with open(COMPILATION_LOG, 'a') as f:
            f.write("[INFO] 正在取消编译进程...\n")

        current_compile_process.terminate()
        current_compile_process.join(timeout=2)

        with open(COMPILATION_LOG, 'a') as f:
            f.write("[INFO] 编译进程已被用户取消\n")

        current_compile_process = None

        return jsonify({
            'success': True,
            'message': '编译进程已成功取消'
        })

    except Exception as e:
        error_msg = f"取消编译失败: {str(e)}"
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
    - 量化日志路径: {PROGRESS_LOG}
    - 评估日志路径: {EVALUATION_LOG}
    - 部署日志路径: {DEPLOYMENT_LOG}
    - 编译日志路径: {COMPILATION_LOG}
    ''')
    app.run(host=HOST, port=PORT)