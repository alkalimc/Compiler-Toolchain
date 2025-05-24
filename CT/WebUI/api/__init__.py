import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from flask import Flask, jsonify
from flask_cors import CORS
from .config import HOST, PORT, ERROR_LOG, PROGRESS_LOG
from .logging_utils import setup_logging
from .auth import auth
from .routes import (
    handle_get_api, handle_post_api, handle_log_client_error,
    handle_verify_auth, get_progress_log, remove_ansi_codes
)
from .quantization import is_quant_running
from .evaluation import is_eval_running

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Authorization", "Content-Type"]
    }
})

# 初始化日志
setup_logging()

@app.route('/api', methods=['GET'])
@auth.login_required
def get_api():
    response = handle_get_api()
    if isinstance(response, str):
        return response
    return jsonify(response[0]), response[1]

@app.route('/api', methods=['POST'])
@auth.login_required
def post_api():
    response = handle_post_api()
    return jsonify(response[0]), response[1]

@app.route('/api/log_client_error', methods=['POST'])
def log_client_error():
    response = handle_log_client_error()
    return jsonify(response[0]), response[1]

@app.route('/api/verify', methods=['GET'])
@auth.login_required
def verify_auth():
    response = handle_verify_auth()
    return jsonify(response[0]), response[1]

@app.route('/api/progress', methods=['GET'])
@auth.login_required
def get_progress():
    response = get_progress_log(PROGRESS_LOG, is_quant_running)
    return jsonify(response)

@app.route('/api/eval_progress', methods=['GET'])
@auth.login_required
def get_eval_progress():
    response = get_progress_log(EVALUATION_LOG, is_eval_running)
    return jsonify(response)

@app.route('/api/cancel_quant', methods=['POST'])
@auth.login_required
def cancel_quantization():
    success, message = cancel_quantization()
    return jsonify({'success': success, 'message': message})

@app.route('/api/cancel_eval', methods=['POST'])
@auth.login_required
def cancel_evaluation():
    success, message = cancel_evaluation()
    return jsonify({'success': success, 'message': message})

def run():
    print(f'''
    服务器已启动！
    - GET 测试: http://{HOST}:{PORT}/api
    - 进度获取: http://{HOST}:{PORT}/api/progress
    - POST 测试需使用 Postman 或前端调用
    - 错误日志路径: {ERROR_LOG}
    - 进度日志路径: {PROGRESS_LOG}
    ''')
    app.run(host=HOST, port=PORT)

if __name__ == '__main__':
    run()

from .routes import app
from . import run