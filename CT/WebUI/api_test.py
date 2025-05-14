from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

# 创建Flask应用
app = Flask(__name__)

# 启用跨域支持(CORS)
CORS(app)

# 初始化认证
auth = HTTPBasicAuth()

# 用户数据库（实际项目中应该使用数据库存储）
users = {
    "admin": generate_password_hash("yuhaolab.CT"),  
    "user": generate_password_hash("yuhaolab.CT")
}

# 验证密码的回调函数
@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

# 存储量化参数的全局变量
quantization_params = {
    "model_name": None,
    "precision": 4  # 默认精度
}

# GET /api 路由 - 用于浏览器直接访问
@app.route('/api', methods=['GET'])
@auth.login_required  # 添加认证装饰器
def get_api():
    return '''
    <h1>API 服务已运行</h1>
    <p>这是 GET /api 的响应</p>
    <p>当前量化参数:</p>
    <ul>
      <li>模型: {}</li>
      <li>精度: {}</li>
    </ul>
    <p>你可以：</p>
    <ul>
      <li>用 Postman 发送 POST 请求到 /api 设置参数</li>
      <li>或者在前端代码中调用这个 API</li>
    </ul>
    <p>当前登录用户: {}</p>
    '''.format(quantization_params["model_name"] or "默认模型", 
               quantization_params["precision"],
               auth.current_user())

# POST /api 路由 - 接收前端数据
@app.route('/api', methods=['POST'])
@auth.login_required  # 添加认证装饰器
def post_api():
    global quantization_params
    
    data = request.get_json()
    print('收到 POST 请求数据:', data)
    
    # 处理获取参数的请求
    if data.get("action") == "get_quantization_params":
        return jsonify(quantization_params)
    
    # 处理设置参数的请求
    if "model_name" in data:
        quantization_params["model_name"] = data["model_name"]
    if "precision" in data:
        quantization_params["precision"] = int(data["precision"])
    
    return jsonify({
        'success': True,
        'message': '参数更新成功',
        'current_params': quantization_params,
        'current_user': auth.current_user()  # 返回当前用户名
    })

# 启动服务器
if __name__ == '__main__':
    PORT = 7678
    print(f'''
    服务器已启动！
    - GET 测试: http://10.20.108.87:{PORT}/api
    - POST 测试需使用 Postman 或前端调用
    ''')
    app.run(host='10.20.108.87', port=PORT)