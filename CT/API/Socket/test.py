import socket
import threading
import time
import json
import requests
import struct

def test_openai_api():
    """测试OpenAI兼容API接口"""
    print("[测试] 等待5秒让服务器启动...")
    time.sleep(5)
    
    # 准备测试数据
    test_data = {
        "model": "glm",
        "messages": [
            {"role": "user", "content": "你好，请介绍一下自己"}
        ],
        "temperature": 0.6,
        "max_tokens": 2048
    }
    
    print("[测试] 发送HTTP请求到OpenAI端点...")
    try:
        # 增加超时时间到60秒
        response = requests.post(
            "http://127.0.0.1:7000/v1/chat/completions",
            json=test_data,
            timeout=60
        )
        
        print(f"[测试] HTTP响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("[测试] 成功收到API响应:")
            # 直接打印原始响应内容以便调试
            print("响应内容:", response.text)
            
            try:
                # 尝试解析JSON
                json_response = response.json()
                print(json.dumps(json_response, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print("[警告] 响应不是有效的JSON格式")
        else:
            print(f"[测试] API错误响应: {response.text}")
            
    except requests.exceptions.Timeout:
        print("[测试] 错误: 请求超时（60秒），请检查服务器处理逻辑")
    except requests.exceptions.RequestException as e:
        print(f"[测试] HTTP请求失败: {str(e)}")

if __name__ == "__main__":
    print("="*50)
    print("H100 Socket通信测试脚本")
    print("="*50)
    
    # 启动API测试线程
    api_thread = threading.Thread(target=test_openai_api)
    api_thread.start()
    
    # 增加主线程等待时间
    api_thread.join(timeout=90)
    
    if api_thread.is_alive():
        print("[测试] 警告: 测试线程仍在运行，可能发生阻塞")
    else:
        print("[测试] 测试完成")
    
    print("\n[测试] 测试完成")