import requests
from typing import Optional, Dict
from requests.auth import HTTPBasicAuth

# API 配置
API_URL = "http://10.20.108.87:7678/api"
API_USERNAME = "admin"  
API_PASSWORD = "yuhaolab.CT" 

class QuantizationParams:
    def __init__(self):
        self.model_name: Optional[str] = None
        self.precision: Optional[int] = None
    
    def update(self, data: Dict):
        """更新参数"""
        self.model_name = data.get("model_name")
        self.precision = data.get("precision")
    
    def __str__(self):
        return f"模型: {self.model_name}, 精度: {self.precision}"

def get_quantization_params() -> Optional[Dict]:
    """从API获取量化参数"""
    try:
        response = requests.post(
            API_URL,
            json={"action": "get_quantization_params"},
            auth=HTTPBasicAuth(API_USERNAME, API_PASSWORD)  # 添加Basic Auth
        )
        if response.status_code == 200:
            return response.json()
        print(f"API请求失败，状态码: {response.status_code}, 响应: {response.text}")
        return None
    except Exception as e:
        print(f"API请求错误: {e}")
        return None

def send_params_back_to_api(params: Dict) -> bool:
    """将参数发送回API验证"""
    try:
        response = requests.post(
            API_URL,
            json={**params, "is_validation": True},
            auth=HTTPBasicAuth(API_USERNAME, API_PASSWORD)  # 添加Basic Auth
        )
        if response.status_code == 200:
            print("API验证成功:", response.json())
            return True
        print("API验证失败:", response.status_code, response.text)
        return False
    except Exception as e:
        print(f"API验证错误: {e}")
        return False

def main():
    print("=== 开始测试 ===")
    
    # 1. 从API获取参数并保存到内存
    params = QuantizationParams()
    if api_data := get_quantization_params():
        params.update(api_data)
        print(f"获取参数成功: {params}")
    else:
        print("获取参数失败")
    
    # 2. 将参数发送回API验证
    print("\n发送参数回API验证...")
    send_params_back_to_api({
        "model_name": params.model_name,
        "precision": params.precision
    })

if __name__ == "__main__":
    main()