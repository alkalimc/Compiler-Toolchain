import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.function_mgr import function_manager

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_functions():
    # 注册测试函数
    from app.models.function import FunctionDescriptor, FunctionParameter
    
    test_desc = FunctionDescriptor(
        name="test_add",
        description="测试加法函数",
        parameters=[
            FunctionParameter(
                name="a", type="number", description="第一个加数"
            ),
            FunctionParameter(
                name="b", type="number", description="第二个加数"
            )
        ],
        return_type="number"
    )
    
    @function_manager.register(test_desc)
    def test_add(a: float, b: float):
        return a + b

def test_function_call_workflow():
    # 测试完整调用链路
    response = client.post(
        "/v1/chat/completions",
        headers={"X-API-KEY": "key1"},
        json={
            "messages": [{
                "role": "user",
                "content": "请计算3.14加2.71"
            }],
            "model": "qwen2-7b"
        }
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "function_call" in result["choices"][0]["message"]
    
    func_call = result["choices"][0]["message"]["function_call"]
    assert func_call["name"] == "test_add"
    assert "3.14" in func_call["arguments"]
    assert "2.71" in func_call["arguments"]
    
def test_invalid_function_call():
    # 测试无效函数调用
    response = client.post(
        "/v1/chat/completions",
        headers={"X-API-KEY": "key1"},
        json={
            "messages": [{
                "role": "user",
                "content": "请调用不存在函数"
            }],
            "model": "qwen2-7b"
        }
    )
    
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]