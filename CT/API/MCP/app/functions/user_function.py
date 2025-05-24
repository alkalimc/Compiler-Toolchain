import json
from datetime import datetime
from app.services.function_mgr import function_manager
from app.models.function import FunctionDescriptor


get_weather_desc = FunctionDescriptor(
    name="get_current_weather",
    description="获取指定城市的当前天气信息",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "城市名称，如：'北京'"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            }
        },
        "required": ["location"]
    }
)

@function_manager.register(get_weather_desc)
async def get_current_weather(location: str, unit: str = "celsius"):
    """模拟天气数据查询"""
    return {
        "location": location,
        "temperature": "25.5" if unit == "celsius" else "78.9",
        "unit": unit,
        "report_time": datetime.now().isoformat()
    }