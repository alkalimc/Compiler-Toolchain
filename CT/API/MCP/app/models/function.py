
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.json import UUID

class FunctionType(str, Enum):
    """函数类型分类"""
    DATA_QUERY = "data_query"
    SYSTEM_ACTION = "system_action"
    EXTERNAL_API = "external_api"
    CUSTOM_LOGIC = "custom_logic"

class FunctionParameter(BaseModel):
    """函数参数详细定义"""
    name: str = Field(..., min_length=1, example="user_id")
    type: str = Field(
        ..., 
        description="参数类型，支持JSON Schema类型定义",
        example="string"
    )
    description: Optional[str] = Field(
        None,
        max_length=200,
        example="用户的唯一标识符"
    )
    enum: Optional[List[str]] = Field(
        None,
        description="可选值列表（仅当type为enum时有效）",
        example=["male", "female"]
    )
    required: bool = Field(
        True,
        description="是否必须参数"
    )
    default: Optional[Union[str, int, bool]] = Field(
        None,
        description="默认值（仅当required=False时有效）"
    )

    @root_validator
    def check_required_constraint(cls, values):
        if values.get('required') and values.get('default') is not None:
            raise ValueError("Required parameters cannot have default values")
        return values

class FunctionDescriptor(BaseModel):
    """函数能力元数据描述"""
    name: str = Field(
        ..., 
        min_length=3,
        regex=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
        example="get_user_profile",
        description="函数名称（符合编程语言函数命名规范）"
    )
    description: str = Field(
        ..., 
        min_length=10,
        max_length=500,
        example="获取用户详细信息，包括联系方式和个人偏好"
    )
    parameters: List[FunctionParameter] = Field(
        ...,
        min_items=1,
        description="函数参数规范列表"
    )
    return_type: str = Field(
        ..., 
        example="UserProfile",
        description="返回值的类型描述"
    )
    function_type: FunctionType = Field(
        FunctionType.DATA_QUERY,
        description="函数分类类型"
    )
    version: str = Field(
        "1.0.0", 
        regex=r"^\d+\.\d+\.\d+$",
        example="1.2.0"
    )
    requires_auth: bool = Field(
        True,
        description="是否需要授权令牌"
    )
    rate_limit: Optional[int] = Field(
        None,
        ge=1,
        example=60,
        description="每分钟最大调用次数"
    )

    def to_qwen_schema(self) -> Dict:
        """转换为Qwen模型要求的函数描述格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description,
                        **({"enum": param.enum} if param.enum else {}),
                        **({"default": param.default} if param.default is not None else {})
                    } for param in self.parameters
                },
                "required": [
                    param.name 
                    for param in self.parameters 
                    if param.required
                ]
            }
        }

class FunctionCallRequest(BaseModel):
    """函数调用请求体"""
    call_id: UUID = Field(
        ..., 
        description="唯一调用标识符（UUID v4）"
    )
    function_name: str = Field(..., example="get_user_profile")
    parameters: Dict[str, Union[str, int, bool]] = Field(
        ..., 
        example={"user_id": "usr_12345"}
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="请求时间戳（UTC）"
    )
    auth_token: Optional[str] = Field(
        None,
        min_length=32,
        description="授权令牌（需要HMAC签名时使用）"
    )

class FunctionCallResult(BaseModel):
    """函数执行结果"""
    success: bool = Field(...)
    data: Optional[Dict] = Field(
        None,
        example={"username": "john_doe", "email": "john@example.com"}
    )
    error_code: Optional[str] = Field(
        None,
        example="PERMISSION_DENIED"
    )
    error_message: Optional[str] = Field(
        None,
        example="Insufficient access rights"
    )
    execution_time_ms: float = Field(
        ..., 
        ge=0,
        example=125.3,
        description="函数执行耗时（毫秒）"
    )

class FunctionCallResponse(BaseModel):
    """函数调用API响应"""
    request_id: UUID = Field(...)
    status: str = Field(
        ..., 
        enum=["pending", "success", "failed"],
        example="success"
    )
    result: Optional[FunctionCallResult] = None
    next_token: Optional[str] = Field(
        None,
        description="分页令牌（用于长耗时操作）"
    )

class FunctionRegistryItem(BaseModel):
    """注册函数完整定义"""
    descriptor: FunctionDescriptor
    endpoint: str = Field(
        ..., 
        example="/internal/api/v1/user_profile",
        description="实际后端服务端点"
    )
    service_name: str = Field(
        ..., 
        example="user-service",
        description="所属微服务名称"
    )
    deprecated: bool = Field(
        False,
        description="是否已弃用"
    )