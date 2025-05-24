from pydantic import BaseModel
from typing import List, Dict, Optional

class QwenMessage(BaseModel):
    role: str  #/user/assistant/function
    content: Optional[str] = None
    name: Optional[str] = None  
    function_call: Optional[Dict] = None

class QwenFunctionCall(BaseModel):
    name: str
    arguments: str  

class QwenFunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict  

class QwenRequest(BaseModel):
    messages: List[QwenMessage]
    functions: Optional[List[QwenFunctionDefinition]] = None
    temperature: float = 0.8
    model: str = "qwen2.5-7b" #需要修改

class QwenResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict