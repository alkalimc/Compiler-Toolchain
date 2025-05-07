from fastapi import APIRouter, Depends, HTTPException
from app.models.qwen_schemas import QwenRequest, QwenResponse
from app.services.qwen_client import QwenClient
from app.services.function_mgr import function_manager
from app.utils.auth import verify_api_key

router = APIRouter()

@router.post("api网址", dependencies=[Depends(verify_api_key)])
async def chat_completion(request: QwenRequest):
    
    if request.functions is None:
        request.functions = [
            desc.to_qwen_format() 
            for desc in function_manager.get_descriptors()
        ]
    
    client = QwenClient()
    raw_response = await client.chat_completion(request)
    
    
    if raw_response.choices[0].message.function_call:
        await execute_function_call(raw_response)
    
    return raw_response

async def execute_function_call(response: QwenResponse):
    message = response.choices[0].message
    func_name = message.function_call.name
    arguments = json.loads(message.function_call.arguments)
    
    func = function_manager.get_function(func_name)
    if not func:
        raise HTTPException(
            status_code=400,
            detail=f"Function {func_name} not found"
        )
    
    
    result = await func(**arguments)
    message.content = f"Function call result: {result}"