import logging
import json
from datetime import datetime
from fastapi import Request
from contextvars import ContextVar

request_id = ContextVar("request_id", default="")

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "request_id": request_id.get() or "",
        }
        return json.dumps(log_data)

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    # 文件输出
    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

class FunctionCallFilter(logging.Filter):
    def filter(self, record):
        return "function_call" in record.msg

async def log_middleware(request: Request, call_next):
    import uuid
    rid = str(uuid.uuid4())
    request_id.set(rid)
    
    logger = logging.getLogger("uvicorn.access")
    logger.info(
        "Request started",
        extra={
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host if request.client else None
        }
    )
    
    response = await call_next(request)
    
    logger.info(
        "Request completed",
        extra={
            "status_code": response.status_code,
            "duration": response.headers.get("X-Response-Time")
        }
    )
    return response