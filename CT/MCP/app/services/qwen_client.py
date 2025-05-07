import os
import httpx
from app.models.qwen_schemas import QwenRequest, QwenResponse
from dotenv import load_dotenv

load_dotenv()

class QwenClient:
    def __init__(self):
        self.base_url = os.getenv("QWEN_API_BASE", "api网址")
        self.api_key = os.getenv("QWEN_API_KEY")
        self.timeout = 30

    async def chat_completion(self, request: QwenRequest) -> QwenResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=request.dict(),
                headers=headers
            )
            response.raise_for_status()
            return QwenResponse(**response.json())