import logging
from openai import OpenAI
from domain.interfaces import ILLMService
from core.config import settings
from core.exceptions import LLMGenerationError

logger = logging.getLogger(__name__)

class QwenLLMService(ILLMService):
    def __init__(self):
        # Mô hình phân hệ Qwen mới chạy trên API tương thích hoàn toàn cấu trúc OpenAI client
        self.client = OpenAI(
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL
        )
        self.model = settings.QWEN_MODEL_NAME

    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        logger.info(f"Đang gửi request sinh văn bản tới Alibaba Qwen model: {self.model}")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Lỗi kết nối Qwen API Dashboard: {str(e)}")
            raise LLMGenerationError(f"Qwen service failed: {str(e)}")