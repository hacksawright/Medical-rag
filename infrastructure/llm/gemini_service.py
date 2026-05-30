import logging
import google.generativeai as genai
from domain.interfaces import ILLMService
from core.config import settings
from core.exceptions import LLMGenerationError

logger = logging.getLogger(__name__)

class GeminiLLMService(ILLMService):
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GEMINI_MODEL_NAME

    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        logger.info(f"Đang gửi request sinh văn bản tới Google Gemini model: {self.model_name}")
        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt
            )
            response = model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1024
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Lỗi kết nối Gemini API: {str(e)}")
            raise LLMGenerationError(f"Gemini service failed: {str(e)}")