import logging
from lingua import Language, LanguageDetectorBuilder
from domain.interfaces import ILanguageDetector
from core.config import settings

logger = logging.getLogger(__name__)

class LinguaLanguageDetector(ILanguageDetector):
    def __init__(self):
        # Ta chỉ cần tập trung phân tách chính xác cao giữa Tiếng Việt và Tiếng Anh để tối ưu hiệu năng
        languages = [Language.VIETNAMESE, Language.ENGLISH]
        self.detector = LanguageDetectorBuilder.from_languages(*languages).build()

    def detect_language(self, text: str) -> str:
        try:
            detected_lang = self.detector.detect_language_of(text)
            if detected_lang == Language.ENGLISH:
                return "en"
            elif detected_lang == Language.VIETNAMESE:
                return "vi"
        except Exception as e:
            logger.warning(f"Lỗi nhận diện ngôn ngữ, áp dụng fallback mặc định: {str(e)}")
        return settings.DEFAULT_LANGUAGE