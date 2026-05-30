from core.config import settings
from prompts.medical_prompt import MEDICAL_SYSTEM_PROMPT, MEDICAL_SYSTEM_PROMPT_EN
from infrastructure.language_detector import LinguaLanguageDetector
from infrastructure.translator import EnViT5Translator
from infrastructure.embeddings import BGEM3EmbeddingService
from infrastructure.vectorstore import ChromaVectorStore
from infrastructure.llm.gemini_service import GeminiLLMService
from infrastructure.llm.qwen_service import QwenLLMService
from services.rag_pipeline import MedicalRAGPipeline

# Khởi tạo Singletons lưu trữ bộ nhớ dài hạn, tránh nạp lại mô hình AI nhiều lần gây tràn RAM
language_detector_instance = LinguaLanguageDetector()
translator_instance = EnViT5Translator()
embedding_service_instance = BGEM3EmbeddingService()
vector_store_instance = ChromaVectorStore()

def get_llm_service():
    provider = settings.LLM_PROVIDER.lower()
    if provider == "gemini":
        return GeminiLLMService()
    elif provider == "qwen":
        return QwenLLMService()
    else:
        raise ValueError(f"Provider LLM không hợp lệ: {provider}")

def get_rag_pipeline() -> MedicalRAGPipeline:
    return MedicalRAGPipeline(
        language_detector=language_detector_instance,
        translator=translator_instance,
        embedding_service=embedding_service_instance,
        vector_store=vector_store_instance,
        llm_service=get_llm_service(),
        system_prompt_template=MEDICAL_SYSTEM_PROMPT,
        system_prompt_template_en=MEDICAL_SYSTEM_PROMPT_EN
    )