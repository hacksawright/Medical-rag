from abc import ABC, abstractmethod
from typing import List, Dict, Any
from domain.models import MedicalDocumentChunk

class ILanguageDetector(ABC):
    @abstractmethod
    def detect_language(self, text: str) -> str:
        """Trả về mã ngôn ngữ chuẩn ISO ('vi' hoặc 'en')"""
        pass

class ITranslator(ABC):
    @abstractmethod
    def translate_en_to_vi(self, text: str) -> str:
        """Dịch chuỗi văn bản từ tiếng Anh sang tiếng Việt"""
        pass

class IEmbeddingService(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Tạo dense vector cho chuỗi truy vấn câu hỏi"""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Tạo danh sách dense vectors cho việc batch ingestion"""
        pass

class IVectorStore(ABC):
    @abstractmethod
    def query_similar_documents(self, query_vector: List[float], top_k: int) -> List[MedicalDocumentChunk]:
        """Tìm kiếm ngữ cảnh tương đồng gần nhất"""
        pass

    @abstractmethod
    def add_documents(self, chunks: List[MedicalDocumentChunk]):
        """Nạp dữ liệu văn bản kèm vectors vào kho lưu trữ"""
        pass

class ILLMService(ABC):
    @abstractmethod
    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        """Gửi prompt y tế hoàn chỉnh đến LLM và thu về câu trả lời duy nhất"""
        pass