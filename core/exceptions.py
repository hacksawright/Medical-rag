class MedicalRAGException(Exception):
    """Base Exception cho toàn bộ hệ thống"""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class TranslationError(MedicalRAGException):
    """Lỗi xảy ra trong quá trình dịch thuật thuật ngữ y tế"""
    pass

class VectorStoreError(MedicalRAGException):
    """Lỗi liên quan đến thao tác lưu trữ/truy vấn ChromaDB"""
    pass

class LLMGenerationError(MedicalRAGException):
    """Lỗi khi gọi API sinh văn bản của LLM Providers"""
    pass