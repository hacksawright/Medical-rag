import logging
from typing import List
from sentence_transformers import SentenceTransformer
from domain.interfaces import IEmbeddingService
from core.config import settings

logger = logging.getLogger(__name__)

class BGEM3EmbeddingService(IEmbeddingService):
    def __init__(self):
        logger.info(f"Đang khởi tạo Embedding Model: {settings.EMBEDDING_MODEL_NAME} trên thiết bị: {settings.EMBEDDING_DEVICE}")
        # bge-m3 tự động hỗ trợ tính toán đa ngữ rất mạnh mẽ
        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL_NAME, 
            device=settings.EMBEDDING_DEVICE
        )

    def embed_query(self, text: str) -> List[float]:
        # Đối với BGE-M3, việc chuẩn hóa vector đầu ra giúp tính toán tương đồng bằng Cosine nhanh hơn
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()