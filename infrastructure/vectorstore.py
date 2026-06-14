import logging
import chromadb
from typing import List
from domain.interfaces import IVectorStore
from domain.models import MedicalDocumentChunk
from core.config import settings
from core.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

class ChromaVectorStore(IVectorStore):
    def __init__(self):
        try:
            logger.info(f"Đang thiết lập kết nối lưu trữ vật lý tới ChromaDB tại: {settings.CHROMA_DB_DIR}")
            self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
            # Khởi tạo hoặc nạp lại collection hiện có bằng khoảng cách Cosine
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Lỗi khởi tạo ChromaDB: {str(e)}")
            raise VectorStoreError(f"Không thể khởi chạy hệ thống cơ sở dữ liệu Vector: {str(e)}")

    def query_similar_documents(self, query_vector: List[float], top_k: int) -> List[MedicalDocumentChunk]:
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )
            
            chunks = []
            if not results or not results['documents'] or len(results['documents'][0]) == 0:
                return chunks
                
            for i in range(len(results['documents'][0])):
                # Khoảng cách Cosine trong ChromaDB trả về giá trị (1 - cosine_similarity)
                score = 1.0 - results['distances'][0][i] if results['distances'] else None
                chunks.append(MedicalDocumentChunk(
                    id=results['ids'][0][i],
                    text=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    score=score
                ))
            return chunks
        except Exception as e:
            logger.error(f"Lỗi truy vấn Vector từ ChromaDB: {str(e)}")
            raise VectorStoreError(f"Truy vấn ngữ cảnh thất bại: {str(e)}")

    def add_documents(self, chunks: List[MedicalDocumentChunk]):
        if not chunks:
            return
        try:
            ids = [chunk.id for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Lưu ý: Client gọi ngoài phải tự pass embeddings qua pipeline để tối ưu hóa kiến trúc
            # Nhưng ở đây để đảm bảo tương thích, ta giả định embeddings đã được tạo hoặc xử lý tại pipeline
            # Ta sẽ nạp trực tiếp bằng hàm chuyên dụng trong file ingestion script.
            pass
        except Exception as e:
            logger.error(f"Lỗi khi thêm tài liệu vào ChromaDB: {str(e)}")
            raise VectorStoreError(f"Ghi tài liệu y tế thất bại: {str(e)}")

    def get_all_documents(self) -> List[MedicalDocumentChunk]:
        try:
            results = self.collection.get(include=["documents", "metadatas"])
            chunks = []
            if not results or not results['ids']:
                return chunks
                
            for i in range(len(results['ids'])):
                chunks.append(MedicalDocumentChunk(
                    id=results['ids'][i],
                    text=results['documents'][i] if results['documents'] else "",
                    metadata=results['metadatas'][i] if results['metadatas'] else {},
                    score=None
                ))
            return chunks
        except Exception as e:
            logger.error(f"Lỗi khi lấy toàn bộ tài liệu từ ChromaDB: {str(e)}")
            raise VectorStoreError(f"Không thể truy xuất toàn bộ tài liệu y tế: {str(e)}")