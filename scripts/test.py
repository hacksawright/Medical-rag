import os
import sys

# Append thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from core.config import settings
from infrastructure.embeddings import BGEM3EmbeddingService

def test_search(query_text: str, top_k: int = 3):
    print(f"\n=== ĐANG TEST TRUY VẤN: '{query_text}' ===")
    
    # 1. Khởi tạo Embedding Service (Để chuyển câu hỏi test thành Vector)
    embed_service = BGEM3EmbeddingService()
    query_vector = embed_service.embed_query(query_text)
    
    # 2. Kết nối tới ChromaDB vật lý
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
    collection = chroma_client.get_collection(name=settings.CHROMA_COLLECTION_NAME)
    
    # 3. Query thử
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    
    # 4. In kết quả xem độ chính xác
    if not results or not results['documents'] or len(results['documents'][0]) == 0:
        print("❌ Không tìm thấy đoạn văn bản nào phù hợp!")
        return
        
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        metadata = results['metadatas'][0][i] if results['metadatas'] else {}
        # Khoảng cách Cosine trong ChromaDB trả về (1 - similarity), nên lấy 1 trừ đi để ra độ tương đồng
        score = 1.0 - results['distances'][0][i] if results['distances'] else 0.0
        
        print(f"\n📍 [Kết quả {i+1}] - Độ tương đồng (Score): {score:.4f}")
        print(f"📄 Nội dung chunk: {text}")
        print(f"🏷️ Metadata đi kèm: {metadata}")

if __name__ == "__main__":
    # Test thử 1 câu bằng Tiếng Việt
    test_search("cúm là gì và triệu chứng như thế nào?")
    
    # Bạn có thể test thêm các câu khác để xem độ nhạy của Vector Space
    # test_search("Biến chứng nguy hiểm của đái tháo đường")