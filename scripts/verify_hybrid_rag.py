import os
import sys

# Thêm thư mục gốc dự án vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from infrastructure.vectorstore import ChromaVectorStore
from infrastructure.embeddings import BGEM3EmbeddingService
from infrastructure.retrievers import ChromaRetriever, BM25Retriever, HybridRetriever
from api.dependencies import get_rag_pipeline
from domain.models import ChatRequest

def run_verification():
    print("==============================================================")
    print("BẮT ĐẦU KIỂM TRA HỆ THỐNG HYBRID RETRIEVAL Y TẾ (CLEAN ARCHITECTURE)")
    print("==============================================================")
    
    # 1. Khởi tạo các Singletons
    print("\n1. Đang khởi tạo các services và stores...")
    vector_store = ChromaVectorStore()
    embedding_service = BGEM3EmbeddingService()
    
    # 2. Khởi tạo Retrievers
    print("\n2. Khởi tạo Retrievers...")
    dense_retriever = ChromaRetriever(vector_store, embedding_service)
    sparse_retriever = BM25Retriever(vector_store)
    hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
    
    test_query = "triệu chứng của bệnh cúm"
    print(f"\nTruy vấn kiểm tra: '{test_query}'")
    
    # 3. Test Dense Retrieval
    print("\n--------------------------------------------------------------")
    print("A. KIỂM TRA DENSE RETRIEVAL (Embedding Search qua ChromaDB)")
    print("--------------------------------------------------------------")
    dense_results = dense_retriever.retrieve(test_query, top_k=3)
    for idx, chunk in enumerate(dense_results):
        print(f"[{idx+1}] ID: {chunk.id} | Score (Cosine Sim): {chunk.score:.4f}")
        print(f"    Nội dung: {chunk.text[:120]}...\n")
        
    # 4. Test Sparse Retrieval
    print("\n--------------------------------------------------------------")
    print("B. KIỂM TRA SPARSE RETRIEVAL (BM25)")
    print("--------------------------------------------------------------")
    sparse_results = sparse_retriever.retrieve(test_query, top_k=3)
    for idx, chunk in enumerate(sparse_results):
        print(f"[{idx+1}] ID: {chunk.id} | Score (BM25): {chunk.score:.4f}")
        print(f"    Nội dung: {chunk.text[:120]}...\n")
        
    # 5. Test Hybrid Retrieval (RRF Fusion)
    print("\n--------------------------------------------------------------")
    print("C. KIỂM TRA HYBRID RETRIEVAL (RRF Fusion)")
    print("--------------------------------------------------------------")
    hybrid_results = hybrid_retriever.retrieve(test_query, top_k=3)
    for idx, chunk in enumerate(hybrid_results):
        print(f"[{idx+1}] ID: {chunk.id} | Score (RRF): {chunk.score:.4f}")
        print(f"    Nội dung: {chunk.text[:120]}...\n")
        
    # 6. Test RAG Pipeline End-to-End
    print("\n--------------------------------------------------------------")
    print("D. KIỂM TRA RAG PIPELINE END-TO-END (Dịch thuật + Hybrid + LLM)")
    print("--------------------------------------------------------------")
    pipeline = get_rag_pipeline()
    
    # Test câu hỏi tiếng Việt
    vi_query = "cúm là gì và có triệu chứng gì?"
    print(f"Yêu cầu tiếng Việt: {vi_query}")
    req_vi = ChatRequest(message=vi_query)
    resp_vi = pipeline.execute(req_vi)
    print(f"Ngôn ngữ phát hiện: {resp_vi.detected_language}")
    print(f"Nguồn trích dẫn: {[s.title for s in resp_vi.sources]}")
    print(f"Câu trả lời LLM:\n{resp_vi.answer}")
    
    # Test câu hỏi tiếng Anh
    en_query = "What is the common cold and what are its symptoms?"
    print(f"\nYêu cầu tiếng Anh: {en_query}")
    req_en = ChatRequest(message=en_query)
    resp_en = pipeline.execute(req_en)
    print(f"Ngôn ngữ phát hiện: {resp_en.detected_language}")
    print(f"Truy vấn được dịch: {resp_en.translated_query}")
    print(f"Nguồn trích dẫn: {[s.title for s in resp_en.sources]}")
    print(f"Câu trả lời LLM:\n{resp_en.answer}")
    print("\n==============================================================")
    print("HOÀN THÀNH KIỂM TRA!")
    print("==============================================================")

if __name__ == "__main__":
    run_verification()
