import os
import sys
import uuid
import logging
from pathlib import Path

# Append thư mục gốc vào PYTHONPATH để nhận diện module app
a = sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import chromadb
from core.config import settings
from core.logging import setup_logging
from infrastructure.embeddings import BGEM3EmbeddingService
setup_logging()
logger = logging.getLogger("markdown_ingestion")

def process_and_chunk_markdown(file_path: str) -> list:
    """
    Đọc và chunk file Markdown dựa trên cấu trúc Tiêu đề để giữ nguyên ngữ nghĩa.
    """
    if not os.path.exists(file_path):
        logger.error(f"Không tìm thấy file: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    # Bước 1: Định nghĩa các cấp độ tiêu đề để bóc tách ngữ cảnh
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    header_splits = markdown_splitter.split_text(markdown_content)
    
    # Bước 2: Với các đoạn văn quá dài dưới một tiêu đề, chia nhỏ tiếp bằng RecursiveCharacterTextSplitter
    # Giới hạn khoảng 400-500 ký tự (~100 từ tiếng Việt), gối đầu 50 ký tự để không mất đoạn câu
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    final_docs = text_splitter.split_documents(header_splits)
    
    return final_docs

def run_markdown_ingestion(file_name: str):
    logger.info("Khởi động Markdown Ingestion Pipeline...")
    
    # 1. Thực hiện tách nhỏ dữ liệu Markdown thông minh
    chunks = process_and_chunk_markdown(file_name)
    if not chunks:
        logger.warning("Không có dữ liệu để nạp.")
        return
        
    logger.info(f"Đã phân mảnh cấu trúc Markdown thành {len(chunks)} chunks.")

    # 2. Khởi tạo Embedding Service (BGE-M3)
    embed_service = BGEM3EmbeddingService()
    
    # 3. Kết nối ChromaDB dạng Persistent
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
    collection = chroma_client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    all_texts = []
    all_metadatas = []
    all_ids = []
    
    for chunk in chunks:
        all_texts.append(chunk.page_content)
        all_ids.append(str(uuid.uuid4()))
        
        # Merge các thông tin tiêu đề (Header) thu được từ Markdown vào metadata để bổ trợ ngữ cảnh hoặc filter sau này
        metadata = chunk.metadata.copy()
        metadata["source_file"] = file_name
        all_metadatas.append(metadata)
        
    # 4. Tính toán Embedding hàng loạt
    logger.info("Đang tạo Vectors với BGE-M3...")
    vectors = embed_service.embed_documents(all_texts)
    
    # 5. Đẩy thẳng vào ChromaDB
    logger.info("Đang ghi vào ChromaDB...")
    collection.add(
        ids=all_ids,
        embeddings=vectors,
        metadatas=all_metadatas,
        documents=all_texts
    )
    
    logger.info(f"Nạp thành công {len(all_texts)} dữ liệu Markdown y tế vào ChromaDB!")

if __name__ == "__main__":
    # Thay tên file .md y tế bạn đã chuẩn bị sẵn ở đây
    # Ví dụ: file data/raw/benh_tieu_duong.md
    folder_path = os.path.join("..\data", "raw", "cúm")

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".md"):
            relative_path = os.path.join("..\data", "raw", "cúm", file_name)
            run_markdown_ingestion(relative_path)