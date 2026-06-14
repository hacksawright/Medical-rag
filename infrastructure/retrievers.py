import re
import os
import pickle
import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from domain.interfaces import IRetriever, IVectorStore, IEmbeddingService
from domain.models import MedicalDocumentChunk
from core.config import settings
from core.exceptions import BM25RetrieverError, HybridRetrieverError

logger = logging.getLogger(__name__)

class ChromaRetriever(IRetriever):
    """
    Dense Retriever sử dụng Embedding Search qua ChromaDB
    """
    def __init__(self, vector_store: IVectorStore, embedding_service: IEmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def retrieve(self, query: str, top_k: int) -> List[MedicalDocumentChunk]:
        try:
            logger.info(f"[Dense Search] Đang tạo vector truy vấn cho: '{query[:50]}...'")
            query_vector = self.embedding_service.embed_query(query)
            
            logger.info(f"[Dense Search] Đang tìm kiếm {top_k} tài liệu trong ChromaDB...")
            return self.vector_store.query_similar_documents(query_vector, top_k)
        except Exception as e:
            logger.error(f"[Dense Search] Lỗi trong quá trình tìm kiếm dense vector: {str(e)}")
            raise

class BM25Retriever(IRetriever):
    """
    Sparse Retriever sử dụng thuật toán BM25 qua thư viện rank-bm25
    """
    def __init__(self, vector_store: IVectorStore):
        self.vector_store = vector_store
        self.index_path = settings.BM25_INDEX_PATH
        self.index_dir = settings.BM25_INDEX_DIR
        self.bm25 = None
        self.corpus_chunks: List[MedicalDocumentChunk] = []
        
        # Tự động load index từ đĩa hoặc rebuild từ ChromaDB
        self.load_or_build_index()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize văn bản tiếng Việt/tiếng Anh ở mức cơ bản, không dấu câu, chữ thường.
        Sử dụng re.findall với \w+ để giữ nguyên các từ Unicode (tiếng Việt có dấu).
        """
        if not text:
            return []
        text = text.lower()
        return re.findall(r'\w+', text)

    def load_or_build_index(self):
        try:
            if os.path.exists(self.index_path):
                logger.info(f"[BM25] Tải chỉ mục từ đường dẫn lưu trữ: {self.index_path}")
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25 = data["bm25"]
                    self.corpus_chunks = data["corpus_chunks"]
                logger.info(f"[BM25] Tải thành công chỉ mục với {len(self.corpus_chunks)} chunks tài liệu.")
            else:
                logger.warning(f"[BM25] Không tìm thấy chỉ mục tại {self.index_path}. Đang tiến hành tạo mới từ dữ liệu ChromaDB...")
                self.rebuild_index()
        except Exception as e:
            logger.error(f"[BM25] Lỗi tải chỉ mục: {str(e)}. Tiến hành tự phục hồi bằng cách rebuild...")
            try:
                self.rebuild_index()
            except Exception as re_err:
                raise BM25RetrieverError(f"Không thể khởi tạo chỉ mục BM25: {str(re_err)}")

    def rebuild_index(self):
        """
        Quét toàn bộ ChromaDB, tách từ và dựng lại chỉ mục BM25
        """
        logger.info("[BM25] Bắt đầu quét dữ liệu nguồn từ ChromaDB...")
        chunks = self.vector_store.get_all_documents()
        
        if not chunks:
            logger.warning("[BM25] ChromaDB trống. Khởi tạo chỉ mục rỗng.")
            self.bm25 = None
            self.corpus_chunks = []
            return
            
        logger.info(f"[BM25] Bắt đầu tokenizer {len(chunks)} tài liệu...")
        tokenized_corpus = [self.tokenize(chunk.text) for chunk in chunks]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_chunks = chunks
        
        # Lưu file nhị phân pkl lên đĩa cứng
        try:
            os.makedirs(self.index_dir, exist_ok=True)
            with open(self.index_path, "wb") as f:
                pickle.dump({
                    "bm25": self.bm25,
                    "corpus_chunks": self.corpus_chunks
                }, f)
            logger.info(f"[BM25] Đã ghi chỉ mục mới lên ổ đĩa: {self.index_path}")
        except Exception as e:
            logger.error(f"[BM25] Lỗi khi lưu chỉ mục nhị phân: {str(e)}")
            raise BM25RetrieverError(f"Lưu trữ chỉ mục BM25 thất bại: {str(e)}")

    def retrieve(self, query: str, top_k: int) -> List[MedicalDocumentChunk]:
        if not self.bm25 or not self.corpus_chunks:
            logger.warning("[BM25] Chỉ mục hiện tại rỗng. Trả về kết quả trống.")
            return []
            
        try:
            logger.info(f"[BM25 Search] Đang truy vấn từ khóa: '{query}'")
            tokenized_query = self.tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            
            scored_chunks = []
            for idx, score in enumerate(scores):
                if score > 0.0:  # Loại bỏ các chunk không có từ khóa trùng khớp (score = 0)
                    chunk = self.corpus_chunks[idx]
                    cloned_chunk = MedicalDocumentChunk(
                        id=chunk.id,
                        text=chunk.text,
                        metadata=chunk.metadata,
                        score=float(score)
                    )
                    scored_chunks.append(cloned_chunk)
                    
            # Sắp xếp theo điểm BM25 giảm dần
            scored_chunks.sort(key=lambda x: x.score, reverse=True)
            logger.info(f"[BM25 Search] Tìm thấy {len(scored_chunks)} ứng viên thích hợp.")
            return scored_chunks[:top_k]
        except Exception as e:
            logger.error(f"[BM25 Search] Lỗi truy vấn: {str(e)}")
            raise BM25RetrieverError(f"BM25 Retrieval Error: {str(e)}")

class HybridRetriever(IRetriever):
    """
    Hybrid Retriever kết hợp Dense + Sparse và sắp xếp lại bằng Reciprocal Rank Fusion (RRF)
    """
    def __init__(self, dense_retriever: IRetriever, sparse_retriever: IRetriever):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = settings.RRF_K
        self.dense_top_k = settings.DENSE_TOP_K
        self.sparse_top_k = settings.SPARSE_TOP_K

    def retrieve(self, query: str, top_k: int) -> List[MedicalDocumentChunk]:
        try:
            logger.info(f"[Hybrid Search] Bắt đầu tìm kiếm song song: '{query}'")
            
            # Lấy danh sách kết quả thô từ 2 retriever
            dense_results = self.dense_retriever.retrieve(query, self.dense_top_k)
            sparse_results = self.sparse_retriever.retrieve(query, self.sparse_top_k)
            
            logger.info(f"[Hybrid Search] Dense thu về {len(dense_results)} chunks, Sparse thu về {len(sparse_results)} chunks.")
            
            # Hợp nhất bằng RRF
            fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)
            
            logger.info(f"[Hybrid Search] Đã hợp nhất và chấm điểm RRF xong. Tổng số lượng duy nhất: {len(fused_results)}")
            return fused_results[:top_k]
        except Exception as e:
            logger.error(f"[Hybrid Search] Thao tác tìm kiếm hỗn hợp thất bại: {str(e)}")
            raise HybridRetrieverError(f"Hybrid Search failed: {str(e)}")

    def _reciprocal_rank_fusion(
        self, 
        dense_results: List[MedicalDocumentChunk], 
        sparse_results: List[MedicalDocumentChunk]
    ) -> List[MedicalDocumentChunk]:
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, MedicalDocumentChunk] = {}
        
        # 1. Tính điểm cho danh sách Dense
        for rank, chunk in enumerate(dense_results):
            chunk_id = chunk.id
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk
            # RRF rank tính từ 1 nên là rank + 1
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + (rank + 1))
            
        # 2. Tính điểm cho danh sách Sparse
        for rank, chunk in enumerate(sparse_results):
            chunk_id = chunk.id
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + (rank + 1))
            
        # 3. Sắp xếp lại các documents theo điểm RRF giảm dần
        sorted_pairs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        fused_chunks = []
        for chunk_id, score in sorted_pairs:
            orig_chunk = chunk_map[chunk_id]
            fused_chunks.append(MedicalDocumentChunk(
                id=orig_chunk.id,
                text=orig_chunk.text,
                metadata=orig_chunk.metadata,
                score=score
            ))
            
        return fused_chunks
