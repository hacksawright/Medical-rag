import logging
import time
from domain.interfaces import ILanguageDetector, ITranslator, IEmbeddingService, IVectorStore, ILLMService
from domain.models import ChatRequest, ChatResponse, SourceCitation
from core.exceptions import MedicalRAGException

logger = logging.getLogger(__name__)

class MedicalRAGPipeline:
    def __init__(
        self,
        language_detector: ILanguageDetector,
        translator: ITranslator,
        embedding_service: IEmbeddingService,
        vector_store: IVectorStore,
        llm_service: ILLMService,
        system_prompt_template: str,
        system_prompt_template_en: str = None
    ):
        self.language_detector = language_detector
        self.translator = translator
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.system_prompt_template = system_prompt_template
        self.system_prompt_template_en = system_prompt_template_en

    def execute(self, request: ChatRequest) -> ChatResponse:
        start_time = time.time()
        original_query = request.message.strip()
        
        # 1. Nhận diện ngôn ngữ
        lang = self.language_detector.detect_language(original_query)
        
        # 2. Dịch thuật nếu là Tiếng Anh
        processed_query = original_query
        translated_query = None
        if lang == "en":
            translated_query = self.translator.translate_en_to_vi(original_query)
            processed_query = translated_query
            
        # 3. Tạo Vector và Truy vấn ChromaDB
        query_vector = self.embedding_service.embed_query(processed_query)
        matched_chunks = self.vector_store.query_similar_documents(query_vector, top_k=4)
        
        # 4. Xử lý Metadata linh hoạt từ cấu trúc Markdown
        context_str = ""
        citations = []
        seen_sources = set()
        
        for idx, chunk in enumerate(matched_chunks):
            meta = chunk.metadata or {}
            
            # Trích xuất thẻ Header từ Markdown (Nếu không có thì dùng giá trị mặc định)
            source_file = meta.get("source_file", "Tài liệu Y khoa cục bộ")
            h1 = meta.get("Header_1", "")
            h2 = meta.get("Header_2", "")
            h3 = meta.get("Header_3", "")
            
            # Gom các thẻ Header lại để làm tiêu đề trích dẫn gọn gàng
            title_parts = [h for h in [h1, h2, h3] if h]
            display_title = " - ".join(title_parts) if title_parts else "Thông tin chung"
            
            source_key = f"{source_file}-{display_title}"
            
            context_str += f"--- ĐOẠN TRI THỨC {idx+1} (Nguồn: {source_file}) ---\n"
            context_str += f"{chunk.text}\n\n"
            
            # Chống trùng lặp nguồn trích dẫn
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                citations.append(SourceCitation(
                    title=display_title,
                    source=source_file,
                    url=meta.get("url") # Mặc định là None đối với file MD nội bộ
                ))

        # 5. Xử lý kịch bản rỗng
        if not matched_chunks:
            logger.warning(f"Không có context cho: '{processed_query}'")
            context_str = "Không tìm thấy tài liệu y khoa nào tương thích trong cơ sở dữ liệu."

        # 6. Gọi LLM
        if lang == "en" and self.system_prompt_template_en:
            system_prompt = self.system_prompt_template_en
            user_prompt = f"Document context (Ngữ cảnh y khoa tài liệu):\n{context_str}\n\nPatient's question (Câu hỏi bệnh nhân): {original_query}"
        else:
            system_prompt = self.system_prompt_template
            user_prompt = f"Ngữ cảnh y khoa tài liệu:\n{context_str}\n\nCâu hỏi bệnh nhân: {processed_query}"

        try:
            llm_answer = self.llm_service.generate_answer(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
        except Exception as e:
            logger.error(f"Lỗi LLM: {str(e)}")
            llm_answer = "Hệ thống đang gặp sự cố, vui lòng thử lại sau."

        return ChatResponse(
            answer=llm_answer,
            detected_language=lang,
            translated_query=translated_query,
            sources=citations
        )