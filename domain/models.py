from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    message: str = Field(..., description="Câu hỏi y tế từ người dùng bằng tiếng Anh hoặc tiếng Việt")

class SourceCitation(BaseModel):
    title: str = Field(..., description="Tiêu đề tài liệu hoặc tên chuyên mục")
    source: str = Field(..., description="Nguồn bài viết (Vinmec, Hello Bacsi, NHS, Mayo Clinic)")
    url: Optional[str] = Field(None, description="Đường dẫn nguồn gốc nếu có")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Câu trả lời y tế tiếng Việt được căn thực từ ngữ cảnh")
    detected_language: str = Field(..., description="Ngôn ngữ được nhận diện ban đầu")
    translated_query: Optional[str] = Field(None, description="Câu hỏi sau khi dịch sang tiếng Việt nếu đầu vào là tiếng Anh")
    sources: List[SourceCitation] = Field(default_factory=list, description="Danh sách tài liệu tham chiếu")

class MedicalDocumentChunk(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    score: Optional[float] = None