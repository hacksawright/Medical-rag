from fastapi import APIRouter, Depends, HTTPException, status
from domain.models import ChatRequest, ChatResponse
from services.rag_pipeline import MedicalRAGPipeline
from api.dependencies import get_rag_pipeline
from core.exceptions import MedicalRAGException

router = APIRouter(prefix="/chat", tags=["Medical Chatbot Interface"])

@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def process_medical_chat(
    request: ChatRequest,
    pipeline: MedicalRAGPipeline = Depends(get_rag_pipeline)
):
    try:
        response = pipeline.execute(request)
        return response
    except MedicalRAGException as me:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=me.message)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý hệ thống RAG cục bộ: {str(e)}"
        )