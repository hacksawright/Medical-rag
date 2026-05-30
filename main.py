from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from core.logging import setup_logging
from api.v1.chat import router as chat_router

# Kích hoạt hệ thống log tập trung
setup_logging()

def create_app() -> FastAPI:
    app = FastAPI(
        title="Multilingual Medical RAG System Core",
        description="Lõi dịch vụ AI phục vụ tra cứu thông tin y tế đa ngữ standalone - PTIT AI Project Research",
        version="1.0.0"
    )
    
    # Cấu hình CORS để Spring Boot backend hoặc ứng dụng React Frontend kết nối thuận tiện
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Đăng ký các phân vùng endpoints API con
    app.include_router(chat_router, prefix="/api/v1")
    
    @app.get("/health", tags=["System Monitoring"])
    async def health_check():
        return {"status": "healthy", "environment": settings.ENV, "llm_provider_active": settings.LLM_PROVIDER}
        
    return app

app = create_app()