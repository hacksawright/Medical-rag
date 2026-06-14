# Hệ thống Medical RAG Đa Ngữ (Multilingual Medical RAG Core)

Dự án **Medical RAG Core** cung cấp giải pháp tra cứu thông tin y học chuyên sâu dạng standalone API. Dự án tích hợp công nghệ tìm kiếm ngữ nghĩa Hybrid Retrieval (ChromaDB + BM25) cùng với khả năng suy luận mạnh mẽ từ mô hình ngôn ngữ lớn (Google Gemini / Qwen) để trả về các câu trả lời y khoa chuẩn xác, có căn thực (grounded) từ cơ sở tri thức y học đáng tin cậy.

Hệ thống được thiết kế theo mô hình **Kiến trúc Sạch (Clean Architecture)** và cung cấp giao diện RESTful API thuần túy, loại bỏ hoàn toàn các thành phần giao diện (UI) không cần thiết để tối ưu hóa hiệu năng và phục vụ trực tiếp cho việc tích hợp vào các nền tảng Web lớn khác (chẳng hạn như hệ thống quản lý bệnh nhân bằng Java).

---

## 🌟 Tính năng Nổi bật

*   **REST API Độc lập (Pure REST API):** Toàn bộ giao diện frontend được tách biệt hoàn toàn. Dự án hoạt động như một microservice độc lập viết bằng Python (FastAPI), sẵn sàng cho các dịch vụ bên ngoài (Java, React, v.v.) gọi sang.
*   **Đa ngôn ngữ đầu ra thông minh (Multilingual Output based on Input):**
    *   Hỗ trợ người dùng gửi câu hỏi bằng cả **Tiếng Việt** và **Tiếng Anh**.
    *   Cơ sở tri thức nguồn được lưu trữ bằng tiếng Việt để đạt hiệu quả chuyên môn cao nhất.
    *   Tự động phát hiện ngôn ngữ câu hỏi, thực hiện dịch truy vấn (qua mô hình EnViT5) và điều hướng sinh câu trả lời đúng ngôn ngữ gốc của bệnh nhân (Tiếng Anh phản hồi Tiếng Anh, Tiếng Việt phản hồi Tiếng Việt).
*   **Tìm kiếm kết hợp (Hybrid Retrieval):** Kết hợp độ chính xác ngữ nghĩa của tìm kiếm dense vector (ChromaDB với mô hình nhúng `BGE-M3`) và khả năng khớp từ khóa chính xác của tìm kiếm sparse vector (`BM25`), sau đó hòa trộn kết quả bằng thuật toán **RRF (Reciprocal Rank Fusion)**.
*   **Căn thực thông tin (Citations & Grounding):** Trả về câu trả lời y học kèm theo danh sách các nguồn tài liệu tham khảo cụ thể (File nguồn, tiêu đề danh mục, URL tham chiếu) nhằm chống hiện tượng AI bịa đặt thông tin (hallucination).

---

## 🛠️ Yêu cầu Hệ thống & Thiết lập Nhanh

### 1. Chuẩn bị Môi trường
*   **Python 3.10+**
*   Tạo môi trường ảo và cài đặt dependencies:
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

### 2. Thiết lập Biến môi trường
Tạo file `.env` ở thư mục gốc của dự án với các tham số mẫu dưới đây:
```env
ENV=production
LOG_LEVEL=INFO
LLM_PROVIDER=gemini  # Chọn 'gemini' hoặc 'qwen'
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
GEMINI_MODEL_NAME=gemini-2.5-flash
ENVIT5_API_URL=https://subplot-strep-ragweed.ngrok-free.dev/predict
```

### 3. Khởi chạy API Server
Chạy lệnh sau để khởi động máy chủ FastAPI:
```powershell
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
*   Tài liệu hướng dẫn API tự động (Swagger UI) sẽ khả dụng tại: `http://127.0.0.1:8000/docs`
*   Endpoint chính để gọi từ ứng dụng bên ngoài: `POST http://127.0.0.1:8000/api/v1/chat`

---

## 📖 Xem thêm Tài liệu Hướng dẫn

Để xem thông tin chi tiết hơn về dự án, vui lòng tham chiếu đến các tài liệu sau:

1.  **[PROJECT_DETAILS.md](file:///c:/Users/Admin/Desktop/TTCS/chatbot/medical-rag/PROJECT_DETAILS.md):** Giải thích chi tiết cấu trúc từng file/thư mục, sơ đồ luồng dữ liệu chi tiết của RAG pipeline, đặc tả API chi tiết và **đặc biệt là mã nguồn mẫu tích hợp phía Java (Java HttpClient)** để kết nối tới API Python này.
2.  **Thư mục [scripts/](file:///c:/Users/Admin/Desktop/TTCS/chatbot/medical-rag/scripts):** Chứa các kịch bản chạy thử nghiệm, nạp dữ liệu (`ingest_documents.py`), và kiểm thử RAG ngoại tuyến (`verify_rag.py`, `verify_hybrid_rag.py`).
