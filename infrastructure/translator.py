import logging
import httpx
from domain.interfaces import ITranslator
from core.config import settings
from core.exceptions import TranslationError

logger = logging.getLogger(__name__)

class EnViT5Translator(ITranslator):
    def __init__(self):
        self.api_url = settings.ENVIT5_API_URL
        # Thiết lập timeout hợp lý phòng trường hợp cụm dịch vụ dịch thuật bị nghẽn
        self.timeout = httpx.Timeout(10.0, connect=5.0)
        # Thiết lập Client dùng chung để tái sử dụng connection pool (tương đương requests.Session)
        self.client = httpx.Client(timeout=self.timeout)

    def translate_en_to_vi(self, text: str) -> str:
        logger.info(f"Bắt đầu dịch truy vấn tiếng Anh: '{text}' sang tiếng Việt qua EnViT5 Service API")
        payload = {"text": "en: " + text}
        try:
            response = self.client.post(
                self.api_url,
                json=payload
            )
            logger.info(f"Status: {response.status_code}")
            logger.info(f"Response: {response.text}")
            logger.info(f"API URL: {self.api_url}")
            
            if response.status_code == 200:
                resp_json = response.json()
                
                # Phân rã cấu trúc JSON linh hoạt
                translated_text = None
                if isinstance(resp_json, dict):
                    translated_text = (
                        resp_json.get("output")
                    )
                    # Trường hợp trả về danh sách predictions
                    if not translated_text and "predictions" in resp_json:
                        preds = resp_json["predictions"]
                        if isinstance(preds, list) and len(preds) > 0:
                            translated_text = preds[0]
                        else:
                            translated_text = preds
                elif isinstance(resp_json, list) and len(resp_json) > 0:
                    translated_text = resp_json[0]
                elif isinstance(resp_json, str):
                    translated_text = resp_json

                if translated_text:
                    if isinstance(translated_text, str):
                        translated_text = translated_text.strip()
                        # Làm sạch tiền tố đầu ra "vi: " nếu model EnViT5 trả về kèm theo
                        if translated_text.startswith("vi:"):
                            translated_text = translated_text[3:].strip()
                        return translated_text
                
                raise TranslationError(f"Phản hồi từ API dịch thuật không đúng định dạng mong đợi: {response.text}")
            else:
                raise TranslationError(f"API dịch thuật trả về lỗi HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Lỗi khi kết nối/gọi API EnViT5: {str(e)}")
            logger.info(f"API URL: {self.api_url}")
            raise TranslationError(f"Dịch thuật thất bại: {str(e)}")