import os
import sys

# Append parent directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.dependencies import get_rag_pipeline
from domain.models import ChatRequest

def verify():
    pipeline = get_rag_pipeline()
    
    # 1. Test Vietnamese Query
    vi_query = "cúm là gì và triệu chứng như thế nào?"
    print("\n==========================================")
    print(f"TEST VIETNAMESE QUERY: {vi_query}")
    print("==========================================")
    request_vi = ChatRequest(message=vi_query)
    response_vi = pipeline.execute(request_vi)
    print(f"Detected Language: {response_vi.detected_language}")
    print(f"Answer:\n{response_vi.answer}")
    print(f"Sources: {[s.title for s in response_vi.sources]}")
    
    # 2. Test English Query
    en_query = "What is type 2 diabetes and its common symptoms?"
    print("\n==========================================")
    print(f"TEST ENGLISH QUERY: {en_query}")
    print("==========================================")
    request_en = ChatRequest(message=en_query)
    response_en = pipeline.execute(request_en)
    print(f"Detected Language: {response_en.detected_language}")
    print(f"Translated Query: {response_en.translated_query}")
    print(f"Answer:\n{response_en.answer}")
    print(f"Sources: {[s.title for s in response_en.sources]}")

if __name__ == "__main__":
    verify()
