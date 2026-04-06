import os
import sys
from dotenv import load_dotenv

# 모듈 경로 설정을 통한 src 패키지 접근
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(frontend_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.pipelines.generation_pipeline import generate_cs_response

# 환경 변수 로드
load_dotenv(os.path.join(root_dir, '.env'))

def get_chat_response(question: str, selected_device: str, thread_id: str = "streamlit_user"):
    """
    고객 CS 챗봇 파이프라인 생성 함수 래퍼.
    질문과 기기 정보를 받아 에이전틱 RAG 파이프라인을 통과시킨 후 답변을 반환합니다.
    """
    try:
        # LangGraph를 통해 생성된 결과 딕셔너리 반환
        result = generate_cs_response(question, selected_device, thread_id)
        
        if isinstance(result, dict):
            messages = result.get('messages', [])
            if messages:
                return messages[-1].content
            return "죄송합니다, 답변을 생성하지 못했습니다. 다시 질문해주세요."
        return "죄송합니다, 내부 연결에 문제가 발생했습니다."
    except Exception as e:
        print(f"API Client Error: {e}")
        return "오류가 발생하여 답변을 생성할 수 없습니다. 고객센터에 직접 문의해주세요."
