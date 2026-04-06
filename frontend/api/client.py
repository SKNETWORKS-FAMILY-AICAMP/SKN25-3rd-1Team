import os
import sys
import requests
from dotenv import load_dotenv

# 모듈 경로 설정을 통한 시스템 접근
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(frontend_dir)

# 환경 변수 로드
load_dotenv(os.path.join(root_dir, '.env'))

FASTAPI_URL = os.getenv("FASTAPI_URL")

def get_chat_response(question: str, selected_device: str, thread_id: str = "streamlit_user"):
    """
    고객 CS 챗봇 API 클라이언트.
    FastAPI 백엔드 서버에 REST API 요청을 보내서 답변을 반환받습니다.
    """
    try:
        payload = {
            "question": question,
            "selected_device": selected_device,
            "thread_id": thread_id
        }
        
        # FastAPI 서버로 POST 요청 전송
        response = requests.post(FASTAPI_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        if isinstance(result, dict) and result.get("answer"):
            return result["answer"]
            
        return "죄송합니다, 답변을 생성하지 못했습니다. 다시 질문해주세요."
        
    except requests.exceptions.RequestException as e:
        print(f"API Connection Error: {e}")
        return "FastAPI 서버에 연결할 수 없습니다. 서버 모델을 처음 불러올 때 약 30초~1분 정도 소요될 수 있습니다. 백엔드 터미널 창에 'Application startup complete' 메시지가 떴는지 확인 후 다시 시도해주세요."
    except Exception as e:
        print(f"API Client Error: {e}")
        return "오류가 발생하여 답변을 생성할 수 없습니다. 고객센터에 직접 문의해주세요."

