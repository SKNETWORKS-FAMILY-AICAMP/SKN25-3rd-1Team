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
    try:
        result = generate_cs_response(question, selected_device, thread_id)
        
        if isinstance(result, dict):
            messages = result.get('messages', [])
            answer = messages[-1].content if messages else "죄송합니다, 답변을 생성하지 못했습니다."
            source = result.get('source_document', '')
            # generate_node 답변일 때만 버튼 표시
            show_buttons = source == "내부 매뉴얼" and "찾을 수 없습니다" not in answer
            print(f"source_document: {source}")
            print(f"show_buttons: {show_buttons}")
            return answer, show_buttons
        return "죄송합니다, 내부 연결에 문제가 발생했습니다.", False
    except Exception as e:
        print(f"API Client Error: {e}")
        return "오류가 발생하여 답변을 생성할 수 없습니다.", False