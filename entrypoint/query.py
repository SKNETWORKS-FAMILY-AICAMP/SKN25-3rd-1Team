"""
실험용 : 코드 돌아가는지 cli로 확인
"""
import sys
import os
from dotenv import load_dotenv

# cp949 인코딩 에러 방지를 위해 stdout 변경
sys.stdout.reconfigure(encoding='utf-8')

# 1. 모듈 경로 설정 (src 폴더 접근용)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 2. 파이프라인 함수 임포트
from src.pipelines.generation_pipeline import generate_cs_response

# 환경 변수 로드
load_dotenv()

if __name__ == "__main__":

    interactive_thread_id = "userdemo"
    print("\n\n" + "=" * 60)
    print("대화형 모드 진입 (종료: 'q', 'quit', 'exit')")
    
    # 1. 초기 기기 선택
    print("\n현재 사용중인 기기 모델을 선택하거나 입력해주세요.")
    print("(예: 갤럭시 S24, S20 Ultra, 기타, 또는 엔터키를 눌러 건너뛰기)")
    user_device = input("기기명: ").strip()
    if not user_device:
        user_device = "선택하지 않음"
    
    print(f"\n[{user_device}] 로 기기가 설정되었습니다. 대화를 시작합니다.")
    
    while True:
        user_input = input("\n👤 사용자: ")
        if user_input.lower() in ['q']:
            print("테스트를 종료합니다.")
            break
            
        if not user_input.strip():
            continue
            
        result = generate_cs_response(user_input, selected_device=user_device, thread_id=interactive_thread_id)
        
        print("-" * 60)
        if isinstance(result, dict):
            print(f"🤖 AI 답변:\n{result.get('answer', '')}")
        else:
            print("🤖 AI 답변 생성 실패")
        print("-" * 60)