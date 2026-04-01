import sys
import os
from dotenv import load_dotenv

# 1. 모듈 경로 설정 (src 폴더 접근용)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 2. 파이프라인 함수 임포트
from src.pipelines.generation_pipeline import generate_cs_response

# 환경 변수 로드
load_dotenv()

def run_test_scenarios():
    """LangGraph 라우팅 및 멀티턴(Memory)이 정상 작동하는지 확인하기 위한 시나리오 테스트"""
    
    # thread_id를 부여하여 대화의 맥락이 이어지는지 테스트합니다.
    scenarios = [
        {
            "title": "테스트 1: 단순 인사말 (Router -> Chat Node)",
            "query": "안녕! 오늘 날씨 참 좋다. 넌 이름이 뭐야?",
            "thread_id": "user_1"
        },
        {
            "title": "테스트 2: 일반 SW CS 문의 (Router -> Retrieve -> Generate)",
            "query": "노트 어시스트 기능은 어떻게 사용하나요?",
            "thread_id": "user_2"
        },
        {
            "title": "테스트 3-1: 하드웨어 파손 + 자가수리 대상 (Ask Intent Node)",
            "query": "갤럭시 S22 액정이 깨졌어요. 수리비가 얼마인가요?",
            "thread_id": "user_3_multiturn" # 멀티턴 대화방 1번
        },
        {
            "title": "테스트 3-2: 3-1에서 이어지는 대답 (Router -> Guide Node)",
            "query": "내가 직접 부품 사서 수리해볼게. 방법 알려줘.",
            "thread_id": "user_3_multiturn" # [핵심] 방금 전과 똑같은 thread_id를 사용!
        },
        {
            "title": "테스트 5: 환각/엉뚱한 질문 (Generate -> Hallucination Fail -> Center)",
            "query": "사과폰 15 프로 배터리 교체 비용 알려주세요.",
            "thread_id": "user_5"
        }
    ]

    print("=" * 60)
    print("🚀 [Agentic RAG 시나리오 테스트 시작 (멀티턴 지원)]")
    print("=" * 60)

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n\n▶️ [{scenario['title']}] - 대화방: {scenario['thread_id']}")
        print(f"👤 사용자: {scenario['query']}")
        print("-" * 60)
        
        # 파이프라인 실행 (thread_id 전달)
        result = generate_cs_response(scenario['query'], thread_id=scenario['thread_id'])
        
        # 결과 출력
        answer = result.get('answer', '답변을 생성하지 못했습니다.')
        source = result.get('source_document', '출처 없음')
        score = result.get('reliability_score', 'N/A')
        
        print(f"🤖 AI 답변:\n{answer}\n")
        print(f"📑 출처: {source} | 📊 신뢰도: {score}")
        
        # 내부 State(Flag 및 추출 데이터) 확인용
        print("\n[🔍 내부 State 모니터링]")
        print(f" - 추출된 기기명: {result.get('device_model', '없음')}")
        print(f" - 하드웨어 이슈 여부: {result.get('is_hardware_issue', False)}")
        print(f" - ⏳ 사용자 선택 대기 상태(Flag): {result.get('waiting_for_repair_choice', False)}")
            
    print("\n" + "=" * 60)
    print("✅ [테스트 종료] 모든 시나리오 테스트가 완료되었습니다.")
    print("=" * 60)

if __name__ == "__main__":
    # 1. 정해진 시나리오 모드 실행
    run_test_scenarios()
    
    # 2. 시나리오 종료 후, 직접 입력해볼 수 있는 대화형 챗봇 모드
    print("\n💡 직접 질문을 입력해보세요. (멀티턴 대화가 유지됩니다. 종료하려면 'q' 입력)")
    
    # 사용자가 직접 테스트할 때는 동일한 세션 ID를 유지해야 대화가 이어집니다.
    interactive_thread_id = "interactive_tester_001" 
    
    while True:
        user_input = input("\n👤 사용자: ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("테스트를 종료합니다.")
            break
            
        if not user_input.strip():
            continue
            
        # 동일한 interactive_thread_id를 넘겨주어 이전 맥락과 Flag를 기억하게 함
        result = generate_cs_response(user_input, thread_id=interactive_thread_id)
        
        print("-" * 60)
        print(f"🤖 AI 답변:\n{result.get('answer')}")
        
        # 플래그 상태 살짝 표시
        if result.get('waiting_for_repair_choice'):
            print("\n  [⚙️ System: 사용자의 선택(자가수리/센터)을 기다리는 중입니다...]")
            
        print("-" * 60)