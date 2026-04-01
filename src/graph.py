from langgraph.graph import StateGraph, END
from src.state import GraphState

from src.nodes import (
    chat_node,
    retrieve_node,
    nearest_center_node,
    generate_node,
    self_repair_classifier_node,
    ask_intent_node,
    self_repair_guide_node,
    route_question,
    check_hallucination_routing,
    route_after_self_repair_check
)

# 추가로 구현해야 할 라우팅 함수 
def route_issue_type(state):
    """retrieve_node 이후, 문서/의도를 바탕으로 SW, HW, 센터방문을 분류합니다."""
    # 실제로는 state 내용을 보고 판단하는 로직이 들어갑니다.
    issue_type = state.get("issue_type", "software") 
    if issue_type == "hardware":
        return "self_repair_classifier_node"
    elif issue_type == "center":
        return "nearest_center_node"
    else:
        return "generate_node"


def build_cs_rag_graph():
    """LangGraph 기반의 Agentic RAG 파이프라인을 구축합니다."""
    
    # 1. 그래프 초기화 
    workflow = StateGraph(GraphState)
    
    # ==========================================
    # 2. 노드(작업장) 등록
    # ==========================================
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("nearest_center_node", nearest_center_node)
    workflow.add_node("generate_node", generate_node)
    workflow.add_node("self_repair_classifier_node", self_repair_classifier_node)
    workflow.add_node("ask_intent_node", ask_intent_node)
    workflow.add_node("self_repair_guide_node", self_repair_guide_node)

    # ==========================================
    # 3. edge 연결 (비즈니스 로직 반영)
    # ==========================================
    
# [시작점] 질문 의도 분류 및 이전 맥락 이어가기
    workflow.set_conditional_entry_point(
        route_question,
        {
            # 일반 시작
            "chat_node": "chat_node",          
            "retrieve_node": "retrieve_node",
            
            # 사용자 선택 후 재진입 시 바로 직행하는 경로 추가
            "nearest_center_node": "nearest_center_node",
            "self_repair_guide_node": "self_repair_guide_node"
            }
    )

    # [검색 후 분기] SW문제인지, HW문제인지, 무조건 센터방문인지 판단
    workflow.add_conditional_edges(
        "retrieve_node",
        route_issue_type, # <-- 이 라우팅 함수에서 SW/HW/Center를 가릅니다.
        {
            "generate_node": "generate_node",                             # SW -> 답변 생성
            "nearest_center_node": "nearest_center_node",                 # Center -> 센터 안내
            "self_repair_classifier_node": "self_repair_classifier_node"  # HW -> 자가수리 모델 판별
        }
    )

    # [자가수리 판별 후 분기] 자가수리 모델인가?
    workflow.add_conditional_edges(
        "self_repair_classifier_node",
        route_after_self_repair_check,
        {
            "ask_intent_node": "ask_intent_node",         # 자가수리 모델 O -> "자가수리 할래? 센터 갈래?" 선택지 제공
            "nearest_center_node": "nearest_center_node", # 자가수리 모델 X -> 묻지도 따지지도 않고 센터 안내
        }
    )

    # [답변 생성 후 분기] 생성된 답변이 환각(거짓말)인지 순수하게 검증만 수행
    workflow.add_conditional_edges(
        "generate_node",
        check_hallucination_routing,
        {
            "END": END,                                   # 환각 없음 -> 생성된 답변 반환하며 정상 종료
            "nearest_center_node": "nearest_center_node"  # 환각 발생 시 -> 안전하게 센터 안내로 Fallback
        }
    )

    # ==========================================
    # 4. 일반 edge 및 종료점(END) 연결
    # ==========================================
    workflow.add_edge("chat_node", END)
    # generate_node는 조건부 엣지로 END 처리가 되었으므로 제외
    workflow.add_edge("nearest_center_node", END)
    workflow.add_edge("ask_intent_node", END)
    workflow.add_edge("self_repair_guide_node", END)

    # 5. 그래프 컴파일
    app = workflow.compile()
    
    return app

# 싱글톤 형태로 앱 인스턴스 내보내기
rag_app = build_cs_rag_graph()