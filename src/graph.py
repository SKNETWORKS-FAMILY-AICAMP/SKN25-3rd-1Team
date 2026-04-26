from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import GraphState

from src.nodes import (
    #노드 불러오기
    chat_node,
    retrieve_node,
    generate_node,
    nearest_center_node,
    self_repair_classifier_node,
    self_repair_guide_node,
    fallback_node,
    translate_input_node,    # 추가
    translate_output_node,   # 추가
    #라우팅
    route_question,
    route_after_self_repair_check,
    route_issue_type,
)

def build_cs_rag_graph():    
    # 1. 그래프 초기화 
    workflow = StateGraph(GraphState)
    
    # ==========================================
    # 2. 노드 등록
    # ==========================================
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("generate_node", generate_node)
    workflow.add_node("nearest_center_node", nearest_center_node)
    workflow.add_node("self_repair_classifier_node", self_repair_classifier_node)
    workflow.add_node("self_repair_guide_node", self_repair_guide_node)
    workflow.add_node("fallback_node", fallback_node)
    workflow.add_node("translate_input_node", translate_input_node)
    workflow.add_node("translate_output_node", translate_output_node)

    # ==========================================
    # 3. 조건부 Edge 연결 (라우팅)
    # ==========================================
    workflow.add_edge(START, "translate_input_node")
    workflow.add_conditional_edges(
        "translate_input_node",
        route_question,
        {
            "chat_node": "chat_node",
            "retrieve_node": "retrieve_node",
            "nearest_center_node": "nearest_center_node"
        }
    )
    workflow.add_conditional_edges(
        "retrieve_node", 
        route_issue_type, 
        {
            "generate_node": "generate_node",
            "fallback_node": "fallback_node",
            "self_repair_classifier_node": "self_repair_classifier_node",
            "nearest_center_node": "nearest_center_node"
        }
    )
    workflow.add_conditional_edges(
        "self_repair_classifier_node", 
        route_after_self_repair_check, 
        {
            "nearest_center_node": "nearest_center_node",
            "self_repair_guide_node": "self_repair_guide_node"
        }
    )

    # ==========================================
    # 4. 일반 Edge 및 종료점(END) 연결
    # ==========================================
    workflow.add_edge("chat_node", "translate_output_node")                 # 일상 대화 끝나면 번역 후 종료
    workflow.add_edge("generate_node", "translate_output_node")             # 답변 생성 끝나면 번역 후 종료
    workflow.add_edge("self_repair_guide_node", "translate_output_node")    # 가이드 제공 끝나면 번역 후 종료
    workflow.add_edge("nearest_center_node", "translate_output_node")       # 센터 안내 끝나면 번역 후 종료
    workflow.add_edge("fallback_node", "translate_output_node")             # 예외 처리 끝나면 번역 후 종료
    workflow.add_edge("translate_output_node", END)                         # 번역 완료 후 종료

    # ==========================================
    # 5. 그래프 컴파일
    # ==========================================
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

# 싱글톤 형태로 앱 인스턴스 내보내기
rag_app = build_cs_rag_graph()
