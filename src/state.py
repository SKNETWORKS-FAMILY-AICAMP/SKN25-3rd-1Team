from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    selected_device: str             # 사용자가 사전 선택한 기기명 (예: "갤럭시 S24", "선택하지 않음")
    context: str
    source_document: str
    reliability_score: float
    
    # --- 자가수리 판별용 상태 추가 ---
    device_model: Optional[str]      # 예: "Galaxy S22"
    is_hardware_issue: bool          # 물리적 파손 여부 (True/False)
    waiting_for_repair_choice: bool  # 가이드 제공 여부 플래그


    # --- 평가용 상태 추가 ---
    question: str
    retrieved_contexts: list[str]
    answer: str
    eval_data: dict