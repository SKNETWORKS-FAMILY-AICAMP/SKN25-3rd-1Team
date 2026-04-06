from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    selected_device: str
    context: str
    source_document: str
    reliability_score: float
    relevance_score: float           # ← 추가
    show_resolution_buttons: bool
    
    # --- 자가수리 판별용 상태 추가 ---
    device_model: Optional[str]
    is_hardware_issue: bool
    waiting_for_repair_choice: bool
