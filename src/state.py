from typing import TypedDict, Optional

class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    source_document: str
    reliability_score: float
    
    # --- 자가수리 판별용 상태 추가 ---
    device_model: Optional[str]      # 예: "Galaxy S22"
    is_hardware_issue: bool          # 물리적 파손 여부 (True/False)
    wants_self_repair: Optional[bool] # 자가수리 의향 (True/False/None)
    waiting_for_repair_choice: bool