import os
from pydantic import BaseModel, Field
from typing import Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.state import GraphState
from src.pipelines.embedding_pipeline import get_vector_store
import json # <-- 추가

def load_self_repair_models():
    """self-repair-list.json 파일을 읽어서 모든 모델명을 1차원 리스트로 반환합니다."""
    # 실제 프로젝트 경로에 맞게 수정하세요 (예: 최상단 기준)
    file_path = os.path.join("data", "processed", "self-repair-list.json")
    
    # 절대 경로로 직접 지정할 경우 아래 주석 해제 후 사용:
    # file_path = r"C:\SKN25-3rd-1Team\data\processed\self-repair-list.json"
    

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    all_models = []
    for series, models in data.items():
        all_models.extend(models)
    return all_models # ['S20', 'S20 Plus', 'S20 Ultra', 'S21' ...]



# 공통으로 사용할 LLM (정확한 판단을 위해 temperature=0.0 유지)
llm = ChatOpenAI(temperature=0.0, model="gpt-4-turbo")

# ==========================================
# [1] 구조화된 출력(Pydantic) 모델 정의 
# ==========================================
class RouteQuery(BaseModel):
    intent: Literal["greeting", "cs_query"] = Field(
        description="질문이 단순 인사/잡담이면 'greeting', 기기 고장이나 서비스 문의면 'cs_query'"
    )

class IssueTypeCheck(BaseModel):
    issue_type: Literal["software", "hardware", "center_visit"] = Field(
        description="질문이 앱 설정 등 소프트웨어 문제면 'software', 부품/파손 등 하드웨어 문제면 'hardware', 다짜고짜 센터를 찾으면 'center_visit'"
    )

class HallucinationCheck(BaseModel):
    is_grounded: Literal["pass", "fail"] = Field(
        description="답변이 제공된 문서 내용에만 기반했으면 'pass', 지어낸 내용이 있으면 'fail'"
    )

class SelfRepairExtraction(BaseModel):
    device_model: str = Field(description="언급된 기기 모델명 (예: Galaxy S22), 없으면 빈 문자열")
    is_hardware_issue: bool = Field(description="액정 파손, 배터리 교체 등 물리적 하드웨어 수리가 필요한지 여부")

# ==========================================
# [2] 조건부 라우팅 함수 (Conditional Edges)
# ==========================================
def route_question(state: GraphState) -> str:
    """(1) 진입점 라우터: 이전 대화 맥락(자가수리 대기 상태)을 먼저 확인하고, 아니면 의도를 분류합니다."""
    print("---ROUTING: 진입점 및 의도 분류 중---")
    
    # [핵심] 이전 턴에서 "자가수리할래? 센터갈래?" 물어보고 끝난 상태인지 확인
    if state.get("waiting_for_repair_choice"):
        user_msg = state.get("question", "")
        print(f"-> 사용자의 선택 답변 감지: {user_msg}")
        
        if "센터" in user_msg or "방문" in user_msg or "가고" in user_msg:
            return "nearest_center_node"
        elif "자가수리" in user_msg or "직접" in user_msg or "가이드" in user_msg or "네" in user_msg:
            return "self_repair_guide_node"
        else:
            # 대답이 모호하면 다시 물어보기
            return "ask_intent_node" 
            
    # 일반적인 새 질문인 경우
    structured_llm = llm.with_structured_output(RouteQuery)
    result = structured_llm.invoke(state["question"])
    
    if result.intent == "greeting":
        return "chat_node"
    return "retrieve_node"


def route_issue_type(state: GraphState) -> str:
    """(3) 문제 유형 라우터: 검색 직후 SW, HW, 센터 방문 여부를 판별합니다."""
    print("---ROUTING: 문제 유형 분류 중 (SW / HW / Center)---")
    structured_llm = llm.with_structured_output(IssueTypeCheck)
    
    prompt = f"질문: {state['question']}\n이 질문의 유형을 분류하세요."
    result = structured_llm.invoke(prompt)
    
    if result.issue_type == "hardware":
        return "self_repair_classifier_node"
    elif result.issue_type == "center_visit":
        return "nearest_center_node"
    else:
        # 기본적으로 소프트웨어 문제나 정보성 질문은 생성 노드로 보냄
        return "generate_node"


def route_after_self_repair_check(state: GraphState) -> str:
    """(7) 자가수리 대상 판별 후 흐름 제어"""
    print("---ROUTING: 자가수리 대상 여부 판단 후 분기---")
    
    # [수정됨] JSON 파일에서 실제 모델 리스트 불러오기
    valid_models = load_self_repair_models()
    
    model = state.get("device_model", "")
    is_hw = state.get("is_hardware_issue", False)
    
    is_target_model = False
    if model:
        # 공백 제거 및 대문자 변환으로 비교 (예: "Z 플립5" -> "Z플립5")
        model_clean = model.replace(" ", "").upper()
        
        for target in valid_models:
            target_clean = target.replace(" ", "").upper()
            if target_clean in model_clean or model_clean in target_clean:
                is_target_model = True
                break
    
    if is_target_model and is_hw:
        print(f"-> 확인됨: {model} 은(는) 자가수리 대상입니다.")
        return "ask_intent_node"
    else:
        print(f"-> 확인됨: {model} 은(는) 자가수리 대상이 아닙니다.")
        return "nearest_center_node"


def check_hallucination_routing(state: GraphState) -> str:
    """(6) 환각 검증: 생성된 답변이 문서에 기반하는지만 순수하게 검사합니다."""
    print("---ROUTING: 환각(Hallucination) 검증 중---")
    structured_llm = llm.with_structured_output(HallucinationCheck)
    
    prompt = f"문서: {state['context']}\n답변: {state['answer']}\n답변이 문서의 내용에만 근거하고 있습니까?"
    result = structured_llm.invoke(prompt)
    
    # 환각이 없으면(pass) 정상 종료를 위해 "END"라는 문자열 반환 (LangGraph에서 내부적으로 처리됨)
    # LangGraph 분기 설정 시 "END": END 로 매핑해야 함.
    if result.is_grounded == "pass":
        return "END" 
    
    # 환각이 발견되면 안전하게 오프라인 센터 방문으로 유도
    return "nearest_center_node"

# ==========================================
# [3] 실제 작업을 수행하는 노드들 (Nodes)
# ==========================================
def chat_node(state: GraphState) -> GraphState:
    print("---NODE: 일반 대화---")
    response = llm.invoke(f"고객 서비스 어시스턴트로서 친절하게 답해주세요. 질문: {state['question']}")
    # 플래그 초기화 포함
    return {"answer": response.content, "source_document": "대화형 AI", "reliability_score": 1.0, "waiting_for_repair_choice": False}

def retrieve_node(state: GraphState) -> GraphState:
    print("---NODE: 문서 검색---")
    vector_store = get_vector_store("faq")
    docs = vector_store.as_retriever(search_kwargs={"k": 2}).invoke(state["question"])
    context = "\n\n".join([f"[{d.metadata.get('title', 'Unknown')}] {d.page_content}" for d in docs])
    return {"context": context if context else "검색된 문서 없음"}

def generate_node(state: GraphState) -> GraphState:
    print("---NODE: 1차 답변 생성---")
    prompt = f"다음 문서에만 기반해서 고객의 질문에 답하세요.\n문서: {state['context']}\n질문: {state['question']}"
    response = llm.invoke(prompt)
    return {"answer": response.content, "source_document": "내부 매뉴얼", "reliability_score": 0.9}

def self_repair_classifier_node(state: GraphState) -> GraphState:
    """하드웨어 문제 시, 기기 모델명과 하드웨어 여부를 추출합니다."""
    print("---NODE: 자가수리 가능 기기인지 추출 중---")
    structured_llm = llm.with_structured_output(SelfRepairExtraction)
    # 컨텍스트와 질문을 바탕으로 추출
    result = structured_llm.invoke(state["question"] + "\n관련 문서: " + state.get("context", ""))
    
    return {
        "device_model": result.device_model,
        "is_hardware_issue": result.is_hardware_issue
    }

def ask_intent_node(state: GraphState) -> GraphState:
    """사용자에게 선택지를 주고 '대기 상태'로 만듭니다."""
    print("---NODE: 자가수리 의향 역질문 생성---")
    appended_answer = (
        "💡 **[안내] 문의하신 기기는 고객님께서 직접 부품을 구매하여 수리하실 수 있는 '자가수리 대상 기기'입니다.**\n"
        "자가수리 매뉴얼과 부품 구매처를 안내해 드릴까요, 아니면 가까운 서비스센터 예약을 도와드릴까요?\n"
        "(예: '자가수리 할게', '센터 예약해줘' 등으로 답변해 주세요)"
    )
    return {
        "answer": appended_answer,
        "waiting_for_repair_choice": True # [핵심] 다음 질문이 대답임을 명시!
    }

def self_repair_guide_node(state: GraphState) -> GraphState:
    """사용자가 자가수리를 선택했을 때의 최종 가이드 출력"""
    print("---NODE: 자가수리 매뉴얼 검색 및 안내---")
    device_model = state.get('device_model', '해당 기기')
    
    # [수정됨] self-repair 컬렉션에서 해당 기기의 매뉴얼 문서 검색
    try:
        vector_store = get_vector_store("self-repair")
        # 모델명으로 문서 검색 (예: "갤럭시 S22 자가수리 매뉴얼")
        search_query = f"{device_model} 자가수리 매뉴얼 부품"
        docs = vector_store.as_retriever(search_kwargs={"k": 2}).invoke(search_query)
        
        # 검색된 문서 내용 조립
        if docs:
            retrieved_guide = "\n\n".join([f"📖 [{d.metadata.get('title', '매뉴얼')}] \n{d.page_content}" for d in docs])
        else:
            retrieved_guide = "- 관련 매뉴얼을 찾을 수 없습니다. 삼성전자 공식 홈페이지를 참고해 주세요."
            
    except Exception as e:
        print(f"자가수리 DB 검색 오류: {e}")
        retrieved_guide = "- DB 검색 중 오류가 발생했습니다."

    # 최종 가이드 텍스트 조립
    guide_text = (
        f"🛠️ **[{device_model} 자가수리 가이드]**\n\n"
        f"{retrieved_guide}\n\n"
        "⚠️ **주의사항:** 수리 중 발생하는 추가 파손은 무상 수리 대상에서 제외될 수 있으니, "
        "반드시 절연 장갑을 착용하고 안전 수칙을 준수해 주세요."
    )
    
    # 가이드를 줬으므로 대기 상태 해제
    return {"answer": guide_text, "waiting_for_repair_choice": False}

def nearest_center_node(state: GraphState) -> GraphState:
    """무조건 센터를 안내하는 노드 (Fallback 또는 사용자 선택 시)"""
    print("---NODE: 센터 방문 안내 수행---")
    fallback_answer = (
        "기기의 정확한 상태 점검 및 안전한 조치를 위해 전문 엔지니어의 확인을 권장합니다.\n\n"
        "📍 **[가까운 삼성전자 서비스센터 찾기 및 방문 예약]**\n"
        "👉 https://www.samsungsvc.co.kr/reserve/reserveCenter"
    )
    # 센터 안내를 마쳤으므로 대기 상태 해제
    return {
        "answer": fallback_answer, 
        "source_document": "오프라인 센터 안내",
        "reliability_score": 1.0,
        "waiting_for_repair_choice": False
    }