import os
from pydantic import BaseModel, Field
from typing import Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.state import GraphState
from src.pipelines.embedding_pipeline import get_vector_store
import json # <-- 추가
import pickle
from langchain_community.retrievers import BM25Retriever
import requests
from dotenv import load_dotenv
load_dotenv()
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")


def load_self_repair_models():
    """self-repair-list.json 파일을 읽어서 모든 모델명을 1차원 리스트로 반환합니다."""
    file_path = os.path.join("data", "processed", "self-repair-list.json")


    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    all_models = []
    for series, models in data.items():
        all_models.extend(models)
    return all_models # ['S20', 'S20 Plus', 'S20 Ultra', 'S21' ...]



# 공통으로 사용할 LLM (정확한 판단을 위해 temperature=0.0 유지)
llm = ChatOpenAI(temperature=0.0, model="gpt-4-turbo")

# ==========================================
# [1] Pydantic 모델 정의 - 구조화된 출력
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
    user_intent: Literal["self_repair", "center_visit", "unknown"] = Field(
        description="사용자가 '직접 고치겠다/매뉴얼을 달라'고 하면 'self_repair', '센터를 찾는다'면 'center_visit', 불확실하거나 질문만 하면 'unknown'"
    )

# ==========================================
# [2] 조건부 라우팅 함수 (Conditional Edges)
# ==========================================
def route_question(state: GraphState) -> str:
    """(1) 진입점 라우터"""
    print("---ROUTING: 진입점 및 의도 분류 중---")
    
    # 일반적인 새 질문인 경우
    structured_llm = llm.with_structured_output(RouteQuery)
    result = structured_llm.invoke(state["messages"][-1].content)
    
    if result.intent == "greeting":
        return "chat_node"
    return "retrieve_node"


# ==========================================
# [3] 실제 작업을 수행하는 노드들 (Nodes)
# ==========================================

def chat_node(state: GraphState) -> GraphState:
    print("---NODE: 일반 대화---")
    
    prompt = f"""당신은 삼성전자 서비스센터의 전문적이고 친절한 인공지능 어시스턴트입니다.
고객이 인사를 하거나 일반적인 질문, 잡담을 시도하고 있습니다. 
따뜻하고 정중한 톤으로 응답하며, 서비스 문의나 기술적으로 도와드릴 일이 있는지 물어보세요.

[고객 사전 선택 정보]
선택한 기기: {state.get("selected_device", "선택하지 않음")}
"""
    sys_msg = SystemMessage(content=prompt)
    response = llm.invoke([sys_msg] + state["messages"])
    
    return {
        "messages": [response], 
        "source_document": "대화형 AI", 
        "reliability_score": 1.0
    }


def retrieve_node(state: GraphState) -> GraphState:
    print("---NODE: 문서 검색 (Vector DB 단독)---")
    question = state["messages"][-1].content
    selected_device = state.get("selected_device", "선택하지 않음")
    
    # 1. Chroma DB (벡터 검색) 호출
    vector_store = get_vector_store("faq")
    
    # 사전에 선택된 기기명이 있다면 쿼리에 포함하여 검색 정확도를 높임
    search_query = question if selected_device in ["선택하지 않음", "기타"] else f"{selected_device} {question}"
    top_docs = vector_store.as_retriever(search_kwargs={"k": 2}).invoke(search_query)
    
    # 2. Context 구성 
    if not top_docs:
        context = "검색된 문서 없음"
    else:
        context_list = []
        for d in top_docs:
            title = d.metadata.get('title', 'Unknown')
            content = d.metadata.get('cleaned_content', d.page_content)
            context_list.append(f"[{title}]\n{content}")
        
        context = "\n\n".join(context_list)

    return {"context": context}

def generate_node(state: GraphState) -> GraphState:
    print("---NODE: 1차 답변 생성---")
    
    prompt = f"""당신은 삼성전자 제품의 문제를 해결해주는 최고의 기술 지원 AI 어시스턴트입니다.
사용자가 제기한 문제(소프트웨어 오류, 기기 설정, 사용법 등)에 대해 아래 [관련 매뉴얼 문서]를 참조하여 명확하고 친절한 해결책을 제시하세요.
- 매뉴얼에 없는 내용을 임의로 지어내지 마세요.
- 단계별로 쉽게 따라할 수 있도록 설명해 주세요.

[고객 사전 선택 정보]
선택한 기기: {state.get("selected_device", "선택하지 않음")}

[관련 매뉴얼 문서]
{state.get('context', '검색된 내용 없음')}
"""
    sys_msg = SystemMessage(content=prompt)
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response], "source_document": "내부 매뉴얼", "reliability_score": 0.9}

def load_self_repair_json_str():
    file_path = os.path.join("data", "processed", "self-repair-list.json")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def self_repair_classifier_node(state: GraphState) -> GraphState:
    """하드웨어 문제 시, 기기 모델명, 하드웨어 여부, 그리고 사용자의 '자가수리 의향'을 동시에 추출합니다."""
    print("---NODE: 자가수리 분류기 (기기, 파손, 수리의향 동시 판별)---")
    
    # 1. 자가수리 가능한 기기 목록 데이터 불러오기
    repair_db_str = load_self_repair_json_str()
    repair_models_list = load_self_repair_models()
    selected_device = state.get("selected_device", "선택하지 않음")
    
    # 2. 세밀한 판별을 위한 프롬프트 구성
    prompt = f"""
다음은 사용자의 문의에서 1) 하드웨어 문제인지, 2) 자가수리가 가능한 기기인지, 3) 수리 의향은 무엇인지 종합적으로 분석하는 작업입니다.

[사용자가 사전에 선택한 기기]
{selected_device}

[자가수리 지원 기기 모델 목록]
{repair_db_str}

분석 지침:
1. 기기 모델명 (device_model): 사용자가 사전에 기기를 선택했다면 그것을 우선 고려하되, 질문 내용에서 다른 기기가 명백하게 언급되었다면 질문 속 기기명을 우선 추출하세요. 만약 추출된 기기가 위 [자가수리 지원 기기 모델 목록]에 존재한다면 공식 모델명(예: "S20 Ultra", "갤럭시 Z 플립5")을 명확하게 출력해 주세요. 목록에 없다면 파악된 이름을 그대로 적습니다.
2. 하드웨어 이슈 (is_hardware_issue): 액정, 뒷면 유리, 배터리 등 물리적인 교체가 필요한 파손/고장인가요?
3. 사용자 의향 (user_intent): 
   - '내가 고칠래', '매뉴얼 줘', '부품 줘' 등 본인이 수리하려는 의지면 'self_repair'
   - '센터 갈래', '센터 찾아줘', '예약해줘' 등 센터 방문 의지면 'center_visit'
   - 단순히 고장을 호소하며 어떻게 할지 묻기만 하거나 의향이 모호하다면 'unknown'

관련 문서: {state.get("context", "")}
"""
    
    # 3. LLM 호출
    sys_msg = SystemMessage(content=prompt)
    structured_llm = llm.with_structured_output(SelfRepairExtraction)
    result = structured_llm.invoke([sys_msg] + state["messages"])
    
    device_model = result.device_model
    is_hw = result.is_hardware_issue
    user_intent = result.user_intent
    
    # 4. 안전을 위한 2차 파이썬 로직 검증 (LLM이 뽑아준 device_model이 실제 목록에 있는지)
    is_repairable = False
    if is_hw and device_model:
        # LLM이 JSON의 공식 명칭을 정확히 뽑아줬는지 확인
        if device_model in repair_models_list:
            is_repairable = True
        else:
            # 보수적으로 부분 일치 검색
            cleaned_device_model = device_model.lower().replace(" ", "").replace("galaxy", "갤럭시").replace("울트라", "ultra").replace("플러스", "plus")
            for m in repair_models_list:
                cleaned_m = m.lower().replace(" ", "").replace("galaxy", "갤럭시").replace("울트라", "ultra").replace("플러스", "plus")
                if cleaned_m in cleaned_device_model or cleaned_device_model in cleaned_m:
                    is_repairable = True
                    device_model = m # 공식 명칭으로 교체
                    break
                    
    return {
        "device_model": device_model,
        "is_hardware_issue": is_hw,
        "waiting_for_repair_choice": is_repairable and user_intent == "self_repair", 
    }

def self_repair_guide_node(state: GraphState) -> GraphState:
    """사용자가 자가수리를 선택했을 때의 최종 가이드 출력"""
    print("---NODE: 자가수리 매뉴얼 검색 및 안내---")
    device_model = state.get('device_model', '해당 기기')
    
    # self-repair 컬렉션에서 해당 기기의 매뉴얼 문서 검색
    vector_store = get_vector_store("self-repair")
    search_query = f"{device_model} 자가수리 매뉴얼 부품" # query 작성...
    #자가수리 데이터 RAG로 추가
    docs = vector_store.as_retriever(search_kwargs={"k": 2}).invoke(search_query)
    
    # 검색된 문서 내용 조립
    if docs:
        retrieved_guide = "\n\n".join([f"📖 [{d.metadata.get('title', '매뉴얼')}] \n{d.page_content}" for d in docs])
    else:
        retrieved_guide = "- 관련 매뉴얼을 찾을 수 없습니다. 삼성전자 공식 홈페이지를 참고해 주세요."
        
    # 최종 가이드 텍스트 조립
    guide_text = (
        f"🛠️ **[{device_model} 자가수리 가이드]**\n\n"
        f"{retrieved_guide}\n\n"
        " **주의사항:** 수리 중 발생하는 추가 파손은 무상 수리 대상에서 제외될 수 있으니, "
        "반드시 절연 장갑을 착용하고 안전 수칙을 준수해 주세요."
    )
    
    # 가이드를 줬으므로 대기 상태 해제
    return {"messages": [("assistant", guide_text)], "waiting_for_repair_choice": False}

import requests
import os

def get_kakao_nearest_centers(lat: float, lng: float) -> str:
    #카카오 API를 호출하여 내 주변 삼성전자 서비스센터를 찾는 함수
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {
        "Authorization": f"KakaoAK {KAKAO_API_KEY}"
    }
    
    params = {
        "query": "삼성전자 서비스센터",
        "y": 37.498095, 
        "x": 127.027610,
        "radius": 5000,
        "sort": "distance"
    }

    try:
        response = requests.get(url, headers=headers, params=params).json()
        documents = response.get("documents", [])
        
        if documents:
            centers = documents[:3] # 가장 가까운 3개만 추출
            
            result_text = "기기의 정확한 상태 점검 및 안전한 조치를 위해 가장 가까운 전문 엔지니어의 확인을 권장합니다.\n\n📍 **[가까운 삼성전자 서비스센터]**\n"
            
            for i, center in enumerate(centers):
                name = center.get("place_name")
                address = center.get("road_address_name") or center.get("address_name")
                distance = center.get("distance")
    
                # 거리를 보기 좋게 변환
                dist_str = f"{int(distance)}m" if int(distance) < 1000 else f"{int(distance)/1000:.1f}km"
                
                result_text += f"{i+1}. **{name}** ({dist_str} 거리)\n   - 주소: {address}\n"
                
            result_text += "\n👉 **[방문 예약하기]**: https://www.samsungsvc.co.kr/reserve/reserveCenter"
            return result_text
            
    except Exception as e:
        print(f"Kakao API 호출 에러: {e}")
        
    return None

def nearest_center_node(state: dict) -> dict: # LangGraph State 타입
    print("---NODE: 동적 센터 방문 안내 수행 (카카오 API)---")
    
    # State에서 사용자의 위치 정보를 가져옵니다.
    # (프론트엔드에서 브라우저 API로 받아온 좌표값을 state에 넣어주면 됩니다)
    user_lat = state.get("latitude", 37.4952) 
    user_lng = state.get("longitude", 127.0276)
    
    dynamic_answer = get_kakao_nearest_centers(user_lat, user_lng)
    
    if not dynamic_answer:
        dynamic_answer = "안되는디요"
        
    return {
        "messages": [("assistant", dynamic_answer)], 
        "source_document": "위치 기반 오프라인 센터 안내",
        "reliability_score": 1.0,
        "waiting_for_repair_choice": False
    }
def route_issue_type(state: GraphState) -> str:
    print("---ROUTING: 이슈 타입 분류 중---")
    
    prompt = f"""사용자의 질문이 어떤 유형에 속하는지 분류하세요.
[사전 선택 기기: {state.get("selected_device", "선택하지 않음")}]

분류 지침:
아래에 해당하는 증상이나 문의는 모두 'software' 로 분류하세요:
- 전원/배터리/충전 (예: 발열 현상, 배터리 소모 심함 등)
- 블루투스
- 멈춤/오류/재시작
- 시스템 설정
- 데이터 이동
- 네트워크/WI-FI
- 카메라/갤러리 앱 오류 및 기능 문의
- 디스플레이 (화면 설정, 다크모드 등)
- 애플리케이션 (앱 설치, 삭제, 오류)
- 전화/문자 (통화 품질, 수발신 문제 등)
- 센서/터치 설정 및 오류
- 소리/진동 설정
- 업데이트 (OS, SW 업데이트)
- 사양/구성품 문의
- 액세서리
- 이동통신사 서비스
- 기타/주의사항
"""
    
    sys_msg = SystemMessage(content=prompt)
    structured_llm = llm.with_structured_output(IssueTypeCheck)
    result = structured_llm.invoke([sys_msg] + state["messages"])
    
    if result.issue_type == "software":
        return "generate_node"
    elif result.issue_type == "hardware":
        return "self_repair_classifier_node"
    else: # center_visit
        return "nearest_center_node"

def route_after_self_repair_check(state: GraphState) -> str:
    print("---ROUTING: 자가수리 라우팅 체크---")
    
    if state.get("waiting_for_repair_choice"):
        return "self_repair_guide_node"
        
    return "nearest_center_node"

def check_hallucination_routing(state: GraphState) -> str:
    print("---ROUTING: 환각 체크---")
    score = state.get("reliability_score", 1.0)
    if score >= 0.8:
        return "END"
    else:
        return "nearest_center_node"