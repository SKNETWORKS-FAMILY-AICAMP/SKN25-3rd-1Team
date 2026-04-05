import os
from pydantic import BaseModel, Field
from typing import Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.state import GraphState
from src.pipelines.embedding_pipeline import get_vector_store
from src.pipelines.self_repair_rag_pipeline import make_rag_chain, get_available_models

import json
import pickle
from langchain_community.retrievers import BM25Retriever
import requests
from dotenv import load_dotenv

load_dotenv()

KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")

def load_self_repair_json_str():
    file_path = os.path.join("data", "processed", "self-repair-list.json")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_self_repair_models():
    file_path = os.path.join("data", "processed", "self-repair-list.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    models = []
    for model_list in data.values():
        models.extend(model_list)
    return models


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
    # 멀티턴 컨텍스트를 위해 히스토리 제공
    result = structured_llm.invoke(state["messages"])
    
    if result.intent == "greeting":
        return "chat_node"
    return "retrieve_node"


# ==========================================
# [3] 실제 작업을 수행하는 노드들 (Nodes)
# ==========================================
def chat_node(state: GraphState) -> GraphState:
    print("---NODE: 일반 대화---")
    
    # 5턴까지의 대화 내역 추출
    history_text = "\n".join([f"{'사용자' if msg.type=='human' else '상담원'}: {msg.content}" for msg in state["messages"][-5:]])
    
    prompt = f"""당신은 삼성전자 갤럭시 고객 서비스 어시스턴트입니다.
고객의 인사나 일상 대화에 친절하고 따뜻하게 응대하세요.
주어진 대화 맥락을 파악하고, 자연스럽게 기기 관련 문의로 유도하세요.
 
[고객 사전 선택 정보]
선택한 기기: {state.get("selected_device", "선택하지 않음")}
 
[최근 대화 기록]
{history_text}

답변:"""
    response = llm.invoke(prompt)
    
    return {
        "messages": [("assistant", response.content)], 
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
    
    history_text = "\n".join([f"{'사용자' if msg.type=='human' else '상담원'}: {msg.content}" for msg in state["messages"][-5:]])
    
    prompt = f"""당신은 삼성전자 제품 전문 기술 지원 AI 어시스턴트입니다.
아래 [규칙]을 반드시 지켜서 고객 문의에 답변하세요.
 
[규칙]
1. 반드시 아래 제공된 [관련 매뉴얼 문서] 내용에만 근거하여 답변하세요.
2. 이전 대화 맥락(최근 대화 기록)을 고려하여 답변하세요.
3. 문서에 없는 내용은 절대 지어내거나 추측하지 마세요. 문서에 내용이 없으면 "제공된 정보에서 해당 내용을 찾을 수 없습니다." 라고 답하세요.
4. 전문 용어는 쉽게 풀어서 설명하세요.
5. 해결 방법을 번호로 단계별 안내하고, 마지막에 추가 도움 안내 문구를 덧붙이세요.
  
[고객 사전 선택 정보]
선택한 기기: {state.get("selected_device", "선택하지 않음")}
 
[관련 매뉴얼 문서]
{state.get('context', '문서 없음')}
 
[최근 대화 기록]
{history_text}

답변:"""
    response = llm.invoke(prompt)
    return {
        "messages": [("assistant", response.content)],
        "source_document": "내부 매뉴얼", 
        "reliability_score": 0.9
    }
 
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
                # 소문자 변환 후 완전히 일치하거나, 이름 끝부분이 완전히 일치하는지 확인 (s22 가 s22ultra 에 잘못 매칭되는 것 방지)
                if cleaned_device_model == cleaned_m or cleaned_device_model.endswith(cleaned_m) or cleaned_m.endswith(cleaned_device_model):
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
    device_model = state.get('device_model')
    if not device_model or device_model == '해당 기기':
        device_model = None
    question = state["messages"][-1].content
    
    # self-repair 컬렉션에서 벡터스토어 로드
    vector_store = get_vector_store("self-repair")
    
    try:
        available_models = get_available_models(vectorstore=vector_store)
    except Exception:
        available_models = []

    chain, _ = make_rag_chain(vector_store, available_models, session_model=device_model, k=8)
    
    # RAG 체인 호출로 답변 생성
    answer = chain.invoke(question)
    
    # 최종 가이드 텍스트 조립
    device_display = device_model if device_model else "해당 기기"
    guide_text = (
        f"🛠️ **[{device_display} 자가수리 가이드]**\n\n"
        f"{answer}\n\n"
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
        "y": 35.179554,   # 부산광역시청 위도
        "x": 129.075641,  # 부산광역시청 경도
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
def fallback_node(state: GraphState) -> GraphState:
    print("---NODE: 검색 실패 예외 처리 (선택지 제공)---")
    
    fallback_message = (
        "말씀하신 내용에 대한 정확한 정보를 서비스 매뉴얼에서 찾지 못했어요 😢\n\n"
        "원하시는 다음 단계를 선택하시거나 내용을 다시 입력해 주세요:\n"
        "1. 📍 **가까운 서비스 스토어 위치 안내받기**\n"
        "2. 💬 **질문을 다른 방식으로 다시 하기**\n"
        "3. 🛠️ **직접 수리(자가수리)가 가능한 모델인지 확인하기**\n"
    )
    
    return {
        "messages": [("assistant", fallback_message)],
        "source_document": "시스템 안내",
        "reliability_score": 1.0,
        "waiting_for_repair_choice": False
    }

def route_issue_type(state: GraphState) -> str:
    print("---ROUTING: 이슈 타입 분류 중---")
    
    if state.get("context") == "검색된 문서 없음":
        return "fallback_node"
    
    prompt = f"""사용자의 질문이 어떤 유형에 속하는지 분류하세요.
[사전 선택 기기: {state.get("selected_device", "선택하지 않음")}]

분류 지침:
1. 'hardware' (하드웨어 문제 및 수리):
- 디스플레이 액정 파손, 뒷판/후면 커버 파손 등 물리적 파손
- "교체", "자가수리", "수리", "분해", "부품" 등의 단어가 포함되거나 스스로 고치기를 원하는 경우
- 배터리 교체 문의 (배터리 소모가 아니라 '교체'나 물리적인 장착/탈착을 의미할 때)

2. 'software' (소프트웨어 및 설정, 사용법 등):
- 전원/배터리/충전 (단순 배터리 소모 심함, 충전 안됨, 발열 등)
- 블루투스, 멈춤/오류/재시작, 시스템 설정, 데이터 이동
- 네트워크/WI-FI, 카메라/갤러리 앱 오류, 디스플레이 화면 설정/다크모드
- 애플리케이션 설치/오류, 전화/문자 수발신 문제, 센서/터치/소리/진동 설정
- 업데이트, 사양/구성품/액세서리 문의, 동기화 기타 주의사항

3. 'center_visit' (서비스센터 방문 응급):
- 당장 서비스센터 위치를 묻거나 예약을 요구할 때
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