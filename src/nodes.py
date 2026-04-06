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
 
 
llm = ChatOpenAI(temperature=0.0, model="gpt-4-turbo")
 
# ==========================================
# [1] Pydantic 모델 정의
# ==========================================
class RouteQuery(BaseModel):
    intent: Literal["greeting", "cs_query", "center_visit"] = Field(
        description="""이전 대화 맥락 전체를 고려하여 분류하세요.
- 'greeting': 인사, 잡담, 요약 요청
- 'cs_query': 새로운 기기 고장이나 서비스 문의. 이전에 센터 안내를 받았더라도 새로운 기기 문제를 언급하면 반드시 'cs_query'로 분류하세요.
- 'center_visit': 센터 찾기, 예약 요청. 수리 관련 잡담이나 질문은 절대 'center_visit'으로 분류하지 마세요.
  또는 이전 대화에서 이미 기기 문제 해결을 시도했는데 아직 해결이 안 됐다는 맥락이면 'center_visit'으로 분류하세요.
"""
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
# [2] 조건부 라우팅 함수
# ==========================================
def route_issue_type(state: GraphState) -> str:
    print("---ROUTING: 이슈 타입 분류 중---")
    
    if state.get("context") == "검색된 문서 없음":
        return "fallback_node"
    
    prompt = f"""사용자의 질문이 어떤 유형인지 분류하세요.
[사전 선택 기기: {state.get("selected_device", "선택하지 않음")}]

1. 'hardware': 자가수리, 배터리 교체, 액정 교체 등 물리적으로 직접 수리하려는 의향이 있는 경우
2. 'software': 그 외 모든 기기 문제 및 설정 문의
3. 'center_visit': 센터 위치나 예약을 명시적으로 요청하는 경우
"""
    sys_msg = SystemMessage(content=prompt)
    structured_llm = llm.with_structured_output(IssueTypeCheck)
    result = structured_llm.invoke([sys_msg] + state["messages"])
    
    if result.issue_type == "hardware":
        return "self_repair_classifier_node"
    elif result.issue_type == "center_visit":
        return "nearest_center_node"
    return "generate_node"
 
    
def route_after_self_repair_check(state: GraphState) -> str:
    print("---ROUTING: 자가수리 라우팅 체크---")
    
    if state.get("waiting_for_repair_choice"):
        return "self_repair_guide_node"
        
    return "nearest_center_node"
 


def route_question(state: GraphState) -> str:
    print("---ROUTING: 진입점 및 의도 분류 중---")
    structured_llm = llm.with_structured_output(RouteQuery)
    recent_messages = state["messages"][-2:]
    result = structured_llm.invoke(recent_messages)
    
    if result.intent == "greeting":
        return "chat_node"
    elif result.intent == "center_visit":
        return "nearest_center_node"
    return "retrieve_node"


# ==========================================
# [3] 노드들
# ==========================================
def chat_node(state: GraphState) -> GraphState:
    print("---NODE: 일반 대화---")
    
    last_msg = state["messages"][-1].content
    
    # 요약 요청이면 전체 히스토리, 아니면 최근 5개
    if any(keyword in last_msg for keyword in ["요약", "정리", "뭐라고", "어떻게 됐"]):
        history_text = "\n".join([f"{'사용자' if msg.type=='human' else '상담원'}: {msg.content}" for msg in state["messages"]])
    else:
        history_text = "\n".join([f"{'사용자' if msg.type=='human' else '상담원'}: {msg.content}" for msg in state["messages"][-5:]])
    
    prompt = f"""당신은 삼성전자 갤럭시 고객 서비스 어시스턴트입니다.
고객의 인사나 일상 대화에 친절하고 따뜻하게 응대하세요.
응대 후 기기 관련 문의가 있는지 자연스럽게 유도하세요.
 
[응대 예시]
고객: 안녕하세요!
답변: 안녕하세요! 😊 삼성 갤럭시 고객 서비스입니다. 오늘 어떤 도움이 필요하신가요?
 
고객: 너 이름이 뭐야?
답변: 저는 삼성 갤럭시 서비스 어시스턴트예요! 기기 관련 문의나 수리 안내가 필요하시면 편하게 말씀해 주세요. 😊
 
고객: 고마워!
답변: 천만에요! 😊 더 궁금한 점이 있으시면 언제든지 문의해 주세요. 도움이 필요하시면 말씀해 주세요!
 
고객: 잘 모르겠어
답변: 걱정하지 마세요! 😊 어떤 기기를 사용 중이신지, 어떤 문제가 있으신지 말씀해 주시면 제가 도와드릴게요.

고객: 해결이 안되었어
답변: 불편을 드려서 죄송합니다. 😟 가까운 삼성전자 서비스센터 방문을 권장드립니다.
👉 https://www.samsungsvc.co.kr/reserve/reserveCenter

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
    
    # LLM으로 쿼리 변환
    query_prompt = f"""사용자의 질문을 삼성 갤럭시 FAQ 검색에 적합한 키워드로 변환하세요.
공식적인 표현과 관련 키워드를 포함하세요.
짧게 한 줄로만 답하세요.
 
예시:
질문: 핸드폰이 뜨거워 → 갤럭시 기기 발열 현상 뜨거워지는
질문: 배터리 빨리 닳아 → 갤럭시 배터리 소모 빠름 방전
질문: 화면이 안 켜져 → 갤럭시 화면 검은 화면 전원 안 켜짐
질문: 소리가 안 나 → 갤럭시 소리 안남 무음 스피커
 
질문: {question} →"""
    
    converted_query = llm.invoke(query_prompt).content.strip()
    print(f"변환된 쿼리: {converted_query}")
    
    # 1. Chroma DB (유사도 점수 포함)
    vector_store = get_vector_store("faq")
    scored_docs = vector_store.similarity_search_with_score(converted_query, k=5)
    top_score = scored_docs[0][1] if scored_docs else 0
    vector_docs = [doc for doc, score in scored_docs]
    print(f"최고 유사도 점수: {top_score:.4f}")
    
    # 2. BM25 (키워드 검색)
    bm25_path = os.path.join("data", "bm25_index", "bm25_corpus.pkl")
    with open(bm25_path, "rb") as f:
        bm25_documents = pickle.load(f)
    bm25_retriever = BM25Retriever.from_documents(bm25_documents, k=5)
    bm25_docs = bm25_retriever.invoke(question)
    
    # 3. 두 결과 합치기 (중복 제거)
    seen_titles = set()
    combined_docs = []
    for d in vector_docs + bm25_docs:
        title = d.metadata.get('title', '')
        if title not in seen_titles:
            seen_titles.add(title)
            combined_docs.append(d)
    
    print(f"검색 쿼리: {converted_query}")
    for i, d in enumerate(combined_docs[:5]):
        print(f"검색결과 {i+1}: {d.metadata.get('title', 'Unknown')}")
    
    if not combined_docs:
        context = "검색된 문서 없음"
    else:
        context_list = []
        for d in combined_docs[:5]:
            title = d.metadata.get('title', 'Unknown')
            content = d.metadata.get('cleaned_content', d.page_content)
            context_list.append(f"[{title}]\n{content}")
        context = "\n\n".join(context_list)
 
    return {"context": context, "relevance_score": float(top_score)}
 
def generate_node(state: GraphState) -> GraphState:
    print("---NODE: 1차 답변 생성---")
    
    history_text = "\n".join([f"{'사용자' if msg.type=='human' else '상담원'}: {msg.content}" for msg in state["messages"][-5:]])
    
    prompt = f"""당신은 삼성전자 제품 전문 기술 지원 AI 어시스턴트입니다.
아래 [규칙]을 반드시 지켜서 고객 문의에 답변하세요.
 
[규칙]
1. 반드시 아래 제공된 [관련 매뉴얼 문서] 내용에만 근거하여 답변하세요.
2. 이전 대화 맥락(최근 대화 기록)을 고려하여 답변하세요.
3. 문서에 없는 내용은 절대 지어내거나 추측하지 마세요.
   - 문서에 없는 해결 방법을 절대 추가하지 마세요.
   - 문서에 서비스센터 방문 언급이 없으면 절대 언급하지 마세요.
   - 문서 내용이 없으면 오직 아래 문구로만 답하세요.
    "제공된 정보에서 해당 내용을 찾을 수 없습니다.
    정확한 점검을 위해 가까운 삼성전자 서비스센터 방문을 권장드립니다.
    👉 https://www.samsungsvc.co.kr/reserve/reserveCenter"
4. 전문 용어는 쉽게 풀어서 설명하세요.
5. 해결 방법을 번호로 단계별 안내하세요.
6. 답변 마지막에 반드시 "위 방법으로 해결이 안 되시면 말씀해 주세요." 를 추가하세요.
 
[답변 형식]
- 문제 원인을 1~2문장으로 간단히 설명
- 해결 방법을 번호로 단계별 안내
- 마지막에 "위 방법으로 해결이 안 되시면 말씀해 주세요." 고정 문구
 
[답변 예시]
고객 문의: 화면 터치가 잘 안 돼요
답변:
화면 터치 문제는 일시적인 오류나 보호필름, 터치 설정으로 인해 발생할 수 있습니다.
 
아래 방법을 순서대로 시도해 보세요.
1. 기기를 재부팅해 보세요. (전원 버튼 + 음량 하 버튼을 7초 이상 누르세요)
2. 보호필름이 부착된 경우, 설정 → 디스플레이 → 터치 민감도를 높여보세요.
3. TalkBack 기능이 켜져 있다면 설정 → 접근성 → TalkBack을 비활성화하세요.
 
위 방법으로 해결이 안 되시면 말씀해 주세요.
 
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
        "reliability_score": 0.9,
        "show_resolution_buttons": True  # ← 추가
    }
 
def self_repair_classifier_node(state: GraphState) -> GraphState:
    print("---NODE: 자가수리 분류기 (기기, 파손, 수리의향 동시 판별)---")
    
    repair_db_str = load_self_repair_json_str()
    repair_models_list = load_self_repair_models()
    selected_device = state.get("selected_device", "선택하지 않음")
    
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
    
    sys_msg = SystemMessage(content=prompt)
    structured_llm = llm.with_structured_output(SelfRepairExtraction)
    result = structured_llm.invoke([sys_msg] + state["messages"])
    
    device_model = result.device_model
    is_hw = result.is_hardware_issue
    user_intent = result.user_intent
    
    is_repairable = False
    if is_hw and device_model:
        if device_model in repair_models_list:
            is_repairable = True
        else:
            cleaned_device_model = device_model.lower().replace(" ", "").replace("galaxy", "갤럭시").replace("울트라", "ultra").replace("플러스", "plus")
            for m in repair_models_list:
                cleaned_m = m.lower().replace(" ", "").replace("galaxy", "갤럭시").replace("울트라", "ultra").replace("플러스", "plus")
                if cleaned_device_model == cleaned_m or cleaned_device_model.endswith(cleaned_m) or cleaned_m.endswith(cleaned_device_model):
                    is_repairable = True
                    device_model = m
                    break
                    
    return {
        "device_model": device_model,
        "is_hardware_issue": is_hw,
        "waiting_for_repair_choice": is_repairable and user_intent == "self_repair", 
    }
 
 
def self_repair_guide_node(state: GraphState) -> GraphState:
    print("---NODE: 자가수리 매뉴얼 검색 및 안내---")
    device_model = state.get('device_model')
    if not device_model or device_model == '해당 기기':
        device_model = None
    question = state["messages"][-1].content
    
    vector_store = get_vector_store("self-repair")
    
    try:
        available_models = get_available_models(vectorstore=vector_store)
    except Exception:
        available_models = []
 
    chain, _ = make_rag_chain(vector_store, available_models, session_model=device_model, k=8)
    answer = chain.invoke(question)
    
    device_display = device_model if device_model else "해당 기기"
    guide_text = (
        f"🛠️ **[{device_display} 자가수리 가이드]**\n\n"
        f"{answer}\n\n"
        " **주의사항:** 수리 중 발생하는 추가 파손은 무상 수리 대상에서 제외될 수 있으니, "
        "반드시 절연 장갑을 착용하고 안전 수칙을 준수해 주세요."
    )
    
    return {"messages": [("assistant", guide_text)], "waiting_for_repair_choice": False}
 
import requests
import os
 
def get_kakao_nearest_centers(lat: float, lng: float) -> str:
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {
        "Authorization": f"KakaoAK {KAKAO_API_KEY}"
    }
    
    params = {
        "query": "삼성전자 서비스센터",
        "y": lat,
        "x": lng,
        "radius": 5000,
        "sort": "distance"
    }
 
    try:
        response = requests.get(url, headers=headers, params=params).json()
        documents = response.get("documents", [])
        
        if documents:
            centers = documents[:3]
            result_text = "기기의 정확한 상태 점검 및 안전한 조치를 위해 가장 가까운 전문 엔지니어의 확인을 권장합니다.\n\n📍 **[가까운 삼성전자 서비스센터]**\n"
            
            for i, center in enumerate(centers):
                name = center.get("place_name")
                address = center.get("road_address_name") or center.get("address_name")
                distance = center.get("distance")
                dist_str = f"{int(distance)}m" if int(distance) < 1000 else f"{int(distance)/1000:.1f}km"
                result_text += f"{i+1}. **{name}** ({dist_str} 거리)\n   - 주소: {address}\n"
                
            result_text += "\n👉 **[방문 예약하기]**: https://www.samsungsvc.co.kr/reserve/reserveCenter"
            return result_text
            
    except Exception as e:
        print(f"Kakao API 호출 에러: {e}")
        
    return None
 
def nearest_center_node(state: dict) -> dict:
    print("---NODE: 동적 센터 방문 안내 수행 (카카오 API)---")
    
    user_lat = state.get("latitude", 37.4952) 
    user_lng = state.get("longitude", 127.0276)
    
    dynamic_answer = get_kakao_nearest_centers(user_lat, user_lng)
    
    if not dynamic_answer:
        dynamic_answer = (
            "현재 주변 서비스센터 정보를 불러오지 못했습니다.\n\n"
            "📍 **[삼성전자 서비스센터 찾기]**\n"
            "👉 https://www.samsungsvc.co.kr/reserve/reserveCenter\n"
            "위 링크에서 직접 가까운 서비스센터를 검색하고 방문 예약하실 수 있습니다."
        )
        
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
