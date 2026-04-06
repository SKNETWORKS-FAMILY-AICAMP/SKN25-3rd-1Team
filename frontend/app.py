import streamlit as st
import uuid
import time
import os
import json
import pandas as pd
from api.client import get_chat_response

# =========================================================
# 1. 데이터 로드 로직 (운구님 로직 100% 유지 + 안전장치)
# =========================================================
@st.cache_data
def load_device_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "data", "processed", "self-repair-list.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception: return {}

@st.cache_data
def load_faq_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate_paths = [
        os.path.join(base_dir, "data", "processed", "faq", "faq_data_v4.csv"),
        os.path.join(base_dir, "data", "processed", "faq", "faq_data_v3.csv"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                required_defaults = {
                    "symptom_category": "기타/주의사항", "title": "", "url": "",
                    "cleaned_content": "", "viewCnt": 0, "exposureDate": None,
                }
                for col, default in required_defaults.items():
                    if col not in df.columns: df[col] = default
                df["symptom_category"] = df["symptom_category"].fillna("기타/주의사항").astype(str)
                df["title"] = df["title"].fillna("").astype(str)
                df["url"] = df["url"].fillna("").astype(str)
                df["cleaned_content"] = df["cleaned_content"].fillna("").astype(str)
                df["viewCnt"] = pd.to_numeric(df["viewCnt"], errors="coerce").fillna(0).astype(int)
                df["exposureDate"] = pd.to_datetime(df["exposureDate"], errors="coerce")
                return df
            except Exception: continue
    return pd.DataFrame(columns=["title", "symptom_category", "url", "viewCnt", "exposureDate", "cleaned_content"])

device_data = load_device_data()
faq_df = load_faq_data()

# =========================================================
# 2. 페이지 설정 및 프리미엄 CSS
# =========================================================
st.set_page_config(page_title="스마트 고객센터", page_icon="💬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

* { font-family: 'Pretendard', sans-serif !important; box-sizing: border-box; }
.stApp { background-color: #F8FAFC !important; }
header, footer, [data-testid="stToolbar"] { display: none !important; }

.block-container {
    max-width: 950px !important;
    padding-top: 2rem !important;
    padding-bottom: 10rem !important;
    margin: 0 auto !important;
}

.main-card {
    background: #ffffff;
    border-radius: 20px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    overflow: hidden;
}

.hero {
    padding: 3rem 2rem;
    text-align: center;
    background: #ffffff;
    border-bottom: 1px solid #F1F5F9;
}
.hero-h1 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.3;
    margin-bottom: 0.8rem;
}
.hero-sub { font-size: 1rem; color: #64748B; }

.device-chip {
    display: inline-block;
    background: #EFF6FF;
    color: #2563EB;
    padding: 3px 12px;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 700;
    border: 1px solid #DBEAFE;
    margin-top: 10px;
}

/* 인기 질문 카드 섹션용 추가 스타일 */
.popular-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #94A3B8 !important;
    margin-bottom: 1rem;
    margin-top: 1rem;
}

div[data-popq] .stButton > button {
    background-color: #FFFFFF !important;
    color: #334155 !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    height: 100px !important; 
    width: 100% !important;
    padding: 1rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    line-height: 1.4 !important;
    word-break: keep-all !important;
    white-space: normal !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    transition: all 0.2s !important;
}

div[data-popq] .stButton > button:hover {
    border-color: #3B82F6 !important;
    color: #3B82F6 !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.05) !important;
}

/* 인기 질문 버튼 텍스트 말줄임 처리 (Image 1 스타일) */
div[data-popq] .stButton > button > div > p {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    display: block !important;
    max-width: 100% !important;
}

/* 채팅 메시지 */
[data-testid="stChatMessageContent"] {
    background-color: #F1F5F9 !important;
    border-radius: 15px !important;
}

/* 자주 묻는 질문 스타일 */
.faq-subtitle {
    font-size: 0.9rem;
    color: #94A3B8;
    margin-top: -0.5rem;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. 세션 초기화 및 유틸리티 함수
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요? 기기 모델을 선택하시면 더 정확한 상담이 가능합니다. 😊"}]
if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())
if "view" not in st.session_state: st.session_state.view = "chat"
if "selected_category" not in st.session_state: st.session_state.selected_category = "전원/배터리/충전"
if "faq_keyword" not in st.session_state: st.session_state.faq_keyword = ""
if "faq_sort" not in st.session_state: st.session_state.faq_sort = "최신순"
if "selected_device" not in st.session_state: st.session_state.selected_device = "선택하지 않음"

def ask_ai(question):
    """AI 상담 요청 및 상태 업데이트 공통 함수"""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("해결 방안을 찾는 중입니다..."):
        response = get_chat_response(question, st.session_state.selected_device, st.session_state.thread_id)
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.view = "chat"
    st.rerun()

# =========================================================
# 4. 사이드바
# =========================================================
with st.sidebar:
    st.markdown("<h2 style='color:#2563EB; font-weight:800;'>Smart CS</h2>", unsafe_allow_html=True)
    st.divider()
    
    if st.button(" AI 채팅 상담", use_container_width=True, type="primary" if st.session_state.view == "chat" else "secondary"):
        st.session_state.view = "chat"; st.rerun()
    if st.button(" 자주 묻는 질문", use_container_width=True, type="primary" if st.session_state.view == "faq" else "secondary"):
        st.session_state.view = "faq"; st.rerun()
    
    st.markdown("<br><p style='font-size:0.8rem; font-weight:700; color:#64748B;'>기기 모델 설정</p>", unsafe_allow_html=True)
    series_options = ["선택하지 않음"] + list(device_data.keys()) + ["기타"]
    selected_series = st.selectbox("시리즈 선택", options=series_options, index=0, label_visibility="collapsed")
    
    selected_device = "선택하지 않음"
    if selected_series not in ["선택하지 않음", "기타"]:
        selected_device = st.selectbox("상세 모델 선택", options=device_data[selected_series], label_visibility="collapsed")
    elif selected_series == "기타":
        selected_device = "기타 기기"
    st.session_state.selected_device = selected_device

    st.divider()
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "대화가 초기화되었습니다. 무엇이든 물어보세요! 😊"}]
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# =========================================================
# 5. 메인 레이아웃 (헤더)
# =========================================================
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

device_display = f"<div class='hero-sub' style='margin-top:10px;'>기기: {st.session_state.selected_device}</div>"
st.markdown(f"""
<div class='hero'>
    <div class='hero-h1'>궁금한 점을 질문하시면<br>빠르게 답변드리겠습니다</div>
    {device_display}
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='padding: 2rem;'>", unsafe_allow_html=True)

# =========================================================
# 6. 인기 질문 섹션 (2행 3열 카드) - 추가됨
# =========================================================
if not faq_df.empty:
    st.markdown("<div class='popular-title'>인기 질문</div>", unsafe_allow_html=True)
    top_faqs = faq_df.sort_values("viewCnt", ascending=False).head(6)["title"].tolist()
    
    for i in range(0, len(top_faqs), 3):
        row = top_faqs[i:i+3]
        cols = st.columns(3)
        for idx, q_title in enumerate(row):
            with cols[idx]:
                st.markdown('<div data-popq="">', unsafe_allow_html=True)
                if st.button(q_title, key=f"pop_{i}_{idx}", use_container_width=True):
                    ask_ai(q_title)
                st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# =========================================================
# 7. 상담 및 FAQ 뷰 처리 (cleaned_content 답변 연동)
# =========================================================
if st.session_state.view == "chat":
    st.markdown("<h3 style='margin-bottom:1.5rem;'>AI 상담 서비스</h3>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("증상이나 궁금한 점을 입력하세요"):
        with st.chat_message("user"): st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("해결 방안을 찾는 중..."):
                response = get_chat_response(prompt, st.session_state.selected_device, st.session_state.thread_id)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

else: # FAQ 뷰
    st.markdown("<h2 style='font-size:1.5rem; font-weight:800; margin-bottom:0.5rem;'>자주 묻는 질문</h2>", unsafe_allow_html=True)
    st.markdown("<p class='faq-subtitle'>증상 카테고리를 선택하시면 관련 정보를 확인하실 수 있습니다</p>", unsafe_allow_html=True)
    
    # 정규화된 카테고리 리스트 (Image 1 기준)
    category_list = [
        "전원/배터리/충전","블루투스","멈춤/오류/재시작","시스템 설정","데이터이동",
        "네트워크/WI-FI","카메라/갤러리","디스플레이","애플리케이션","전화/문자",
        "센서/터치","소리/진동","업데이트","사양/구성품","액세서리",
        "이동통신사서비스","기타/주의사항"
    ]
    
    # 5열 그리드 레이아웃
    for i in range(0, len(category_list), 5):
        cols = st.columns(5)
        row_items = category_list[i : i + 5]
        for idx, cat in enumerate(row_items):
            with cols[idx]:
                if st.button(cat, key=f"cat_{cat}", use_container_width=True, 
                             type="primary" if st.session_state.selected_category == cat else "secondary"):
                    st.session_state.selected_category = cat
                    st.rerun()

    st.divider()
    
    # 검색 및 정렬
    s_col1, s_col2 = st.columns([3, 1])
    with s_col1:
        keyword = st.text_input("질문 내용 검색", value=st.session_state.faq_keyword, placeholder="검색어를 입력하세요...", label_visibility="collapsed")
        st.session_state.faq_keyword = keyword
    with s_col2:
        sort_val = st.selectbox("정렬 기준", ["최신순", "조회순", "제목순"], label_visibility="collapsed")
        st.session_state.faq_sort = sort_val

    # 데이터 필터링
    df = faq_df.copy()
    if st.session_state.selected_category:
        df = df[df["symptom_category"] == st.session_state.selected_category]
    if keyword:
        df = df[df["title"].str.contains(keyword, case=False) | df["cleaned_content"].str.contains(keyword, case=False)]
    
    # 정렬 처리
    if sort_val == "최신순": df = df.sort_values("exposureDate", ascending=False)
    elif sort_val == "조회순": df = df.sort_values("viewCnt", ascending=False)
    else: df = df.sort_values("title")

    st.markdown(f"<p style='color:#94A3B8; font-size:0.85rem; margin-top:1rem;'>총 {len(df)}건의 검색 결과</p>", unsafe_allow_html=True)

    # FAQ 리스트 출력 (중요: cleaned_content 연동 부분)
    for idx, row in df.head(15).iterrows():
        c_main, c_btn = st.columns([5, 1])
        with c_main:
            st.markdown(f"""
            <div class='faq-row'>
                <div class='faq-title-link'>{row['title']}</div>
                <div class='faq-meta'>👁️ 조회수 {row['viewCnt']:,}</div>
            </div>""", unsafe_allow_html=True)
        with c_btn:
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            st.markdown("<div data-askbtn=''>", unsafe_allow_html=True)
            if st.button("AI 상담", key=f"ask_btn_{idx}", use_container_width=True):
                # 1. 채팅창에 질문 추가
                st.session_state.messages.append({"role": "user", "content": f"**{row['title']}**에 대해 상담하고 싶어요."})
                
                # 2. 답변 생성 (CSV 내용 우선 활용)
                with st.spinner("답변을 준비 중입니다..."):
                    if pd.notnull(row['cleaned_content']) and len(str(row['cleaned_content']).strip()) > 5:
                        ai_answer = f"해당 증상에 대한 해결 방법입니다.\n\n{row['cleaned_content']}"
                    else:
                        # 내용이 없으면 RAG API 호출
                        ai_answer = get_chat_response(row['title'], st.session_state.selected_device, st.session_state.thread_id)
                    
                    st.session_state.messages.append({"role": "assistant", "content": ai_answer})
                
                # 3. 채팅 뷰로 전환 및 화면 갱신
                st.session_state.view = "chat"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True) # padding div
st.markdown("</div>", unsafe_allow_html=True) # main-card
