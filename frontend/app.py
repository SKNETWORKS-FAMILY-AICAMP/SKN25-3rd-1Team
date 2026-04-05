import streamlit as st
import uuid
import time
import os
import json
from api.client import get_chat_response

# JSON 데이터 로드
def load_device_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "data", "processed", "self-repair-list.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

device_data = load_device_data()

# 1. 페이지 설정 (세련된 테마 적용)
st.set_page_config(
    page_title="고객 서비스 챗봇",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. 세련된 UI를 위한 사용자 정의 CSS (Pretendard 폰트 및 모던 스타일)
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    * {
        font-family: 'Pretendard', sans-serif !important;
    }

    /* 제목 그라데이션 스타일 */
    .main-title {
        background: linear-gradient(135deg, #2563EB, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        text-align: center;
        color: #6B7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Streamlit 기본 헤더/푸터 숨김 */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* 버튼 스타일 조정 */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
</style>
""", unsafe_allow_html=True)

# 3. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요? 기기 관련 문의라면 왼쪽에서 기기명을 먼저 선택해주시면 더 정확한 답변을 드릴 수 있습니다. 😊"}]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 4. 사이드바 구성
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'>⚙️ 챗봇 설정</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; font-size: 0.9em; margin-bottom: 20px;'>개인 맞춤형 진단 지원</p>", unsafe_allow_html=True)
    
    st.markdown("#### 📱 기기 선택")
    st.caption("더 정확하고 빠른 문제 해결을 위해 현재 사용 중이신 기기 모델을 선택해주세요.")
    
    series_options = ["선택하지 않음"] + list(device_data.keys()) + ["기타"]
    selected_series = st.selectbox("1차: 기기 시리즈", options=series_options, index=0)
    
    selected_device = "선택하지 않음"
    if selected_series not in ["선택하지 않음", "기타"]:
        model_options = device_data[selected_series]
        selected_device = st.selectbox("2차: 상세 기기 모델", options=model_options)
    elif selected_series == "기타":
        selected_device = "기타"
    
    st.divider()
    
    st.markdown("#### 🔄 새로운 대화")
    if st.button("대화 초기화", use_container_width=True, type="primary"):
        st.session_state.messages = [{"role": "assistant", "content": "새로운 대화를 시작합니다. 기기 관련 문의가 있으신가요? 😊"}]
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
        
    st.divider()
    st.caption("Powered by Agentic RAG")

# 5. 메인 화면 구성
st.markdown("<h1 class='main-title'>스마트 고객센터 챗봇</h1>", unsafe_allow_html=True)

device_display = f"<span style='color: #2563EB; font-weight: 600;'>{selected_device}</span>" if selected_device != "선택하지 않음" else "선택되지 않음"
st.markdown(f"<p class='sub-title'>현재 설정된 기기: {device_display}</p>", unsafe_allow_html=True)

# 기존 메시지 렌더링
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. 사용자 입력 및 파이프라인 처리
if prompt := st.chat_input("증상이나 궁금한 점을 입력해주세요 (예: 스마트폰 발열이 너무 심해요)"):
    # 사용자 메시지 화면에 표시 (저장)
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})


    # AI 응답 생성 및 표시
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("답변을 분석하고 있습니다..."):
            # generation_pipeline.py 래퍼 함수 호출
            response = get_chat_response(
                question=prompt, 
                selected_device=selected_device, 
                thread_id=st.session_state.thread_id
            )
            
            # 매끄러운 텍스트 스트리밍 효과 적용
            full_response = ""
            # 단어 단위가 아닌 글자 단위나 약간의 청크로 스트리밍하면 더 자연스럽습니다.
            chunk_size = 3
            for i in range(0, len(response), chunk_size):
                full_response += response[i:i+chunk_size]
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)
                
            message_placeholder.markdown(full_response)
            
    # AI 메시지 상태 저장
    st.session_state.messages.append({"role": "assistant", "content": response})
