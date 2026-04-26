import streamlit as st
from pymongo import MongoClient
from collections import Counter
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta

load_dotenv()

client = MongoClient(os.getenv('MONGO_URI'))
db = client['chatbot_db']

st.set_page_config(page_title="Smart CS 대시보드", layout="wide")
st.title("📊 Smart CS 인기 질문 대시보드")

# 기간 필터
period = st.selectbox("기간 선택", ["전체", "이번 주", "이번 달"])
now = datetime.now()

docs = list(db['usage_logs'].find({"status": "success", "question": {"$ne": ""}}))

if period == "이번 주":
    docs = [d for d in docs if datetime.fromisoformat(d['timestamp']) >= now - timedelta(days=7)]
elif period == "이번 달":
    docs = [d for d in docs if datetime.fromisoformat(d['timestamp']) >= now - timedelta(days=30)]

# ==========================================
# 숫자 카드
# ==========================================
total = len(docs)
hw = sum(1 for d in docs if d.get('is_hardware_issue'))
sw = total - hw
hw_ratio = round(hw / total * 100, 1) if total > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("총 문의 수", f"{total}건")
c2.metric("하드웨어 문의", f"{hw}건")
c3.metric("소프트웨어 문의", f"{sw}건")
c4.metric("하드웨어 비율", f"{hw_ratio}%")

st.divider()

# ==========================================
# 인기 질문 + 기기별 통계
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 인기 질문 Top 10")
    greetings = ["안녕", "안녕하세요", "고마워", "감사합니다", "ㅎㅇ", "hi", "hello"]
    questions = [d['question'] for d in docs if d['question'] not in greetings]
    question_counts = Counter(questions).most_common(10)
    if question_counts:
        df_q = pd.DataFrame(question_counts, columns=["질문", "횟수"])
        st.dataframe(df_q, use_container_width=True, hide_index=True)
    else:
        st.write("데이터 없음")

with col2:
    st.subheader("📱 기기별 문의 통계")
    devices = [d['selected_device'] for d in docs if d.get('selected_device') and d['selected_device'] != "선택하지 않음"]
    device_counts = Counter(devices).most_common(10)
    if device_counts:
        df_d = pd.DataFrame(device_counts, columns=["기기", "횟수"])
        # x축 라벨 겹침 방지 - plotly 사용
        import plotly.express as px
        fig = px.bar(df_d, x="기기", y="횟수", text="횟수")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("데이터 없음")

st.divider()

# ==========================================
# 노드별 응답 속도
# ==========================================
st.subheader("⚡ 노드별 평균 응답 속도 (초)")

perf_docs = list(db['node_perf_logs'].find())

if perf_docs:
    df_perf = pd.DataFrame(perf_docs)
    if 'node_name' in df_perf.columns and 'duration' in df_perf.columns:
        avg_perf = df_perf.groupby('node_name')['duration'].mean().reset_index()
        avg_perf.columns = ["노드", "평균 응답 속도(초)"]
        avg_perf["평균 응답 속도(초)"] = avg_perf["평균 응답 속도(초)"].round(3)
        avg_perf = avg_perf.sort_values("평균 응답 속도(초)", ascending=False)

        import plotly.express as px
        fig2 = px.bar(avg_perf, x="노드", y="평균 응답 속도(초)", text="평균 응답 속도(초)")
        fig2.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("node_name 또는 duration 필드 없음")
else:
    st.write("데이터 없음")