#!/bin/bash

echo "Starting Agentic Mobile CS API, Streamlit UI, Redis, and Celery..."

# 0. Start Redis (필수 기반 서비스)
echo "[INFO] Attempting to start Redis server..."
redis-server &
sleep 2 # Redis가 완전히 뜰 때까지 잠시 대기

# 1. Start FastAPI Backend
echo "[INFO] FastAPI Server starting..."
export PYTHONPATH=$PYTHONPATH:.
.venv/bin/python -m uvicorn entrypoint.main:app --host 0.0.0.0 --port 8000 --reload &

# 2. Start Streamlit UI
echo "[INFO] Streamlit UI starting..."
.venv/bin/python -m streamlit run frontend/app.py --server.headless true &

# 3. Start Celery Worker & Beat
# tasks.py가 src/utils에 있으므로 모듈 경로로 지정합니다.
echo "[INFO] Celery Worker & Beat starting..."
export PYTHONPATH=$PYTHONPATH:.
.venv/bin/python -m celery -A src.utils.tasks worker --beat --loglevel=info &

echo "All services are started in the background."

# 모든 백그라운드 프로세스가 유지되도록 대기
wait