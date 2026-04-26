#!/bin/bash

echo "Starting Agentic Mobile CS API, Streamlit UI, Redis, and Celery..."

# 0. Start Redis
redis-server --daemonize yes
sleep 2

# Check Redis connection
if redis-cli ping > /dev/null 2>&1; then
    echo "[SUCCESS] Redis is up and running."
else
    echo "[ERROR] Redis failed to start or is not reachable."
    exit 1
fi

# 환경 변수 설정
export PYTHONPATH=$PYTHONPATH:.

# 1. Start FastAPI Backend
python3 -m uvicorn entrypoint.main:app --host 0.0.0.0 --port 8000 &

# 2. Start Streamlit UI
python3 -m streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &

# 3. Start Celery Worker & Beat
python3 -m celery -A src.utils.tasks worker --beat --loglevel=info &

echo "All services are started."

# 모든 백그라운드 프로세스가 유지되도록 대기
wait