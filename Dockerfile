# 1. 베이스 이미지 설정
FROM python:3.12.12-slim

# 2. 시스템 의존성 및 Redis 설치
RUN apt-get update && apt-get install -y \
    redis-server \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 전체 소스 코드 복사
COPY . .

# 6. 실행 스크립트 권한 부여
RUN chmod +x run_scripts.sh

# 7. 포트 노출 (FastAPI: 8000, Streamlit: 8501, Redis: 6379)
EXPOSE 8000 8501 6379

# 8. 실행 명령
CMD ["./run_scripts.sh"]