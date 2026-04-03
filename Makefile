.PHONY: ingest api ui all

# 데이터 적재
ingest:
	PYTHONPATH=. python entrypoint/ingest.py

# FastAPI 백엔드 실행
api:
	uvicorn entrypoint.main:app --host 0.0.0.0 --port 8000 --reload

# Streamlit 프론트엔드 실행
ui:
	PYTHONPATH=. streamlit run frontend/app.py

# 전체 시스템 구동 
all: api ui