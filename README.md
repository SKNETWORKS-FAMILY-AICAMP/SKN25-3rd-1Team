# SKN25-3rd-1Team

uv venv --python 3.12.12


entrypoint에서 ingest 실행
python -m entrypoint.ingest


python -m entrypoint.query


<img width="867" height="432" alt="agentic_rag_architecture" src="https://github.com/user-attachments/assets/e60d14f6-f242-4e25-bed7-56dd0a6fb35a" />


## 📁 디렉토리 구조 (Directory Structure)

```text
SKN25-3RD-1TEAM/
├── .venv/                             # 파이썬 가상 환경 (Git 추적 제외)
├── data/                              # 데이터 및 DB 폴더 (Git 추적 제외)
│   ├── processed/                     # 전처리 완료된 데이터
│   ├── raw/                           # 원본 파일 (CSV, PDF 등)
│   └── vector_store/                  # Chroma DB 등 벡터 저장소
├── entrypoint/                        # 실행 진입점 스크립트 모음
│   ├── check_db.py                    # DB 적재 상태 확인용 스크립트
│   ├── ingest.py                      # 데이터 파싱 및 DB 적재 실행
│   ├── main.py                        # 메인 애플리케이션 실행 스크립트
│   └── query.py                       # RAG 파이프라인 질의 테스트 스크립트
├── frontend/                          # 사용자 인터페이스(UI) 코드
│   ├── api/                           # 백엔드/모델 연동 API 통신 모듈
│   ├── assets/                        # 이미지, 폰트 등 정적 리소스
│   ├── components/                    # UI 재사용 컴포넌트 모음
│   └── app.py                         # 프론트엔드 실행 파일
├── notebooks/                         # 데이터 수집 및 전처리 Notebook
├── src/                               # 핵심 로직 및 LangGraph 소스 코드
│   ├── pipelines/                     # 파이프라인 모듈
│   │   ├── embedding_pipeline.py      # 임베딩 처리 및 벡터화 파이프라인
│   │   ├── generation_pipeline.py     # 답변 생성 파이프라인
│   │   └── ingestion_pipeline.py      # 데이터 적재 파이프라인
│   ├── draw_graph.py                  # LangGraph 구조 시각화 스크립트
│   ├── graph.py                       # LangGraph 워크플로우 구성
│   ├── nodes.py                       # LangGraph의 각 노드 정의
│   └── state.py                       # GraphState  구조 정의
├── .env                               # 로컬 환경 변수 파일 (Git 추적 제외)
├── .env.example                       # 환경 변수 템플릿 파일
├── .gitignore                         # Git 추적 예외 처리 파일
├── architecture.png                   # RAG 아키텍처 다이어그램 이미지
├── Makefile                           # 빌드 및 실행 자동화 명령어 모음
├── README.md                          # 프로젝트 설명서
└── requirements.txt                   # 파이썬 의존성 패키지 목록