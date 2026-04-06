# 🤖 [프로젝트 제목] LangGraph 기반 csai 챗봇 

**팀명 : SKN25-3rd-1Team**

## 👥 팀원

| 김나연 | 김지현 | 박범수 | 이하윤 | 여해준 |
| --- | --- | --- | --- | --- |
| 역할 | 역할 | 역할 | 역할 | 역할 |

---

## 1. 프로젝트 소개 및 목표

### 1.1 프로젝트 소개
모바일 CS RAG 시스템

### 1.2 목표
- 환각을 방지하고 원하는 내외부 데이터 내에서 RAG 기반 LLM 활용 질의 응답 시스템 구현
- 문서를 벡터 형태로 임베딩하여 벡터DB에 저장 및 검색
- LangChain을 활용하여 벡터DB와 LLM 연동

---

## 2. 프로젝트 디렉토리 구조

```
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
│   └── state.py                       # GraphState 구조 정의
├── .env                               # 로컬 환경 변수 파일 (Git 추적 제외)
├── .env.example                       # 환경 변수 템플릿 파일
├── .gitignore                         # Git 추적 예외 처리 파일
├── architecture.png                   # RAG 아키텍처 다이어그램 이미지
├── Makefile                           # 빌드 및 실행 자동화 명령어 모음
├── README.md                          # 프로젝트 설명서
└── requirements.txt                   # 파이썬 의존성 패키지 목록
```

---

## 3. 시스템 아키텍처

![architecture](architecture.png)

---

## 4. 데이터 파이프라인과 모듈별 상세 설명

### 4.1 ingestion_pipeline.py

**역할**
 
FAQ 데이터(CSV/Excel)와 자가수리 매뉴얼(MD) 파일을 읽어 벡터 DB(ChromaDB)와 BM25 인덱스에 적재하는 데이터 파이프라인입니다.
 
**주요 기능**
 
| 기능 | 설명 |
| --- | --- |
| **FAQ 적재** | CSV/Excel 파일에서 FAQ 데이터를 읽어 Title 기반으로 ChromaDB에 임베딩 적재 |
| **BM25 인덱스 구축** | cleaned_content 기반 키워드 검색용 BM25 코퍼스를 `.pkl` 파일로 저장 |
| **자가수리 적재** | MD 형식의 자가수리 매뉴얼을 청킹하여 ChromaDB에 적재 |
 
**처리 흐름**
 
```
CSV/Excel 파일 로드
    ↓
Document 객체 생성
    ├── ChromaDB용 (Title 임베딩)
    └── BM25용 (cleaned_content 키워드)
    ↓
ChromaDB 배치 적재 (batch_size=150)
    ↓
BM25 코퍼스 pkl 파일 저장
```

### 4.2 embedding_pipeline.py

**역할**

ChromaDB 벡터 저장소를 생성하거나 불러오는 모듈입니다. OpenAI 임베딩 모델을 사용해 텍스트를 벡터로 변환하고, 컬렉션 이름 기반으로 독립적인 저장소를 관리합니다.

**주요 기능**

| 기능 | 설명 |
| --- | --- |
| **벡터 저장소 로드** | 컬렉션 이름(`faq`, `self-repair`)으로 ChromaDB 저장소 생성 또는 로드 |
| **임베딩 모델** | OpenAI `text-embedding-3-small` 모델 사용 |
| **저장 경로 관리** | `.env`의 `CHROMA_PERSIST_DIR` 경로 기반으로 물리적 저장소 관리 |

### 4.3 generation_pipeline.py

**역할**

LangGraph RAG 파이프라인을 실행하고 최종 답변을 생성하는 모듈입니다. MongoDB와 연동하여 대화 로그를 저장하고 관리합니다.

**주요 기능**

| 기능 | 설명 |
| --- | --- |
| **답변 생성** | LangGraph `rag_app`을 통해 사용자 질문에 대한 최종 답변 생성 |
| **대화 로그 저장** | MongoDB에 질문, 답변, 기기 정보, 신뢰도 점수 등 대화 기록 적재 |
| **멀티턴 지원** | `thread_id` 기반으로 대화 히스토리 유지 |
| **오류 처리** | 파이프라인 오류 발생 시 에러 로그 저장 후 안전하게 반환 |

**처리 흐름**
```
사용자 질문 수신
    ↓
MongoDB 로그 초기화 (status: pending)
    ↓
LangGraph rag_app 실행
    ↓
최종 답변 추출
    ↓
MongoDB 로그 업데이트 (status: success / error)
    ↓
결과 반환
```

### 4.4 nodes.py

**역할**

LangGraph의 각 노드와 라우팅 함수를 정의하는 핵심 모듈입니다. 사용자 질문을 분류하고 FAQ 검색, 답변 생성, 자가수리 안내, 서비스센터 안내 등을 처리합니다.

**노드 구성**

| 노드명 | 역할 |
| --- | --- |
| `chat_node` | 인사/잡담 응대 및 대화 요약 처리 (Few-shot 프롬프트 적용) |
| `retrieve_node` | LLM 쿼리 변환 + ChromaDB 벡터 검색 + BM25 하이브리드 검색 |
| `generate_node` | 검색된 FAQ 문서 기반 단계별 답변 생성 (CoT 프롬프트 적용) |
| `self_repair_classifier_node` | 기기 모델명, 하드웨어 여부, 자가수리 의향 동시 판별 |
| `self_repair_guide_node` | 자가수리 매뉴얼 RAG 검색 및 가이드 제공 |
| `nearest_center_node` | 카카오맵 API 기반 주변 서비스센터 안내 |
| `fallback_node` | FAQ 검색 실패 시 선택지 제공 |

**라우팅 구성**

| 라우팅 함수 | 역할 |
| --- | --- |
| `route_question` | 진입점 라우터 - 인사/기기문의/센터방문 분류 |
| `route_issue_type` | SW/HW/센터방문 분류 후 다음 노드 결정 |
| `route_after_self_repair_check` | 자가수리 가능 여부에 따라 가이드 또는 센터 안내 |

### 4.5 graph.py

**역할**

LangGraph를 사용하여 전체 CS RAG 워크플로우를 구성하고 컴파일하는 모듈입니다. 노드와 라우팅 함수를 연결하여 대화 흐름을 정의합니다.

**그래프 구조**

| 시작 | 라우팅 | 종료 |
| --- | --- | --- |
| START | → chat_node | → END |
| START | → nearest_center_node | → END |
| START | → retrieve_node → generate_node | → END |
| START | → retrieve_node → fallback_node | → END |
| START | → retrieve_node → nearest_center_node | → END |
| START | → retrieve_node → self_repair_classifier_node → self_repair_guide_node | → END |
| START | → retrieve_node → self_repair_classifier_node → nearest_center_node | → END |

**주요 특징**

| 항목 | 설명 |
| --- | --- |
| **MemorySaver** | 멀티턴 대화 히스토리를 메모리에 유지 |
| **싱글톤 인스턴스** | `rag_app` 으로 앱 전역에서 단일 인스턴스 공유 |
| **조건부 라우팅** | `route_question`, `route_issue_type`, `route_after_self_repair_check` 로 동적 흐름 제어 |

---

## 5. 시연 결과

### [시연 케이스 제목]

![시연결과](이미지경로)


---

## 6. 기술 스택

### Backend

| 구분 | 기술 | 설명 |
| --- | --- | --- |
| | | |

### Frontend

| 구분 | 기술 | 설명 |
| --- | --- | --- |
| | | |

### Data & Storage

| 구분 | 기술 | 설명 |
| --- | --- | --- |
| | | |

---

## 7. 환경 구축 및 실행 방법

### 환경 구축
```bash
uv venv --python 3.12.12
```

### 데이터 적재
```bash
python -m entrypoint.ingest
```

### 실행
```bash
python -m entrypoint.query
```

---

## 8. 향후 개발 계획

---

## 💬 한 줄 회고

> #### 이름
>

---

> #### 이름
>

---

> #### 이름
>

---

> #### 이름
>

---

> #### 이름
>
