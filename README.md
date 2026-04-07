# 🤖 [프로젝트 제목] LangGraph 기반 csai 챗봇 

팀명 : **알려조**

## 👥 팀원

| 김나연 | 김지현 | 박범수 | 이하윤 | 여해준 |
| --- | --- | --- | --- | --- |
| 역할 | 역할 | 역할 | 역할 | 역할 |

---

## 프로젝트 소개 및 목표

### 프로젝트 소개

**서비스 명:** Smart CS (스마트 고객센터)

삼성 갤럭시 스마트폰 사용자를 위한 AI 기반 고객 서비스 챗봇입니다.

기존 FAQ형 상담은 정해진 키워드에만 반응하여 사용자의 다양한 표현을 이해하지 못하고,   
획일적인 답변만 제공하는 한계가 있었습니다.   
Smart CS는 이러한 한계를 극복하고 사용자 맞춤형 해결책을 제시하기 위해 기획되었습니다.

삼성전자 서비스 FAQ 1,850개와 기기별 자가수리 매뉴얼 PDF를 데이터 소스로 활용하여,  
사용자의 기기 증상 문의에 단계별 해결 방법을 제공합니다.

사용자가 직접 수리 의향을 밝히면 기기 모델에 맞는 자가수리 매뉴얼을 안내하고,  
서비스센터 방문을 원하거나 자가수리가 어렵다고 판단될 경우 카카오맵 API를 통해  
사용자 위치 기반 가까운 삼성전자 서비스센터를 실시간으로 안내하는 완성형 CS 솔루션입니다.

### 목표

* 삼성 갤럭시 FAQ 및 자가수리 매뉴얼 기반 RAG 파이프라인 구현으로 환각 방지 및 출처 기반 정확한 답변 제공
* FAQ 데이터와 자가수리 PDF를 벡터 임베딩하여 ChromaDB에 저장하고 BM25 하이브리드 검색으로 검색 품질 향상
* LangGraph를 활용한 조건부 라우팅으로 일반 문의 / 자가수리 안내 / 서비스센터 안내를 자동 분기하는 Agentic RAG 시스템 구현


---

## 주요 기능

### 🔀 지능형 의도 파악 (Intent Routing)
사용자의 메시지를 GPT-4 Turbo가 실시간으로 분석하여 **인사/잡담**, **기기 증상 문의**, **서비스센터 찾기** 3가지 의도로 자동 분류합니다.  
멀티턴 대화 맥락을 고려하여 이전 대화 흐름에 맞는 응답을 제공합니다.

### ⚡ Agentic RAG 파이프라인
단순 문서 검색을 넘어 LangGraph 기반의 조건부 라우팅으로 **소프트웨어 문제 → FAQ 답변**, **하드웨어 문제 → 자가수리 가이드**, **검색 실패 → Fallback 안내**로 이어지는 동적 워크플로우를 수행합니다.

### 🔍 하이브리드 검색
ChromaDB 벡터 검색(의미 기반)과 BM25(키워드 기반)를 결합한 하이브리드 검색을 적용합니다.   
LLM 쿼리 변환으로 일상적인 표현을 FAQ 검색에 최적화된 키워드로 변환하여 검색 정확도를 높입니다.

### 🛠️ 자가수리 가이드
사용자가 직접 수리 의향을 밝히면 선택한 기기 모델을 자동 감지하여 삼성 공식 수리 매뉴얼 기반의 단계별 분해·교체 절차를 안내합니다.  
자가수리 지원 모델 여부를 자동으로 판별합니다.

### 📍 위치 기반 서비스센터 안내
카카오맵 API와 연동하여 사용자 현재 위치 기반 반경 5km 이내 가까운 삼성전자 서비스센터 3곳의 이름, 주소, 거리를 실시간으로 안내합니다.

---

## 프로젝트 디렉토리 구조

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

## 시스템 아키텍처

### 전체 파이프라인

Data Ingestion → Embedding → Vector Store (ChromaDB) → LangGraph Node Logic

### LangGraph 워크플로우

* **GraphState** 기반 상태 구조 정의로 노드 간 데이터 공유
* **조건부 라우팅 (Conditional Edges)** 으로 Chat / Retrieve / Repair 동적 분기
* **자가수리 분류기 (Self-Repair Classifier)** 로 기기 모델, 하드웨어 여부, 수리 의향 동시 판별
* **Fail-over 에스컬레이션** 으로 FAQ → 자가수리 가이드 → 서비스센터 안내 단계적 처리

### 시스템 아키텍처 다이어그램

![architecture](MDimages/architecture.png)

### 기술 스택

| 구분 | 기술 |
| --- | --- |
| LLM & Framework | LangChain / LangGraph / GPT-4 Turbo |
| Vector DB | ChromaDB / BM25 (rank-bm25) |
| Backend | FastAPI + Uvicorn / Pydantic |
| Frontend | Streamlit |
| Database | MongoDB |
| Language | Python 3.12 / Pandas / PyPDF |
| External API | OpenAI API / 카카오맵 API |

---

## 데이터 파이프라인과 모듈별 상세 설명

### ingestion_pipeline.py

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
### self_repair_rag_pipeline.py

**역할**

MD 파일을 수리 가이드 특성에 맞게 청킹하여 ChromaDB에 저장하고,  
Parent Document Retrieval(PDR)로 정밀 검색을 수행하는 핵심 RAG 모듈입니다.  
`nodes.py`의 `retrieve_node`에서 활용됩니다.

**주요 기능**

| 기능 | 설명 |
| --- | --- |
| **헤더 자동 감지** | 모델마다 다른 헤더 레벨(`#` / `##`) 자동 감지 후 통일 처리 |
| **노이즈 필터링** | 이미지 캡션 잔재, 페이지번호, 차례 항목 등 자동 제거 |
| **PDR 청킹** | Child(300자) / Parent(전체 섹션) 분리 저장으로 검색 정확도와 컨텍스트 풍부함 동시 달성 |
| **모델명 자동 감지** | 구어체 모델명 → 공식 모델명 변환 (예: "S24 울트라" → `SM-S928N`) |
| **쿼리 최적화** | 구어체 질문을 매뉴얼 전문 용어로 변환하여 검색 정확도 향상 |

**처리 흐름**

```
MD 파일 로드 (mds/md_files/)
    ↓
헤더 레벨 자동 감지 + 노이즈 필터링
    ↓
Parent / Child 분리 청킹
    ├── Child (300자, 검색용)  → samsung_cs_child 컬렉션
    └── Parent (전체 섹션, LLM 전달용) → samsung_cs_parent 컬렉션
    ↓
retrieve_node 호출 시
    Child로 정밀 검색
        ↓
    히트된 Child의 parent_id 추출
        ↓
    Parent(전체 섹션) → generate_node로 전달
```

**Parent Document Retrieval 설계 의도**

기존 RAG의 딜레마를 해결하기 위해 도입

| 방식 | 문제점 |
| --- | --- |
| 청크 크게 | 여러 내용이 섞여 벡터 희석 → 검색 부정확 |
| 청크 작게 | LLM에 전달되는 컨텍스트 부족 → 답변 품질 저하 |
| **PDR (채택)** | 작은 청크로 정밀 검색 + 전체 섹션으로 풍부한 컨텍스트 전달 |
### embedding_pipeline.py

**역할**

ChromaDB 벡터 저장소를 생성하거나 불러오는 모듈입니다.  
OpenAI 임베딩 모델을 사용해 텍스트를 벡터로 변환하고, 컬렉션 이름 기반으로 독립적인 저장소를 관리합니다.

**주요 기능**

| 기능 | 설명 |
| --- | --- |
| **벡터 저장소 로드** | 컬렉션 이름(`faq`, `self-repair`)으로 ChromaDB 저장소 생성 또는 로드 |
| **임베딩 모델** | OpenAI `text-embedding-3-small` 모델 사용 |
| **저장 경로 관리** | `.env`의 `CHROMA_PERSIST_DIR` 경로 기반으로 물리적 저장소 관리 |

### generation_pipeline.py

**역할**

LangGraph RAG 파이프라인을 실행하고 최종 답변을 생성하는 모듈입니다.  
MongoDB와 연동하여 대화 로그를 저장하고 관리합니다.

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

### nodes.py

**역할**

LangGraph의 각 노드와 라우팅 함수를 정의하는 핵심 모듈입니다.  
사용자 질문을 분류하고 FAQ 검색, 답변 생성, 자가수리 안내, 서비스센터 안내 등을 처리합니다.

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

### graph.py

**역할**

LangGraph를 사용하여 전체 CS RAG 워크플로우를 구성하고 컴파일하는 모듈입니다.  
노드와 라우팅 함수를 연결하여 대화 흐름을 정의합니다.

**그래프 구조**

<img width="967" height="432" alt="agentic_rag_architecture" src="https://github.com/user-attachments/assets/d020dedd-4b1b-466a-80c9-fd133382239c" />


**주요 특징**

| 항목 | 설명 |
| --- | --- |
| **MemorySaver** | 멀티턴 대화 히스토리를 메모리에 유지 |
| **싱글톤 인스턴스** | `rag_app` 으로 앱 전역에서 단일 인스턴스 공유 |
| **조건부 라우팅** | `route_question`, `route_issue_type`, `route_after_self_repair_check` 로 동적 흐름 제어 |

---

## 시연 결과

### [시연 케이스 제목]

![시연결과](이미지경로)


---

## 기술 스택

### Backend

| 구분 | 기술 | 설명 |
| --- | --- | --- |
| 언어 | ![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=flat-square&logo=python&logoColor=white) | 전체 파이프라인 구현 핵심 언어 |
| LLM | ![GPT-4](https://img.shields.io/badge/GPT--4_Turbo-412991?style=flat-square&logo=openai&logoColor=white) | 답변 생성, 의도 분류, 쿼리 변환 |
| RAG Framework | ![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat-square&logo=chainlink&logoColor=white) ![LangGraph](https://img.shields.io/badge/LangGraph-121212?style=flat-square&logo=chainlink&logoColor=white) | 하이브리드 검색 + 조건부 라우팅 RAG 파이프라인 |
| 임베딩 | ![OpenAI](https://img.shields.io/badge/text--embedding--3--small-412991?style=flat-square&logo=openai&logoColor=white) | 텍스트 벡터 임베딩 |
| 웹 프레임워크 | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![Uvicorn](https://img.shields.io/badge/Uvicorn-499848?style=flat-square&logo=gunicorn&logoColor=white) | 고성능 비동기 API 서버 |
| 데이터 검증 | ![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white) | 구조화된 출력 및 데이터 검증 |
| 비동기 작업 | ![Celery](https://img.shields.io/badge/Celery-37814A?style=flat-square&logo=celery&logoColor=white) ![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white) | 스케줄러 및 비동기 태스크 처리 |

### Frontend

| 구분 | 기술 | 설명 |
| --- | --- | --- |
| UI | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) | 챗봇 인터페이스 |

### Data & Storage

| 구분 | 기술 | 설명 |
| --- | --- | --- |
| 벡터 DB | ![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=flat-square) | FAQ 및 자가수리 임베딩 저장·검색 |
| 키워드 검색 | ![BM25](https://img.shields.io/badge/BM25-4285F4?style=flat-square) | 키워드 기반 하이브리드 검색 |
| 로그 DB | ![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white) | 대화 로그 및 사용 기록 저장 |
| 캐시 | ![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white) | Celery 브로커 및 캐시 |
| 데이터 처리 | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![OpenPyXL](https://img.shields.io/badge/OpenPyXL-217346?style=flat-square&logo=microsoftexcel&logoColor=white) | FAQ 데이터 전처리 및 Excel 파싱 |
| PDF 파싱 | ![PyPDF](https://img.shields.io/badge/PyPDF-EC1C24?style=flat-square) | 자가수리 매뉴얼 PDF 파싱 |

---

## 환경 구축 및 실행 방법

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

## 향후 개발 계획

---

## 💬 한 줄 회고

> #### 김나연
>

---

> #### 김지현
>

---

> #### 박범수
>

---

> #### 이하윤
>

---

> #### 여해준
> LLM을 활용한 챗봇 개발을 진행하면서 프롬프트를 얼마나 잘짜냐에 따라서 챗봇이 바보가 되냐 아니냐가 정해지는지를 깨달았습니다.
