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

**기존 CS 시스템의 한계**

기존 FAQ 시스템은 정해진 키워드에만 반응하여 사용자의 다양한 구어체 표현이나 복잡한 맥락을 이해하기 어려웠습니다. 또한 자가수리 정보와 서비스센터 안내가 분리되어 있어, 문제 해결을 위해 사용자가 여러 채널을 직접 찾아야 하는 번거로움이 있었습니다.

**Smart CS의 차별점**

| | 내용 |
| --- | --- |
| **멀티 소스 데이터 활용** | FAQ 1,850개와 휴대폰 모델별 공식 자가수리 매뉴얼 PDF를 결합하여 답변의 전문성 확보 |
| **지능형 워크플로우** | 사용자의 의도(인사, 증상 문의, 센터 방문 등)를 실시간으로 분류하여 최적의 해결 경로로 유도 |
| **End-to-End 솔루션** | 증상 진단부터 자가수리 가이드 제공, 인근 서비스센터 위치 안내까지 하나의 인터페이스에서 완결 |


### 목표

**의도 분류 및 맥락 파악을 통한 사용자 경험 최적화**

- **Conditional Routing** : 사용자의 질문에 따라 소프트웨어/하드웨어 문제를 자동 판별하고, 일반 상담과 기술 지원 노드를 동적으로 분기
- **State Management** : 대화 상태(State)를 유지하여 멀티턴 대화에서도 기기 모델명과 이전 맥락을 놓치지 않게 함

**고도화된 문서 검색 파이프라인(Hybrid Search)을 통한 정확도 향상**

- FAQ 데이터의 제목 → ChromaDB (의미 기반 벡터 검색)
- FAQ 데이터의 본문 → BM25 (키워드 기반 검색)
- 의미상의 검색과 키워드 검색을 같이 활용하여 검색 정확도 향상


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
### 라우팅 함수

| 함수 | 역할 | 판단 로직 | 연결 노드 |
| --- | --- | --- | --- |
| **사용자 질문 의도 구분 함수** | 사용자의 첫 메시지를 분석하여 어떤 노드로 보낼지 결정 | 질문의 맥락이 단순 인사인지, 기술적 결함인지, 위치 정보 요청인지를 분석 | 일상 대화 → **일반 대화 응대 노드**<br>기술/수리 문의 → **휴대폰 수리 응대 노드**<br>위치/센터 문의 → **서비스 센터 안내 노드** |
| **문제 종류 구분 함수** | 휴대폰 수리 응대 노드에서 검색된 문서를 기반으로 문제 유형을 분류 | 휴대폰 수리 응대 노드에서 검색된 문서와 질문의 연관성을 대조 | SW 이슈 → **소프트웨어 문제 응대 노드**<br>HW 이슈 → **하드웨어 문제 응대 노드**<br>데이터 부족/검색 실패 → **사용자 의도 파악 노드** |
| **자가수리 관련 구분 함수** | 수리 난이도, 부품 보유 여부, 사용자의 자가 조치 의향을 최종 확인 | 지원 모델 리스트와 사용자의 수락 여부를 매칭 | 자가수리 적합 → **자가 수리 안내 노드**<br>수리 불가/방문 선호 → **서비스 센터 안내 노드** |

---
## Streamlit 주요 기능 및 UI/UX 구현 포인트

**고객의 질문을 먼저 읽는 '선제적 추천 UI'**
* **기획 의도 :** '액정 파손', '전원 안 켜짐' 등 자주 발생하는 치명적인 문제는 고객이 질문을 타이핑할 시간조차 아껴주어야 한다고 판단했습니다.
* **기술적 해결:** FAQ 데이터의 조회수(`viewCnt`)를 분석해 **'인기 질문 Top 6'를 메인 화면 상단에 칩(Chip) 형태로 배치**했습니다. 고객은 긴 타이핑 없이 단 한 번의 클릭만으로 즉시 AI 상담 파이프라인(FastAPI)과 연결되어 해결책을 제공받을 수 있습니다

**맥락이 끊기지 않는 매끄러운 상담 흐름 (SPA구현)**
* **문제 인식 :** 기존 고객센터의 가장 큰 불만인 페이지를 이동할 때마다 내 기기 정보나 이전 질문을 다시 입력해야 한다는 점을 해결 .
* **기술적 해결 :** `st.session_state`를 활용해 **단일 페이지 애플리케이션(SPA)**처럼 동작하게 구현했습니다. 고객이 사이드바에서 선택한 '기기 모델명'과 '이전 대화 기록'이 세션에 안전하게 유지되며, '자주 묻는 질문(FAQ)' 뷰와 'AI 상담' 뷰를 오가더라도 정보가 초기화되지 않고 유기적으로 연결됩니다.

**속도와 비용을 모두 잡은 스마트 하이브리드 라우팅**
* **문제 인식:** 모든 질문을 생성형 AI에 맡기면 응답 지연과 API 비용이 증가합니다.
* **기술적 해결:** 정형화된 FAQ 카테고리 검색을 지원하며, 필요에 따라 **1차적으로 정제된 매뉴얼(CSV)을 우선 검토하고, 추가 분석이 필요할 때만 RAG 기반 백엔드를 호출**하도록 스마트 라우팅 로직을 설계했습니다. 이를 통해 대기 시간을 최소화하고 인프라 비용을 절감했습니다.
---
## 성능 평가

총 4단계에 걸쳐 DeepEval과 RAGAS를 활용한 단계적 성능 평가를 진행했습니다.

| 단계 | 도구 | 내용 |
| --- | --- | --- |
| Step 1 | DeepEval | 디버깅용 샘플 10개 적용 · 문제 유형 파악 |
| Step 2 | DeepEval | 샘플 확대 (random 30개) · 전체 응답 흐름 및 성능 확인 |
| Step 3 | RAGAS | reference 데이터 추가 후 정답 기준 정량 평가 도입 |
| Step 4 | RAGAS | 전체 질문 50개로 확장 · 최종 성능 확인 |

### DeepEval 평가 결과

| 지표 | 점수 | 설명 |
| --- | --- | --- |
| Answer Relevancy | 0.864 | 답변이 질문 의도에 부합하는 정도 |
| Faithfulness | 0.883 | 답변이 검색 문서에 기반한 정도 (환각 방지) |

### RAGAS 평가 결과

| 지표 | 점수 | 설명 |
| --- | --- | --- |
| Faithfulness | 0.740 | 답변이 검색 문서에 충실한 정도 |
| Context Precision | 0.865 | 검색된 문서가 질문과 관련 있는 정도 |
| Context Recall | 0.857 | 필요한 문서를 빠짐없이 검색한 정도 |

전반적으로 0.8 이상의 높은 신뢰도를 보였으며, Faithfulness는 프롬프트 엔지니어링 고도화를 통해 추가 개선이 가능합니다.

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
