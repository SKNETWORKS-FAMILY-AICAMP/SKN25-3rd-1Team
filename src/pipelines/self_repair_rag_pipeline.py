"""
삼성 CS RAG 파이프라인 v4
- 세션 시작 시 모델 선택 UI (선택 사항)
- 질문 텍스트에서 모델명 3단계 자동 추출
- 모델 미특정 + 모델별 상이 가능성 높은 질문 → 되묻기
- REPEAT_HEADERS 청킹 버그 수정
- detect_header_level 동점 처리 수정
- 질문 번역기 (Query Translation) 추가
- 친숙한 기종명 검색 기능 추가 (S24, 플립5 등)

실행 전 설치:
pip install langchain langchain-community langchain-openai chromadb openai python-dotenv

실행:
  대화 모드:   python rag_pipeline_v4.py
  테스트 모드: python rag_pipeline_v4.py --test
"""

import os
import re
import sys
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── 설정 ────────────────────────────────────────────────────────────────────
MD_FOLDER   = "./md_files"
DB_DIR      = "./chroma_db"
COLLECTION  = "samsung_cs"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL   = "gpt-4o-mini"

REPEAT_HEADERS = {
    '분해 및 조립', '자가 진단',
    '소프트웨어 업데이트', '후면 커버 제거하기',
}
MIN_BODY = 50
MAX_SIZE = 1200
TARGET   = 1000

# 모델별로 답이 달라질 가능성이 높은 키워드
MODEL_SPECIFIC_KEYWORDS = [
    '제품 코드', '부품 코드', '코드', '테이프', '접착',
    'mm', '토크', '나사', '스크류', '용량', '규격',
    'GH', 'GH8',  # 삼성 부품 코드 prefix
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. 청킹
# ══════════════════════════════════════════════════════════════════════════════

def detect_header_level(text: str) -> str:
    """
    파일마다 다른 헤더 레벨 자동 감지
    S20 Ultra → '#'   SM-A256N → '##'
    동점 처리: h1 > 0 이면 '#' 우선
    """
    h1 = len(re.findall(r'^# [^#]', text, re.MULTILINE))
    h2 = len(re.findall(r'^## ',    text, re.MULTILINE))
    if h1 >= h2 and h1 > 0:
        return '# '
    return '## '


def split_by_steps(text: str, max_size: int) -> list:
    """번호 단계(1. 2. 3.) 경계를 보존하며 분할"""
    parts   = re.compile(r'\n(?=\d+[\.\s])').split(text)
    chunks, cur = [], ''
    for p in parts:
        if len(cur) + len(p) <= max_size:
            cur += p
        else:
            if cur.strip(): chunks.append(cur.strip())
            cur = p
    if cur.strip(): chunks.append(cur.strip())
    if not chunks:
        for p in text.split('\n\n'):
            if len(cur) + len(p) <= max_size:
                cur += '\n\n' + p
            else:
                if cur.strip(): chunks.append(cur.strip())
                cur = p
        if cur.strip(): chunks.append(cur.strip())
    return chunks or [text]


def chunk_md(text: str, model_name: str) -> list:
    """md 텍스트 → LangChain Document 리스트"""
    level    = detect_header_level(text)
    sep      = '\n' + level
    sections = text.split(sep)
    docs     = []
    last_independent_title = '기타'

    for sec in sections:
        lines = sec.strip().split('\n')
        title = lines[0].strip().lstrip('#').strip()
        body  = '\n'.join(lines[1:]).strip()

        body = re.sub(r'\(이미지:.+?\)', '', body)
        body = re.sub(r'\n{3,}', '\n\n', body).strip()

        if len(body) < MIN_BODY:        continue
        if re.match(r'^\d+\s', title): continue

        if title in REPEAT_HEADERS:
            full_title = f"{last_independent_title} > {title}"
            chunks = [body] if len(body) <= MAX_SIZE else split_by_steps(body, TARGET)
            for i, sub in enumerate(chunks):
                docs.append(Document(
                    page_content=f"## {full_title}\n\n{sub}",
                    metadata={"title": full_title, "chunk_index": i,
                              "model": model_name, "doc_type": "repair_manual",
                              "source": "삼성전자 자가수리 가이드"}
                ))
        else:
            last_independent_title = title
            chunks = [body] if len(body) <= MAX_SIZE else split_by_steps(body, TARGET)
            for i, sub in enumerate(chunks):
                docs.append(Document(
                    page_content=f"## {title}\n\n{sub}",
                    metadata={"title": title, "chunk_index": i,
                              "model": model_name, "doc_type": "repair_manual",
                              "source": "삼성전자 자가수리 가이드"}
                ))
    return docs


def extract_model_from_filename(filename: str) -> str:
    m = re.search(r'SM-[A-Z0-9]+', filename)
    if m: return m.group()
    m = re.search(r'[SG]\d{2}', filename)
    if m: return m.group()
    return os.path.splitext(filename)[0]


# ══════════════════════════════════════════════════════════════════════════════
# 2. 벡터DB
# ══════════════════════════════════════════════════════════════════════════════

def build_vectordb(md_folder=MD_FOLDER, db_dir=DB_DIR):
    embeddings  = OpenAIEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(collection_name=COLLECTION,
                         embedding_function=embeddings,
                         persist_directory=db_dir)

    md_files = [f for f in os.listdir(md_folder) if f.endswith('.md')]
    if not md_files:
        raise FileNotFoundError(f"'{md_folder}' 폴더에 .md 파일이 없습니다.")

    print(f"🚀 총 {len(md_files)}개 md 파일 처리 시작\n")

    for md_file in md_files:
        model_name = extract_model_from_filename(md_file)
        with open(os.path.join(md_folder, md_file), 'r', encoding='utf-8') as f:
            full_md = f.read()

        level = detect_header_level(full_md)
        docs  = chunk_md(full_md, model_name)

        
        if not docs:
            print(f"  ⚠️  {md_file}: 추출할 청크 없음 → 스킵합니다.")
            continue  

        try:
            vectorstore.delete(where={"model": model_name})
        except Exception:
            pass

        vectorstore.add_documents(docs)
        print(f"  ✅ {md_file}")
        print(f"     헤더: {repr(level.strip())}  |  청크: {len(docs)}개  |  모델: {model_name}")

    vectorstore.persist()
    print(f"\n✅ 완료: 총 {vectorstore._collection.count()}개 청크 저장")
    return vectorstore


def load_vectordb(db_dir=DB_DIR):
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return Chroma(collection_name=COLLECTION,
                  embedding_function=embeddings,
                  persist_directory=db_dir)


def get_available_models(vectorstore) -> list[str]:
    result = vectorstore._collection.get(include=["metadatas"])
    models = sorted(set(
        m["model"] for m in result["metadatas"] if "model" in m
    ))
    return models


# ══════════════════════════════════════════════════════════════════════════════
# 3. 모델명 추출 (3단계)
# ══════════════════════════════════════════════════════════════════════════════

def extract_model_from_query(query: str, available_models: list[str]) -> str | None:
    q_upper = query.upper().replace(" ", "")

    m = re.search(r'SM-[A-Z0-9]+', query, re.IGNORECASE)
    if m:
        found = m.group().upper()
        for model in available_models:
            if model.upper() == found:
                return model
        return found  

    for model in available_models:
        if model.upper() in q_upper:
            return model

    series_match = re.search(
        r'[SAZGF]\d{2,3}\s*(?:ULTRA|PLUS|FE|U\b|\+|FLIP|FOLD)?',
        q_upper
    )
    if series_match:
        series = series_match.group().replace(" ", "")
        for model in available_models:
            if series in model.upper():
                return model

    return None


def needs_model_clarification(query: str, model: str | None) -> bool:
    if model:
        return False
    return any(kw in query for kw in MODEL_SPECIFIC_KEYWORDS)


# ══════════════════════════════════════════════════════════════════════════════
# 4. RAG 체인
# ══════════════════════════════════════════════════════════════════════════════

PROMPT = ChatPromptTemplate.from_template("""
당신은 삼성 스마트폰 CS 전문 상담원입니다.

[답변 규칙]
1. 반드시 아래 [참고 문서] 내용만 근거로 답변하세요.
2. 제품 코드, 수치, 부품명처럼 모델마다 다를 수 있는 정보는
   반드시 해당 모델명을 함께 명시하세요.
3. 여러 모델의 정보가 문서에 포함된 경우 모델별로 구분하여 답변하세요.
4. 문서에 없는 내용은 "해당 내용은 매뉴얼에서 찾을 수 없습니다." 라고 답하세요.
5. 답변 마지막에 반드시 아래 형식으로 출처를 표시하세요:
   [출처: 섹션명 | 모델: XXX]

[참고 문서]
{context}

[고객 질문]
{question}

[답변]
""")


def make_rag_chain(vectorstore, available_models: list[str],
                   session_model: str | None = None, k: int = 8):

    query_translator = ChatPromptTemplate.from_template("""
    당신은 삼성 스마트폰 수리 검색 도우미입니다.
    고객의 질문을 파악하여, 하드웨어 수리 매뉴얼에서 검색하기 가장 좋은 '공식 전문 용어' 키워드로 변환하세요.
    
    [번역 규칙]
    1. 일상어는 매뉴얼 용어로 바꿉니다. (예: 액정/화면 → 디스플레이, 메인보드 → PBA, 뒷판 → 후면 커버)
    2. 수리/교체 후 세팅 관련 질문은 '교정(Calibration)' 이라는 단어를 반드시 포함하세요.
    3. 문장이 아닌, 검색에 최적화된 명사 형태의 키워드 3~4개로만 출력하세요.
    
    [고객 질문]: {question}
    [검색용 키워드]:
    """) | ChatOpenAI(model=LLM_MODEL, temperature=0) | StrOutputParser()

    def retrieve(query: str) -> list[Document]:
        model = session_model
        if model and model not in available_models:
            # 전달받은 세션 모델명이 DB의 실제 모델명(SM-...)과 다르면 변환
            model = find_model_by_nickname(model, available_models)
            
        if not model:
            model = extract_model_from_query(query, available_models)
            if model:
                print(f"  🔍 모델 자동 감지: {model}")
            else:
                print("  🔍 모델 미특정 → 전체 검색")

        optimized_query = query_translator.invoke({"question": query})
        print(f"  ✨ [검색어 최적화]: '{query}' ➡️ '{optimized_query}'")

        search_kwargs = {"k": k}
        if model:
            search_kwargs["filter"] = {"model": model}

        return vectorstore.as_retriever(search_kwargs=search_kwargs).invoke(optimized_query)

    def format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(
            f"[출처: {d.metadata.get('title','N/A')} | 모델: {d.metadata.get('model','N/A')}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        {
            "context": lambda q: format_docs(retrieve(q)),
            "question": RunnablePassthrough() 
        }
        | PROMPT
        | ChatOpenAI(model=LLM_MODEL, temperature=0)
        | StrOutputParser()
    )

    return chain, retrieve


# ══════════════════════════════════════════════════════════════════════════════
# 5. 테스트
# ══════════════════════════════════════════════════════════════════════════════

TEST_QA = [
    {"question": "배터리 교체 후 반드시 해야 하는 작업은 무엇인가요?",
     "ground_truth": "배터리 정보를 반드시 초기화해야 합니다."},
    {"question": "후면 커버 제거 시 발열팩을 몇 분간 올려두어야 하나요?",
     "ground_truth": "약 3분간 올려두세요."},
    {"question": "광학식 지문 센서 교정에 필요한 준비물은 무엇인가요?",
     "ground_truth": "흰색 교정 상자, 검은색 교정 상자, 3D 지문 모형이 필요합니다."},
    {"question": "오프닝 픽을 후면 커버에 넣을 때 최대 깊이는 얼마인가요?",
     "ground_truth": "3mm 이내로 넣어야 합니다."},
    {"question": "디스플레이 교체 후 교정이 필요한 항목을 모두 알려주세요.",
     "ground_truth": "광학식 지문 센서 교정, 거리 센서 교정, 터치 화면 패널 교정이 필요합니다."},
    {"question": "소프트웨어 업데이트 실패 시 복구 방법은?",
     "ground_truth": "Smart Switch의 소프트웨어 응급복구 및 초기화 기능을 사용하세요."},
    {"question": "SM-S908N 후면 커버용 접착 테이프 제품 코드는 무엇인가요?",
     "ground_truth": "GH81-18120A 입니다."},
    {"question": "발열팩 예열 권장 시간은 1000W 전자레인지 기준으로 몇 초인가요?",
     "ground_truth": "50초입니다."},
]

def run_test(vectorstore, k: int = 8):
    available_models = get_available_models(vectorstore)
    results = []

    print("\n" + "="*60)
    print("RAG 성능 테스트 (v4)")
    print(f"등록 모델: {available_models}")
    print("="*60)

    for i, qa in enumerate(TEST_QA):
        q, gt = qa["question"], qa["ground_truth"]

        chain, retriever = make_rag_chain(
            vectorstore, available_models, session_model=None, k=k
        )

        detected_model = extract_model_from_query(q, available_models)
        if needs_model_clarification(q, detected_model):
            print(f"\n[Q{i+1}] {q}")
            print(f"  → 모델 미특정 + 모델별 상이 가능 → 실서비스에서 되묻기 처리")
            continue

        answer   = chain.invoke(q)
        contexts = retriever(q)

        keywords = qa.get("keywords") or [
            w for w in gt.replace('.', '').split() if len(w) > 1
        ]
        score = sum(1 for w in keywords if w in answer) / len(keywords) if keywords else 0

        results.append({
            "question": q, "answer": answer,
            "ground_truth": gt, "score": score,
            "sections": [c.metadata.get('title') for c in contexts]
        })

        print(f"\n[Q{i+1}] {q}")
        print(f"  정답:      {gt}")
        print(f"  RAG답변:   {answer[:200].strip()}")
        print(f"  검색섹션:  {results[-1]['sections'][:3]}")
        print(f"  적중률:    {score:.0%}")

    if results:
        avg = sum(r["score"] for r in results) / len(results)
        print("\n" + "="*60)
        print(f"평균 키워드 적중률: {avg:.1%}")
        print("="*60)

    print("\n※ 정확한 평가는 RAGAS 사용 권장: pip install ragas")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6. 실행 
# ══════════════════════════════════════════════════════════════════════════════


MODEL_MAPPING = {
    # S 시리즈
    "S20": "SM-G981", "S20+": "SM-G986", "S20PLUS": "SM-G986", "S20플러스": "SM-G986", "S20ULTRA": "SM-G988", "S20울트라": "SM-G988", "S20U": "SM-G988",
    "S21": "SM-G991", "S21+": "SM-G996", "S21PLUS": "SM-G996", "S21플러스": "SM-G996", "S21ULTRA": "SM-G998", "S21울트라": "SM-G998", "S21U": "SM-G998",
    "S22": "SM-S901", "S22+": "SM-S906", "S22PLUS": "SM-S906", "S22플러스": "SM-S906", "S22ULTRA": "SM-S908", "S22울트라": "SM-S908", "S22U": "SM-S908",
    "S23": "SM-S911", "S23+": "SM-S916", "S23PLUS": "SM-S916", "S23플러스": "SM-S916", "S23ULTRA": "SM-S918", "S23울트라": "SM-S918", "S23U": "SM-S918",
    "S23FE": "SM-S711", "S23팬에디션": "SM-S711",
    "S24": "SM-S921", "S24+": "SM-S926", "S24PLUS": "SM-S926", "S24플러스": "SM-S926", "S24ULTRA": "SM-S928", "S24울트라": "SM-S928", "S24U": "SM-S928",
    "S25": "SM-S931", "S25+": "SM-S936", "S25PLUS": "SM-S936", "S25플러스": "SM-S936", "S25ULTRA": "SM-S938", "S25울트라": "SM-S938", "S25U": "SM-S938",
    
    # Z 시리즈
    "플립5": "SM-F731", "ZFLIP5": "SM-F731", "Z플립5": "SM-F731",
    "플립6": "SM-F741", "ZFLIP6": "SM-F741", "Z플립6": "SM-F741",
    "플립7": "SM-F761", "ZFLIP7": "SM-F761", "Z플립7": "SM-F761",
    "플립7FE": "SM-F766", "ZFLIP7FE": "SM-F766", "Z플립7FE": "SM-F766",
    "폴드5": "SM-F946", "ZFOLD5": "SM-F946", "Z폴드5": "SM-F946",
    "폴드6": "SM-F956", "ZFOLD6": "SM-F956", "Z폴드6": "SM-F956",
    "폴드7": "SM-F966", "ZFOLD7": "SM-F966", "Z폴드7": "SM-F966",
    
    # A 시리즈
    "A25": "SM-A256", "A255G": "SM-A256", 
    "A35": "SM-A356", "A355G": "SM-A356"
}

def find_model_by_nickname(user_input: str, available_models: list[str]) -> str | None:
    clean_input = user_input.upper().replace(" ", "").replace("갤럭시", "").replace("GALAXY", "")
    
    for nickname in sorted(MODEL_MAPPING.keys(), key=len, reverse=True):
        if nickname in clean_input:
            prefix = MODEL_MAPPING[nickname] 
            
            for db_model in available_models:
                if db_model.startswith(prefix):
                    return db_model
                    
    for db_model in available_models:
        if clean_input in db_model.upper():
            return db_model
            
    return None

def select_model_interactive(available_models: list[str]) -> str | None:
    if not available_models:
        print("⚠️  등록된 모델이 없습니다.")
        return None

    while True:
        print("\n📱 수리하실 스마트폰 기종을 입력해 주세요.")
        print("  (예시: S24, 갤럭시 s23 울트라, 플립5 등)")
        user_input = input("기종 입력 (Enter = 질문마다 자동 감지): ").strip()

        if user_input == "":
            print("  → 🔍 기종을 지정하지 않았습니다. 질문마다 자동 감지합니다.\n")
            return None

        selected_model = find_model_by_nickname(user_input, available_models)

        if selected_model:
            print(f"\n  ✅ [{selected_model}] 모델로 고정하여 맞춤 상담을 시작합니다!\n")
            return selected_model
        else:
            print(f"\n  ⚠️ '{user_input}'(은)는 등록되지 않았거나 인식할 수 없는 기종입니다.")
            print("  다시 한 번 정확히 입력해 주세요. (예: S22 플러스)")

CLARIFICATION_MSG = (
    "정확한 답변을 위해 모델명을 알려주세요.\n"
    "  예) S24, 갤럭시 S22 울트라, 플립5\n"
    "  (또는 /모델 입력으로 고정할 기종을 다시 선택할 수 있습니다.)"
)

if __name__ == "__main__":

    if not os.path.exists(MD_FOLDER):
        os.makedirs(MD_FOLDER)
        import shutil
        for f in os.listdir('.'):
            if f.endswith('.md'):
                shutil.copy(f, os.path.join(MD_FOLDER, f))
                print(f"  복사: {f} → {MD_FOLDER}/")

    if os.path.exists(DB_DIR):
        print(f"기존 벡터DB 로드: {DB_DIR}")
        vectorstore = load_vectordb()
        print(f"  로드된 청크 수: {vectorstore._collection.count()}\n")
    else:
        vectorstore = build_vectordb()

    available_models = get_available_models(vectorstore)

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test(vectorstore, k=8)
        sys.exit(0)

    # ── 대화 모드 ──────────────────────────────────────────────────────────
    session_model = select_model_interactive(available_models)
    chain, _ = make_rag_chain(vectorstore, available_models,
                               session_model=session_model, k=8)

    print("💬 삼성 CS 챗봇 시작 (종료: q | 모델 변경: /모델)")
    print("  자동 테스트: python rag_pipeline_v4.py --test\n")

    while True:
        q = input("질문: ").strip()
        if not q: continue
        if q.lower() in ('q', 'quit', 'exit'): break

        if q == "/모델":
            session_model = select_model_interactive(available_models)
            chain, _ = make_rag_chain(vectorstore, available_models,
                                       session_model=session_model, k=8)
            continue

        if not session_model:
            detected = extract_model_from_query(q, available_models)
            if needs_model_clarification(q, detected):
                print(f"\n답변: {CLARIFICATION_MSG}\n")
                continue

        print(f"\n답변: {chain.invoke(q)}\n")