import os
import pandas as pd
import pickle
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.pipelines.embedding_pipeline import get_vector_store

#======================================================
#=========faq 적재 시작
#======================================================

def ingest_faq_data(file_path: str):
    """FAQ 데이터를 Chroma(Title)와 BM25용으로 분리하여 적재하는 파이프라인"""
    print(f"[{file_path}] 데이터 로드 시작...")
    
    file_name = os.path.basename(file_path)
    
    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            print("(.csv, .xls, .xlsx 만 지원)")
            return
            
        df = df.fillna("") 
    except Exception as e:
        print(f"❌ 파일 로드 중 오류 발생: {e}")
        return

    chroma_documents = [] # title 비교를 위한 벡터 db 구축
    bm25_documents = [] #cleaned_content와 키워드 비교를 위한 인덱스 구축
    
    # 1. Document 객체 생성 (Chroma용 / BM25용 분리)
    for _, row in df.iterrows():
        title = str(row.get('title', '')).strip()
        cleaned_content = str(row.get('cleaned_content', '')).strip()
        doc_id = str(row.get('id', ''))
        

        # 공통 메타데이터 ('id'를 기준으로 동일 문서인지 판별해야 함)
        metadata = {
            "source_file": file_name,
            "id": doc_id,
            "contentType": str(row.get('contentType', '')),
            "exposureDate": str(row.get('exposureDate', '')),
            "symptom_category": str(row.get('symptom_category', '')),
            "symptom_code": str(row.get('symptom_code', '')),
            "url": str(row.get('url', '')),
            "img_count": str(row.get('img_count', '0')),
            # 교차 참조를 위해 서로의 데이터를 메타데이터에 보관
            "title": title, 
            "cleaned_content": cleaned_content 
        }
        
        # [Chroma DB용] 의미 검색 대상: Title
        chroma_documents.append(Document(page_content=title, metadata=metadata.copy()))
        
        # [BM25용] 키워드 검색 대상: Cleaned Content
        bm25_documents.append(Document(page_content=cleaned_content, metadata=metadata.copy()))
        
    print(f"총 {len(chroma_documents)}개의 FAQ 데이터를 변환했습니다.")

    # 2. Vector DB (Chroma) 적재
    print(f"(1) Vector DB(Chroma)에 Title 임베딩 데이터를 적재합니다...")
    vector_store = get_vector_store("faq") # 사전에 정의된 vector store 호출 함수
    
    batch_size = 150
    for i in range(0, len(chroma_documents), batch_size):
        batch = chroma_documents[i : i + batch_size]
        vector_store.add_documents(batch)
        print(f"Chroma 적재 진행률: {min(i + batch_size, len(chroma_documents))} / {len(chroma_documents)}")

    # 3. BM25 코퍼스(Corpus) 저장
    print(f"(2) BM25 키워드 검색을 위한 Content 코퍼스를 저장합니다...")
    
    # [체크포인트 1] 문서가 제대로 담겼는지 확인
    print(f"저장할 BM25 문서 개수: {len(bm25_documents)}개")
    if len(bm25_documents) == 0:
        print("경고: 저장할 문서가 없습니다.")
    

    save_dir = os.path.abspath("./data/bm25_index")
    print(f"실제 저장될 디렉토리 절대 경로: {save_dir}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "bm25_corpus.pkl")
    
    try:
        with open(save_path, "wb") as f:
            pickle.dump(bm25_documents, f)
            
        # [체크포인트 3] 파일이 실제로 하드디스크에 쓰였는지 검증
        if os.path.exists(save_path):
            print(f"bm25기반 원문 검색용 데이터 구축 완료! ")
        else:
            print("디스크에 쓰이지 않았습니다.")
            
    except Exception as e:
        print(f"pickle 저장 중 에러 발생: {e}")
#======================================================
#=========자가수리 적재 시작
#======================================================
# def ingest_selfrepair_data(folder_path: str):
#     print(f"\n[{folder_path}] 디렉토리 내 자가수리 매뉴얼 데이터 로드 시작...")

#     # 입력된 경로가 폴더인지 확인
#     if not os.path.isdir(folder_path):
#         print(f"❌ 오류: '{folder_path}'는 유효한 폴더 경로가 아닙니다.")
#         return

#     documents = []

#     # 1. 폴더 내 모든 PDF 파일 탐색
#     pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
#     if not pdf_files:
#         print(f"'{folder_path}' 파일을 찾지 못했습니다.")
#         return

#     print(f"총 {len(pdf_files)}개의 PDF 파일을 발견했습니다. 차례대로 로드합니다.")

#     # 2. PDF 파일 순회 및 메타데이터 보강
#     for file_name in pdf_files:
#         file_path = os.path.join(folder_path, file_name)
#         print(f" '{file_name}' 읽는 중...")
        
#         try:
#             loader = PyPDFLoader(file_path)
#             pages = loader.load() 
            
#             for page in pages:
#                 metadata = page.metadata
#                 metadata.update({
#                     "source_file": file_name, # 모델 분리.
#                     "category": "자가수리",           
#                     "classification": "모바일 수리 가이드" 
#                 })
#                 documents.append(Document(page_content=page.page_content, metadata=metadata))
                
#         except Exception as e:
#             print(f" ❌ '{file_name}' 로드 중 오류 발생: {e}")
#             continue # 특정 파일에서 에러가 나도 멈추지 않고 다음 파일로 넘어감

#     print(f"\n총 {len(documents)}페이지의 PDF를 읽어왔습니다. 청킹을 진행합니다...")

#     # 3. 텍스트 청킹
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,       
#         chunk_overlap=50,    
#         separators=["\n\n", "\n", ".", " ", ""]
#     )
#     split_docs = text_splitter.split_documents(documents)
    
#     # 4. Vector DB 적재
#     print(f"총 {len(split_docs)}개의 청크(Chunk)를 Vector DB에 적재합니다...")
#     vector_store = get_vector_store("self-repair")
    
#     batch_size = 150
#     for i in range(0, len(split_docs), batch_size):
#         batch = split_docs[i : i + batch_size]
#         vector_store.add_documents(batch)
#         print(f"✅ 적재 진행률: {min(i + batch_size, len(split_docs))} / {len(split_docs)}")
        
#     print(f"🎉 자가수리 데이터 Vector DB(ChromaDB) 구축 완료 (처리된 파일 수: {len(pdf_files)}개)")

