import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.pipelines.embedding_pipeline import get_vector_store


#=========자가수리 적재 시작
def ingest_selfrepair_data(folder_path: str):
    print(f"\n[{folder_path}] 디렉토리 내 자가수리 매뉴얼 데이터 로드 시작...")

    # 입력된 경로가 폴더인지 확인
    if not os.path.isdir(folder_path):
        print(f"❌ 오류: '{folder_path}'는 유효한 폴더 경로가 아닙니다.")
        return

    documents = []

    # 1. 폴더 내 모든 PDF 파일 탐색
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"'{folder_path}' 파일을 찾지 못했습니다.")
        return

    print(f"총 {len(pdf_files)}개의 PDF 파일을 발견했습니다. 차례대로 로드합니다.")

    # 2. PDF 파일 순회 및 메타데이터 보강
    for file_name in pdf_files:
        file_path = os.path.join(folder_path, file_name)
        print(f" '{file_name}' 읽는 중...")
        
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load() 
            
            for page in pages:
                metadata = page.metadata
                metadata.update({
                    "source_file": file_name, # 모델 분리.
                    "category": "자가수리",           
                    "classification": "모바일 수리 가이드" 
                })
                documents.append(Document(page_content=page.page_content, metadata=metadata))
                
        except Exception as e:
            print(f" ❌ '{file_name}' 로드 중 오류 발생: {e}")
            continue # 특정 파일에서 에러가 나도 멈추지 않고 다음 파일로 넘어감

    print(f"\n총 {len(documents)}페이지의 PDF를 읽어왔습니다. 청킹을 진행합니다...")

    # 3. 텍스트 청킹
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       
        chunk_overlap=50,    
        separators=["\n\n", "\n", ".", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    
    # 4. Vector DB 적재
    print(f"총 {len(split_docs)}개의 청크(Chunk)를 Vector DB에 적재합니다...")
    vector_store = get_vector_store("self-repair")
    
    batch_size = 150
    for i in range(0, len(split_docs), batch_size):
        batch = split_docs[i : i + batch_size]
        vector_store.add_documents(batch)
        print(f"✅ 적재 진행률: {min(i + batch_size, len(split_docs))} / {len(split_docs)}")
        
    print(f"🎉 자가수리 데이터 Vector DB(ChromaDB) 구축 완료 (처리된 파일 수: {len(pdf_files)}개)")

#=========FAQ 적재 시작
def ingest_faq_data(file_path: str):
    """FAQ 데이터를 파싱하고 Vector DB에 적재하는 파이프라인"""
    print(f"[{file_path}] 데이터 로드 시작...")
    
    file_name = os.path.basename(file_path)
    
    # 1. 데이터 로드 및 전처리
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

    documents = []
    
    # 2. Document 객체 생성
    for _, row in df.iterrows():
        content = (
            f"질문(고객증상): {row.get('제목', '')}\n"
            f"관련 카테고리: {row.get('카테고리', '')}\n"
            f"해결책(가이드): {row.get('본문', '')}"
        )
        
        views_val = str(row.get('조회수', '0'))
        views_int = int(float(views_val)) if views_val.replace('.','',1).isdigit() else 0

        metadata = {
            "source_file": file_name,
            "source_id": str(row.get('ID', '')),
            "title": str(row.get('제목', '')),
            "category": str(row.get('카테고리', '')),
            "classification": str(row.get('분류', '')),
            "views": views_int
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
        
    print(f"총 {len(documents)}개의 FAQ 데이터를 변환했습니다. 청킹을 진행합니다...")

    # 3. 텍스트 청킹
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,       
        chunk_overlap=10,    
        separators=["\n\n", "\n", ".", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    
    # 4. Vector DB 적재
    print(f"총 {len(split_docs)}개의 청크(Chunk)를 Vector DB에 적재합니다...")
    vector_store = get_vector_store("faq")
    
    batch_size = 150
    for i in range(0, len(split_docs), batch_size):
        batch = split_docs[i : i + batch_size]
        vector_store.add_documents(batch)
        print(f"적재 진행률: {min(i + batch_size, len(split_docs))} / {len(split_docs)}")
        
    print(f" Vector DB(ChromaDB) 구축 완료! (적재된 원본 파일: {file_name})")