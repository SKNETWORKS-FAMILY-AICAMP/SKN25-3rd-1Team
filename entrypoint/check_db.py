import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def inspect_chromadb(collection_name: str):
    print(f"\n{'='*50}")
    print(f"🔍 ChromaDB 점검 시작 (컬렉션: '{collection_name}')")
    print(f"{'='*50}")
    
    # 1. DB 연결 설정
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./data/vector_store")
    
    # DB 폴더 존재 여부 확인
    if not os.path.exists(persist_directory):
        print(f"❌ DB 폴더를 찾을 수 없습니다: {persist_directory}")
        return

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    # Chroma 내부 컬렉션 객체 접근
    collection = vector_store._collection
    
    # 2. 총 데이터(Chunk) 개수 확인
    doc_count = collection.count()
    print(f"✅ 현재 적재된 청크(Chunk) 총 개수: {doc_count}개")
    
    if doc_count == 0:
        print(f"⚠️ 데이터가 비어있습니다. '{collection_name}' 적재 스크립트가 정상적으로 완료되었는지 확인해주세요.")
        return

    # 3. 샘플 데이터 1개 엿보기
    print("\n[DB 샘플 데이터 1개 확인]")
    sample_data = collection.peek(limit=1)
    if sample_data and sample_data['documents']:
        # 텍스트가 너무 길면 보기 힘드므로 150자까지만 출력
        print(f"📄 텍스트 내용: {sample_data['documents'][0][:150]}... (중략)")
        print(f"🏷️ 메타데이터: {sample_data['metadatas'][0]}")

    # 4. 간단한 테스트 검색 (컬렉션별 맞춤 키워드)

    if collection_name == "self-repair":
        test_query = "후면 커버 조립할 때 주의할 점"
    else:
        test_query = "스마트폰 액정이 깨졌을 때"
        
    print(f"\n[테스트 검색 실행] 검색어: '{test_query}'")
    
    # k=2 : 유사도 높은 상위 2개 추출
    results = vector_store.similarity_search(test_query, k=2)
    
    for i, doc in enumerate(results):
        print(f"\n--- 검색 결과 {i+1} ---")
        print(f"내용: {doc.page_content[:150]}... (중략)")
        print(f"출처(메타데이터): {doc.metadata}")
        
    print("\n")

if __name__ == "__main__":
    # 점검할 컬렉션 이름들을 리스트로 지정하여 한 번에 모두 검사합니다.
    target_collections = ["faq", "self-repair"]
    
    for col_name in target_collections:
        inspect_chromadb(collection_name=col_name)