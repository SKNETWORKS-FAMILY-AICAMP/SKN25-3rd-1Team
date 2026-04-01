from src.graph import rag_app

def generate_cs_response(question: str, thread_id: str = "default_user"): 
    
    config = {"configurable": {"thread_id": thread_id}}
    print(f" [Pipeline Start] 사용자 질문 접수: {question}")
 
    try:
        result = rag_app.invoke({"question": question}, config)
        print("[Pipeline End] 최종 답변 생성 완료")
        return result
        
    except Exception as e:
        print(f"[Pipeline Error] 그래프 실행 중 오류 발생: {e}")
        # 시스템 에러 발생 시에도 무너지지 않도록 안전한 기본값 반환
        return 0