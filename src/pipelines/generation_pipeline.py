from src.graph import rag_app

def generate_cs_response(question: str, selected_device: str = "선택하지 않음", thread_id: str = "default_user"): 
    
    config = {"configurable": {"thread_id": thread_id}}
    print(f" [Pipeline Start] 사용자 질문 접수: {question} (사전 선택 기기: {selected_device})")
 
    try:
        result = rag_app.invoke({"question": question, "selected_device": selected_device}, config)
        print("[Pipeline End] 최종 답변 생성 완료")
        return result
        
    except Exception as e:
        print(f"[Pipeline Error] 그래프 실행 중 오류 발생: {e}")
        return 0