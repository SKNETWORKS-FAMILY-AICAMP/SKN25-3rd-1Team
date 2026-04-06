import os
import json
from datetime import datetime
from src.graph import rag_app

LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "logs", "usage_log.jsonl")

def save_log(log_data: dict):
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Log Error] 로그 저장 실패: {e}")

def generate_cs_response(question: str, selected_device: str = "선택하지 않음", thread_id: str = "default_user"): 
    
    config = {"configurable": {"thread_id": thread_id}}
    print(f" [Pipeline Start] 사용자 질문 접수: {question} (사전 선택 기기: {selected_device})")
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "thread_id": thread_id,
        "selected_device": selected_device,
        "question": question,
        "status": "pending"
    }
 
    try:
        result = rag_app.invoke({"messages": [("user", question)], "selected_device": selected_device}, config)
        print("[Pipeline End] 최종 답변 생성 완료")
        
        final_answer = ""
        if result.get("messages"):
            final_answer = result["messages"][-1].content
        
        log_entry.update({
            "status": "success",
            "answer": final_answer,
            "device_model": result.get("device_model", ""),
            "is_hardware_issue": result.get("is_hardware_issue", False),
            "reliability_score": result.get("reliability_score", 0.0)
        })
        save_log(log_entry)
        
        return result
        
    except Exception as e:
        print(f"[Pipeline Error] 그래프 실행 중 오류 발생: {e}")
        
        log_entry.update({
            "status": "error",
            "error_msg": str(e)
        })
        save_log(log_entry)
        
        return 0