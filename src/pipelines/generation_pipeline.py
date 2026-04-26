import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from src.graph import rag_app
from src.utils.logger import save_usage_log

load_dotenv()

def generate_cs_response(question: str, selected_device: str = "선택하지 않음", thread_id: str = "default_user", selected_language: str = "korean"): 
    config = {"configurable": {"thread_id": thread_id}}
    trace_id = str(uuid.uuid4())
    print(f" [Pipeline Start] 사용자 질문 접수: {question} (TraceID: {trace_id})")
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "trace_id": trace_id,
        "thread_id": thread_id,
        "selected_device": selected_device,
        "question": question,
        "status": "pending"
    }
 
    try:
        result = rag_app.invoke(
            {
                "messages": [("user", question)],
                "selected_device": selected_device,
                "selected_language": selected_language,
                "trace_id": trace_id
            }, 
            config
        )
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
        save_usage_log(log_entry)
        
        return result
        
    except Exception as e:
        print(f"[Pipeline Error] 그래프 실행 중 오류 발생: {e}")
        
        log_entry.update({
            "status": "error",
            "error_msg": str(e)
        })
        save_usage_log(log_entry)
        
        return 0