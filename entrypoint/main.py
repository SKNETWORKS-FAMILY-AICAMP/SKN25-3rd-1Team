from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipelines.generation_pipeline import generate_cs_response 

app = FastAPI(title="Agentic Mobile CS API")

class QueryRequest(BaseModel):
    question: str
    selected_device: str = "선택하지 않음"
    thread_id: str = "streamlit_user"

@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        # LangGraph 파이프라인 실행
        response_data = generate_cs_response(
            question=request.question, 
            selected_device=request.selected_device, 
            thread_id=request.thread_id
        )
        
        answer = ""
        # LangGraph 메시지 객체는 Pydantic 직렬화 과정에서 에러가 발생할 수 있으므로 최종 텍스트만 추출
        if isinstance(response_data, dict) and "messages" in response_data:
            messages = response_data.get("messages", [])
            if messages:
                answer = messages[-1].content
        
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))