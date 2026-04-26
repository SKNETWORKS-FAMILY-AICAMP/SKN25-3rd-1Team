from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipelines.generation_pipeline import generate_cs_response 

app = FastAPI(title="Agentic Mobile CS API")

class QueryRequest(BaseModel):
    question: str
    selected_device: str = "선택하지 않음"
    thread_id: str = "streamlit_user"
    selected_language: str = "korean"

@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        response_data = generate_cs_response(
            question=request.question, 
            selected_device=request.selected_device, 
            thread_id=request.thread_id,
            selected_language=request.selected_language
        )
        
        answer = ""
        if isinstance(response_data, dict) and "messages" in response_data:
            messages = response_data.get("messages", [])
            if messages:
                answer = messages[-1].content
        
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))