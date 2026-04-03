from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipelines.generation_pipeline import generate_cs_response 

app = FastAPI(title="Agentic Mobile CS API")

class QueryRequest(BaseModel):
    question: str

@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        # LangGraph 파이프라인 실행
        response_data = generate_cs_response(request.question)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))