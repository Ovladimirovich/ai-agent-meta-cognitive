#!/usr/bin/env python3
"""Simple test backend server for frontend testing"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pydantic import BaseModel

app = FastAPI(title="Test Backend API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentRequest(BaseModel):
    query: str
    user_id: str = "test_user"
    session_id: str = "test_session"

class AgentResponse(BaseModel):
    id: str
    content: str
    timestamp: str
    confidence: float = 0.85
    processing_time: float = 0.5

@app.get("/")
async def root():
    return {"message": "Test Backend API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/agent/process")
async def process_request(request: AgentRequest):
    # Simple mock response
    response = AgentResponse(
        id=f"resp_{int(datetime.now().timestamp())}",
        content=f"Я получил ваш запрос: '{request.query}'. Это тестовый ответ от AI агента.",
        timestamp=datetime.now().isoformat(),
        confidence=0.85,
        processing_time=0.5
    )
    return response

if __name__ == "__main__":
    import uvicorn
    print("Starting test backend server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
