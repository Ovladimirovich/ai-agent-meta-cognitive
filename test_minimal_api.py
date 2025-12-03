#!/usr/bin/env python3
"""Minimal FastAPI test to check if the framework works"""

from fastapi import FastAPI
from datetime import datetime

app = FastAPI(title="Minimal Test API")

@app.get("/")
async def root():
    return {"message": "Minimal API works", "timestamp": datetime.now().isoformat()}

@app.get("/test")
async def test():
    return {"status": "ok", "data": "FastAPI is working"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
