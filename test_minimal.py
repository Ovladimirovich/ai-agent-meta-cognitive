#!/usr/bin/env python3
"""
Минимальный тест сервера без lifespan
"""

import logging
from fastapi import FastAPI
import uvicorn

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание минимального приложения
app = FastAPI(title="Test API", version="1.0.0")

@app.get("/")
async def root():
    logger.info("🔄 Root endpoint called")
    return {"message": "Test API", "status": "ok"}

@app.get("/health")
async def health():
    logger.info("🔄 Health endpoint called")
    return {"status": "healthy"}

if __name__ == "__main__":
    print("Starting minimal test server...")
    uvicorn.run(app, host="127.0.0.1", port=8004, log_level="info")
