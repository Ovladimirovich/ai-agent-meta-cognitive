#!/usr/bin/env python3
"""
Минимальный тест API сервера
"""

import uvicorn
from fastapi import FastAPI, Depends
from api.input_validator import validate_query
from agent.core.input_preprocessor import InputPreprocessor

app = FastAPI(title="AI Agent API - Test")

preprocessor = InputPreprocessor()

@app.get("/")
async def root():
    return {"message": "AI Agent API Test Server", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "test"}

@app.post("/agent/process")
async def process_agent_request(query: str = Depends(validate_query)):
    """Тестовый эндпоинт агента с валидацией"""
    try:
        # Проверяем безопасность
        security = preprocessor.validate_security(query)
        comprehensive = preprocessor.validate_comprehensive(query)

        return {
            "query": query,
            "security_check": {
                "is_safe": security["is_safe"],
                "risk_level": security["risk_level"],
                "found_words": security["checks"]["blocked_words"]["found_words"]
            },
            "comprehensive_check": {
                "is_valid": comprehensive["is_valid"],
                "is_safe": comprehensive["is_safe"],
                "risk_level": comprehensive["risk_level"]
            },
            "status": "processed"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.get("/cache/status")
async def cache_status():
    """Статус кэширования"""
    try:
        from cache.cache_system_enhanced import EnhancedCacheSystem
        cache = EnhancedCacheSystem()
        stats = cache.get_stats()
        return {"cache_status": "active", "stats": stats}
    except Exception as e:
        return {"cache_status": "error", "error": str(e)}

if __name__ == "__main__":
    print("🚀 Запуск тестового API сервера...")
    print("📍 Доступно на: http://localhost:8001")
    print("🔧 Эндпоинты:")
    print("   GET  /           - Корень")
    print("   GET  /health     - Проверка здоровья")
    print("   POST /agent/process - Обработка запроса (с валидацией)")
    print("   GET  /cache/status  - Статус кэша")
    print("🛑 Для остановки нажмите Ctrl+C")

    uvicorn.run(app, host="localhost", port=8002)
