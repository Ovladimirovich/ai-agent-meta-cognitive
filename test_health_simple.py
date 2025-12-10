#!/usr/bin/env python3
"""
Прямое тестирование эндпоинта /api/health
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_health_endpoint_simple():
    """Тестируем эндпоинт /api/health напрямую"""
    from fastapi.testclient import TestClient

    try:
        # Импортируем зависимости в правильном порядке
        from api.health_endpoints import HealthStatusResponse
        from monitoring.health_check_system import health_registry
        from fastapi import FastAPI

        # Создаем приложение
        app = FastAPI(title="AI Agent Meta-Cognitive API", version="1.0.0")

        # Добавляем CORS middleware
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Добавляем эндпоинты
        @app.get("/health", response_model=HealthStatusResponse)
        async def health_check():
            """Проверка здоровья системы"""
            from datetime import datetime
            try:
                # Запускаем все проверки
                results = await health_registry.run_all()

                # Получаем сводку
                summary = health_registry.get_summary(results)

                # Рассчитываем health score
                total_checks = summary['total_checks']
                if total_checks > 0:
                    health_score = (
                        (summary['healthy'] * 1.0 + summary['degraded'] * 0.5) / total_checks
                    )
                else:
                    health_score = 1.0  # Если нет проверок, считаем систему здоровой

                return HealthStatusResponse(
                    status=summary['overall_status'],
                    health_score=round(health_score, 2),
                    issues_count=summary['degraded'] + summary['unhealthy'],
                    last_check=summary['timestamp'],
                    details=summary
                )
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error getting health status: {e}")
                return HealthStatusResponse(
                    status="unhealthy",
                    health_score=0.0,
                    issues_count=1,
                    last_check=datetime.now().isoformat(),
                    details={"error": str(e)}
                )

        @app.get("/api/health", response_model=HealthStatusResponse)
        async def api_health_check():
            """Проверка здоровья системы (совместимость с API)"""
            from datetime import datetime
            try:
                # Запускаем все проверки
                results = await health_registry.run_all()

                # Получаем сводку
                summary = health_registry.get_summary(results)

                # Рассчитываем health score
                total_checks = summary['total_checks']
                if total_checks > 0:
                    health_score = (
                        (summary['healthy'] * 1.0 + summary['degraded'] * 0.5) / total_checks
                    )
                else:
                    health_score = 1.0  # Если нет проверок, считаем систему здоровой

                return HealthStatusResponse(
                    status=summary['overall_status'],
                    health_score=round(health_score, 2),
                    issues_count=summary['degraded'] + summary['unhealthy'],
                    last_check=summary['timestamp'],
                    details=summary
                )
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error getting health status: {e}")
                return HealthStatusResponse(
                    status="unhealthy",
                    health_score=0.0,
                    issues_count=1,
                    last_check=datetime.now().isoformat(),
                    details={"error": str(e)}
                )

        @app.get("/")
        async def root():
            """Корневой эндпоинт"""
            return {
                "message": "AI Agent Meta-Cognitive API",
                "version": "1.0.0",
                "status": "running",
                "health": "/health"
            }

        client = TestClient(app)

        # Тестируем основной эндпоинт /health
        print("Testing /health endpoint...")
        response = client.get("/health")
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data}")
            print("SUCCESS: /health endpoint works correctly")
        else:
            print(f"ERROR /health: {response.text}")

        print("\n" + "="*50 + "\n")

        # Тестируем эндпоинт /api/health
        print("Testing /api/health endpoint...")
        response = client.get("/api/health")
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data}")
            print("SUCCESS: /api/health endpoint works correctly")
        else:
            print(f"ERROR /api/health: {response.text}")

        print("\n" + "="*50 + "\n")

        # Тестируем корневой эндпоинт для проверки общего состояния приложения
        print("Testing root endpoint /...")
        response = client.get("/")
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data}")
            print("SUCCESS: Root endpoint works correctly")
        else:
            print(f"ERROR root endpoint: {response.text}")

        return True

    except Exception as e:
        print(f"Error creating application: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_health_endpoint_simple()
    if success:
        print("\nAll tests passed successfully!")
    else:
        print("\nTests failed")
        sys.exit(1)
