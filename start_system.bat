@echo off
echo Starting AI Agent Meta Cognitive System in venv...

REM Активация виртуального окружения
call venv\Scripts\activate.bat

REM Запуск сервера через uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

echo System started successfully!
echo Main application: http://localhost:8000
echo PostgreSQL: localhost:5432 (should be running separately)
echo Redis: localhost:6379 (should be running separately)
echo ChromaDB: http://localhost:8001 (should be running separately)

pause
