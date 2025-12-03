# Makefile для AI Агента с Мета-Познанием
# Обеспечивает простоту разработки и воспроизводимость для DOI

.PHONY: help install install-dev test test-cov lint format clean docker-build docker-run docker-dev docker-stop docs serve-docs ci cd release

# ================================
# Помощь
# ================================
help: ## Показать эту справку
	@echo "Доступные команды:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ================================
# Установка и настройка
# ================================
install: ## Установить основные зависимости
	pip install -r requirements.txt

install-dev: ## Установить зависимости для разработки
	pip install -r requirements-dev.txt
	pre-commit install

setup: ## Полная настройка проекта для разработки
	python -m venv venv
	venv\Scripts\activate && make install-dev
	@echo "Настройка завершена. Активируйте виртуальное окружение: venv\Scripts\activate"

# ================================
# Тестирование
# ================================
test: ## Запустить все тесты
	pytest tests/ -v

test-cov: ## Запустить тесты с покрытием
	pytest tests/ --cov=agent --cov-report=html --cov-report=term

test-fast: ## Быстрый запуск тестов (без интеграционных)
	pytest tests/ -v -m "not integration"

# ================================
# Качество кода
# ================================
lint: ## Проверить код линтерами
	flake8 agent/ api/ database/ integrations/
	mypy agent/ api/ database/ integrations/
	bandit -r agent/ api/ database/ integrations/

format: ## Форматировать код
	black agent/ api/ database/ integrations/
	isort agent/ api/ database/ integrations/

check: ## Полная проверка качества кода
	make lint
	make test-cov
	safety check

# ================================
# Docker команды
# ================================
docker-build: ## Собрать Docker образ
	docker build -t ai-agent-meta-cognitive .

docker-run: ## Запустить приложение в Docker
	docker-compose up -d ai-agent

docker-dev: ## Запустить development окружение
	docker-compose --profile dev up -d ai-agent-dev pgadmin redis-commander

docker-full: ## Запустить полное окружение (с базами данных)
	docker-compose up -d

docker-stop: ## Остановить все Docker контейнеры
	docker-compose down

docker-clean: ## Очистить Docker (удалить volumes)
	docker-compose down -v
	docker system prune -f

docker-logs: ## Посмотреть логи Docker контейнеров
	docker-compose logs -f ai-agent

# ================================
# Разработка
# ================================
run: ## Запустить приложение локально
	python -m api.main

run-dev: ## Запустить в режиме разработки с перезагрузкой
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

shell: ## Запустить Python shell с загруженным проектом
	python -c "import agent; print('AI Agent loaded successfully')"

jupyter: ## Запустить Jupyter notebook
	jupyter lab

# ================================
# Документация
# ================================
docs: ## Сгенерировать документацию
	mkdocs build

serve-docs: ## Запустить сервер документации
	mkdocs serve

# ================================
# Базы данных
# ================================
db-init: ## Инициализировать базу данных
	docker-compose exec ai-agent python -c "from database.postgres import PostgreSQLManager; import asyncio; asyncio.run(PostgreSQLManager('postgresql://ai_agent:ai_agent_password@postgres:5432/ai_agent').initialize())"

db-migrate: ## Запустить миграции базы данных
	alembic upgrade head

db-reset: ## Сбросить базу данных
	docker-compose exec postgres psql -U ai_agent -d ai_agent -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

# ================================
# CI/CD
# ================================
ci: ## Запустить CI пайплайн локально
	make check
	make docker-build
	docker run --rm ai-agent-meta-cognitive python -c "import agent; print('Import test passed')"

cd-prepare: ## Подготовить релиз
	make check
	make docs
	git tag -a v$(shell python -c "import agent; print(agent.__version__)") -m "Release v$(shell python -c "import agent; print(agent.__version__)")"

release: ## Создать релиз
	make cd-prepare
	git push --tags
	@echo "Создан релиз v$(shell python -c "import agent; print(agent.__version__)")"

# ================================
# Очистка
# ================================
clean: ## Очистить временные файлы
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -rf .mypy_cache/
	rm -rf .tox/

clean-all: ## Полная очистка (включая venv и Docker)
	make clean
	rm -rf venv/
	make docker-clean

# ================================
# DOI и публикация
# ================================
doi-prepare: ## Подготовить файлы для DOI
	@echo "Подготовка файлов для DOI..."
	@mkdir -p doi_files
	@cp requirements.txt doi_files/
	@cp requirements-dev.txt doi_files/
	@cp Dockerfile doi_files/
	@cp docker-compose.yml doi_files/
	@python -c "
import json
import agent
metadata = {
    'title': 'AI Agent with Meta-Cognition',
    'version': agent.__version__,
    'description': 'Advanced AI agent with meta-cognitive capabilities for autonomous learning and adaptation',
    'authors': [{'name': 'AI Research Team'}],
    'license': 'MIT',
    'keywords': ['AI', 'machine learning', 'meta-cognition', 'autonomous agents', 'neural networks'],
    'python_version': '>=3.8',
    'dependencies': open('requirements.txt').read().splitlines()
}
with open('doi_files/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"
	@echo "Файлы для DOI подготовлены в папке doi_files/"

# ================================
# Статус и информация
# ================================
status: ## Показать статус проекта
	@echo "=== AI Agent Meta-Cognitive Status ==="
	@echo "Python version: $(shell python --version)"
	@echo "Project version: $(shell python -c "import agent; print(agent.__version__)" 2>/dev/null || echo "Not installed")"
	@echo "Docker images: $(shell docker images | grep ai-agent | wc -l 2>/dev/null || echo "Docker not available")"
	@echo "Tests: $(shell find tests/ -name "*.py" | wc -l) files"
	@echo "Code lines: $(shell find agent/ api/ database/ integrations/ -name "*.py" | xargs wc -l | tail -1 | awk '{print $$1}')"
	@echo "Coverage: $(shell make test-cov 2>/dev/null | grep TOTAL | awk '{print $$4}' || echo "Run make test-cov")"

info: ## Показать информацию о проекте
	@echo "AI Agent with Meta-Cognition"
	@echo "============================"
	@echo "Version: $(shell python -c "import agent; print(agent.__version__)" 2>/dev/null || echo "Unknown")"
	@echo "Python: $(shell python --version)"
	@echo "Platform: $(shell python -c "import platform; print(platform.platform())")"
	@echo ""
	@echo "Основные компоненты:"
	@echo "- Agent Core: мета-познание и самообучение"
	@echo "- Learning Engine: адаптивное обучение"
	@echo "- Memory System: управление памятью"
	@echo "- Tool Orchestrator: координация инструментов"
	@echo "- API: REST/GraphQL интерфейсы"
	@echo ""
	@echo "Для запуска: make setup && make run-dev"
