import time
import logging
from .base_tool import BaseTool, Task, ToolResult
from integrations.circuit_breaker import circuit_breaker_decorator, CircuitBreakerConfig

logger = logging.getLogger(__name__)

class RAGTool(BaseTool):
    name = "rag_search"
    description = "Семантический поиск и извлечение информации из базы знаний"
    version = "1.0.0"

    def __init__(self):
        self.rag_manager = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Инициализация RAG менеджера"""
        if self._initialized:
            return True

        try:
            from rag.hybrid_rag import HybridRAGSystem, SearchMethod

            # Используем новую гибридную систему вместо старого RAGManager
            self.rag_system = HybridRAGSystem(
                db_path="./rag_db.sqlite",  # SQLite вместо ChromaDB
                embedding_provider="auto",  # Автоматический выбор (FastEmbed -> TF-IDF -> Random)
                enable_caching=True
            )

            success = await self.rag_system.initialize()
            if success:
                self._initialized = True
                logger.info("RAGTool initialized with new HybridRAGSystem")
                return True
            else:
                logger.error("Failed to initialize HybridRAGSystem")
                return False

        except Exception as e:
            logger.error(f"Failed to initialize RAGTool: {e}")
            return False

    def can_handle(self, task: Task) -> bool:
        """Проверка необходимости RAG поиска"""
        if not isinstance(task.query, str):
            return False

        query_lower = task.query.lower().strip()

        # Исключаем простые приветствия и бытовые вопросы
        casual_phrases = [
            'привет', 'здравствуй', 'добрый день', 'доброе утро', 'добрый вечер',
            'как дела', 'что нового', 'чем занимаешься', 'как жизнь',
            'спасибо', 'пожалуйста', 'извини', 'прощай', 'до свидания'
        ]

        if any(phrase in query_lower for phrase in casual_phrases):
            return False

        # Ключевые слова указывающие на поиск информации
        search_indicators = [
            'что такое', 'как работает', 'почему происходит', 'где находится',
            'когда произошло', 'кто такой', 'расскажи о', 'объясни',
            'найди информацию', 'покажи данные', 'узнать о',
            'информация о', 'данные по', 'статистика', 'история'
        ]

        return any(indicator in query_lower for indicator in search_indicators)

    @circuit_breaker_decorator("rag_tool", CircuitBreakerConfig(
        failure_threshold=4,
        recovery_timeout=120.0,
        timeout=45.0,
        name="rag_tool"
    ))
    async def execute(self, task: Task) -> ToolResult:
        """Выполнение RAG поиска"""
        start_time = time.time()

        try:
            if not await self.initialize():
                return ToolResult(
                    success=False,
                    data=None,
                    metadata={},
                    execution_time=time.time() - start_time,
                    error_message="RAG system not available"
                )

            # Выполнение поиска с новой системой
            search_results = await self.rag_system.search(
                query=task.query,
                method="hybrid",  # Используем гибридный поиск
                limit=task.metadata.get('top_k', 5)
            )

            execution_time = time.time() - start_time

            # Конвертируем результаты в старый формат для совместимости
            sources = []
            context_parts = []

            for result in search_results:
                source = {
                    'content': result.content,
                    'metadata': result.metadata,
                    'similarity': result.similarity,
                    'search_type': result.search_type,
                    'method_used': result.method_used.value
                }
                sources.append(source)
                context_parts.append(f"[Источник]: {result.content}")

            context = "\n\n".join(context_parts)

            # Создаем метрики в старом формате
            search_metrics = {
                'total_chunks_found': len(search_results),
                'relevant_chunks': len(search_results),
                'search_time': execution_time,
                'method_used': 'hybrid'
            }

            return ToolResult(
                success=True,
                data={
                    'results': sources,
                    'context': context,
                    'search_metrics': search_metrics
                },
                metadata={
                    'tool': self.name,
                    'results_count': len(search_results),
                    'total_chunks_found': len(search_results),
                    'is_final': False  # RAG дает контекст, не финальный ответ
                },
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"RAGTool execution error: {e}")

            return ToolResult(
                success=False,
                data={},
                metadata={'tool': self.name},
                execution_time=execution_time,
                error_message=str(e)
            )

    async def get_status(self) -> dict:
        """Получение статуса инструмента"""
        status = {
            'name': self.name,
            'initialized': self._initialized,
            'version': self.version,
            'capabilities': ['semantic_search', 'information_retrieval', 'context_extraction', 'hybrid_search'],
        }

        if self._initialized and hasattr(self, 'rag_system'):
            try:
                # Получаем статистику от новой системы
                system_stats = await self.rag_system.get_stats()
                status.update({
                    'embedding_provider': system_stats.get('embedding_provider', {}),
                    'available_methods': system_stats.get('available_methods', []),
                    'vector_store': system_stats.get('vector_store', {}),
                    'cache_stats': system_stats.get('cache_stats', {})
                })
            except Exception as e:
                logger.warning(f"Failed to get system stats: {e}")

        return status
