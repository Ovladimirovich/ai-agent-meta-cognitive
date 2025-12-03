"""
Система предварительной загрузки часто используемых данных для улучшения скорости обработки запросов.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..caching.cache_manager import get_cache_instance, make_cache_key
from ..memory.memory_manager import MemoryManager
from .models import AgentConfig

logger = logging.getLogger(__name__)


class PreloadManager:
    """
    Менеджер предварительной загрузки данных для ускорения обработки запросов.
    Загружает часто используемые данные в кэш заранее.
    """
    
    def __init__(self, memory_manager: MemoryManager, config: AgentConfig):
        self.memory_manager = memory_manager
        self.config = config
        self.preloaded_data: Dict[str, Any] = {}
        self.preload_schedule: Dict[str, timedelta] = {}
        self._preload_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Определяем часто используемые данные для предварительной загрузки
        self.frequently_used_data = [
            "common_responses",
            "system_prompts", 
            "domain_knowledge",
            "user_preferences",
            "context_templates"
        ]
        
    async def start_preloading(self):
        """Запуск фоновой задачи предварительной загрузки"""
        if self._running:
            return
            
        self._running = True
        self._preload_task = asyncio.create_task(self._preload_worker())
        logger.info("Preload manager started")
        
    async def stop_preloading(self):
        """Остановка фоновой задачи предварительной загрузки"""
        if not self._running:
            return
            
        self._running = False
        if self._preload_task:
            self._preload_task.cancel()
            try:
                await self._preload_task
            except asyncio.CancelledError:
                pass
        logger.info("Preload manager stopped")
        
    async def _preload_worker(self):
        """Фоновая задача для предварительной загрузки данных"""
        while self._running:
            try:
                # Загружаем часто используемые данные
                await self.preload_frequently_used_data()
                
                # Ждем перед следующей итерацией
                await asyncio.sleep(300)  # 5 минут между обновлениями
                
            except asyncio.CancelledError:
                logger.info("Preload worker task was cancelled")
                break
            except Exception as e:
                logger.error(f"Error in preload worker: {e}")
                await asyncio.sleep(60)  # Ждем минуту перед повторной попыткой
                
    async def preload_frequently_used_data(self):
        """Предварительная загрузка часто используемых данных"""
        logger.info("Starting preloading of frequently used data...")
        
        preload_tasks = []
        
        # Загрузка часто используемых ответов
        preload_tasks.append(self._preload_common_responses())
        
        # Загрузка системных промптов
        preload_tasks.append(self._preload_system_prompts())
        
        # Загрузка доменных знаний
        preload_tasks.append(self._preload_domain_knowledge())
        
        # Загрузка предпочтений пользователей
        preload_tasks.append(self._preload_user_preferences())
        
        # Загрузка шаблонов контекста
        preload_tasks.append(self._preload_context_templates())
        
        # Выполняем все задачи предзагрузки параллельно
        results = await asyncio.gather(*preload_tasks, return_exceptions=True)
        
        successful_preloads = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Completed preloading: {successful_preloads}/{len(preload_tasks)} tasks successful")
        
    async def _preload_common_responses(self):
        """Предварительная загрузка часто используемых ответов"""
        try:
            # Получаем часто используемые ответы из памяти
            common_responses = await self.memory_manager.retrieve_semantic_memory(
                query="common responses", 
                limit=50,
                metadata_filter={"preload_priority": {"$gt": 0}}
            )
            
            cache = get_cache_instance()
            for i, response in enumerate(common_responses):
                cache_key = make_cache_key("common_response", i, str(response.id))
                await cache.set(cache_key, response.content, ttl=3600)  # 1 час
                
            logger.info(f"Preloaded {len(common_responses)} common responses")
            
        except Exception as e:
            logger.error(f"Error preloading common responses: {e}")
            
    async def _preload_system_prompts(self):
        """Предварительная загрузка системных промптов"""
        try:
            # Основные системные промпты для агента
            system_prompts = {
                "meta_cognition": "Ты - AI агент с мета-познанием. Ты можешь анализировать свои мысли, оценивать уверенность в ответах и использовать различные инструменты для решения задач.",
                "task_analysis": "Проанализируй задачу по следующим критериям: сложность, необходимые инструменты, временные рамки.",
                "response_generation": "Сформируй ответ, учитывая контекст, точность и полезность для пользователя.",
                "error_handling": "Объясни пользователю проблему и предложи альтернативные решения."
            }
            
            cache = get_cache_instance()
            for name, prompt in system_prompts.items():
                cache_key = make_cache_key("system_prompt", name)
                await cache.set(cache_key, prompt, ttl=7200)  # 2 часа
                
            logger.info(f"Preloaded {len(system_prompts)} system prompts")
            
        except Exception as e:
            logger.error(f"Error preloading system prompts: {e}")
            
    async def _preload_domain_knowledge(self):
        """Предварительная загрузка доменных знаний"""
        try:
            # Загружаем важные доменные знания из семантической памяти
            domain_topics = ["AI", "machine learning", "cognitive systems", "meta-cognition", "common questions"]
            
            cache = get_cache_instance()
            for topic in domain_topics:
                knowledge_items = await self.memory_manager.retrieve_semantic_memory(
                    query=topic,
                    limit=20,
                    metadata_filter={"importance": {"$gt": 0.7}}
                )
                
                for i, item in enumerate(knowledge_items):
                    cache_key = make_cache_key("domain_knowledge", topic, i, str(item.id))
                    await cache.set(cache_key, item.content, ttl=10800)  # 3 часа
                    
            logger.info(f"Preloaded domain knowledge for {len(domain_topics)} topics")
            
        except Exception as e:
            logger.error(f"Error preloading domain knowledge: {e}")
            
    async def _preload_user_preferences(self):
        """Предварительная загрузка предпочтений пользователей"""
        try:
            # Загружаем общие предпочтения пользователей
            user_preferences = await self.memory_manager.retrieve_episodic_memory(
                limit=100,
                metadata_filter={"preference_data": True}
            )
            
            # Группируем и анализируем общие предпочтения
            common_preferences = self._analyze_common_preferences(user_preferences)
            
            cache = get_cache_instance()
            for pref_name, pref_value in common_preferences.items():
                cache_key = make_cache_key("user_preference", pref_name)
                await cache.set(cache_key, pref_value, ttl=1800)  # 30 минут
                
            logger.info(f"Preloaded {len(common_preferences)} user preferences")
            
        except Exception as e:
            logger.error(f"Error preloading user preferences: {e}")
            
    def _analyze_common_preferences(self, preferences_list) -> Dict[str, Any]:
        """Анализ общих предпочтений пользователей"""
        # Простая реализация - в реальном приложении может быть сложнее
        common_prefs = {}
        
        for pref in preferences_list:
            if hasattr(pref, 'metadata') and pref.metadata:
                for key, value in pref.metadata.items():
                    if key not in common_prefs:
                        common_prefs[key] = value
                        
        return common_prefs
        
    async def _preload_context_templates(self):
        """Предварительная загрузка шаблонов контекста"""
        try:
            # Основные шаблоны контекста
            context_templates = {
                "conversation": {
                    "system_message": "Ты ведешь разговор с пользователем. Будь дружелюбным и полезным.",
                    "context_format": "Предыдущий контекст: {previous_context}\nТекущий запрос: {current_request}"
                },
                "task_execution": {
                    "system_message": "Ты выполняешь задачу для пользователя. Делай это шаг за шагом.",
                    "context_format": "Задача: {task}\nКонтекст: {context}\nШаги выполнения: {steps}"
                },
                "analysis": {
                    "system_message": "Проанализируй предоставленную информацию и предоставь структурированный ответ.",
                    "context_format": "Данные для анализа: {data}\nТребования: {requirements}"
                }
            }
            
            cache = get_cache_instance()
            for template_name, template in context_templates.items():
                cache_key = make_cache_key("context_template", template_name)
                await cache.set(cache_key, template, ttl=3600)  # 1 час
                
            logger.info(f"Preloaded {len(context_templates)} context templates")
            
        except Exception as e:
            logger.error(f"Error preloading context templates: {e}")
            
    async def get_preloaded_data(self, query: str) -> Optional[Any]:
        """Получение предварительно загруженных данных для запроса"""
        try:
            # Ищем в кэше предварительно загруженные данные, соответствующие запросу
            cache = get_cache_instance()
            
            # Проверяем кэш с разными префиксами
            possible_keys = [
                make_cache_key("common_response", query),
                make_cache_key("domain_knowledge", query),
                make_cache_key("context_template", query),
            ]
            
            for key in possible_keys:
                result = await cache.get(key)
                if result:
                    return result
                    
            # Также проверяем по ключевым словам в запросе
            query_lower = query.lower()
            if any(word in query_lower for word in ["привет", "здравствуй", "добрый день"]):
                greeting_key = make_cache_key("system_prompt", "greeting_response")
                result = await cache.get(greeting_key)
                if result:
                    return result
                    
            if any(word in query_lower for word in ["помощь", "помоги", "как", "что"]):
                help_key = make_cache_key("system_prompt", "help_response")
                result = await cache.get(help_key)
                if result:
                    return result
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting preloaded data: {e}")
            return None
            
    def register_preload_item(self, item_id: str, data: Any, ttl: int = 3600):
        """Регистрация элемента для предварительной загрузки"""
        self.preloaded_data[item_id] = {
            'data': data,
            'ttl': ttl,
            'last_updated': datetime.now()
        }
        
    async def preload_specific_item(self, item_id: str):
        """Предварительная загрузка конкретного элемента"""
        if item_id in self.preloaded_data:
            cache = get_cache_instance()
            item_data = self.preloaded_data[item_id]
            cache_key = make_cache_key("preloaded_item", item_id)
            await cache.set(cache_key, item_data['data'], ttl=item_data['ttl'])
            logger.info(f"Preloaded specific item: {item_id}")