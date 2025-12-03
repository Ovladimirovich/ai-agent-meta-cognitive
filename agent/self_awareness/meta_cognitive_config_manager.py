"""
Менеджер конфигурации мета-когнитивных параметров
Фаза 3: Расширенные мета-когнитивные функции
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class CognitiveLoadThresholds:
    """Пороги когнитивной нагрузки"""
    low_to_medium: float = 0.3
    medium_to_high: float = 0.6
    high_to_critical: float = 0.8


@dataclass
class AdaptationSettings:
    """Настройки адаптации"""
    enable_auto_adaptation: bool = True
    adaptation_frequency: int = 300  # секунды
    minimum_effectiveness_threshold: float = 0.1


@dataclass
class LearningParameters:
    """Параметры обучения"""
    learning_rate: float = 0.01
    memory_decay_rate: float = 0.001
    experience_importance_threshold: float = 0.7


@dataclass
class SelfReflectionSettings:
    """Настройки саморефлексии"""
    reflection_frequency: int = 60  # секунды
    min_confidence_for_reflection: float = 0.4
    enable_deep_reflection: bool = True


@dataclass
class MonitoringSettings:
    """Настройки мониторинга"""
    health_check_interval: int = 30  # секунды
    performance_monitoring_enabled: bool = True
    cognitive_load_monitoring_enabled: bool = True


@dataclass
class MetaCognitiveConfig:
    """Основная конфигурация мета-когнитивных параметров"""
    cognitive_load_thresholds: CognitiveLoadThresholds = None
    adaptation_settings: AdaptationSettings = None
    learning_parameters: LearningParameters = None
    self_reflection_settings: SelfReflectionSettings = None
    monitoring_settings: MonitoringSettings = None
    
    def __post_init__(self):
        if self.cognitive_load_thresholds is None:
            self.cognitive_load_thresholds = CognitiveLoadThresholds()
        if self.adaptation_settings is None:
            self.adaptation_settings = AdaptationSettings()
        if self.learning_parameters is None:
            self.learning_parameters = LearningParameters()
        if self.self_reflection_settings is None:
            self.self_reflection_settings = SelfReflectionSettings()
        if self.monitoring_settings is None:
            self.monitoring_settings = MonitoringSettings()


class MetaCognitiveConfigManager:
    """
    Менеджер конфигурации мета-когнитивных параметров
    
    Обеспечивает настройку и управление параметрами мета-когнитивных функций
    """
    
    def __init__(self, agent_core: 'AgentCore'):
        self.agent_core = agent_core
        self.config: MetaCognitiveConfig = MetaCognitiveConfig()
        self.defaults: MetaCognitiveConfig = MetaCognitiveConfig()
        
        # Загружаем конфигурацию
        self._load_config_from_agent()
        
        logger.info("MetaCognitiveConfigManager initialized")

    def _load_config_from_agent(self):
        """Загрузка конфигурации из агента"""
        try:
            if hasattr(self.agent_core, 'meta_config') and self.agent_core.meta_config:
                # Загружаем сохраненную конфигурацию
                saved_config = self.agent_core.meta_config
                self.config = self._dict_to_config(saved_config)
            else:
                # Используем значения по умолчанию
                self.config = self.defaults
                self.agent_core.meta_config = self._config_to_dict(self.config)
                
            logger.info("Meta-cognitive configuration loaded")
        except Exception as e:
            logger.error(f"Error loading meta-cognitive config: {e}")
            self.config = self.defaults
            self.agent_core.meta_config = self._config_to_dict(self.config)

    def _save_config_to_agent(self):
        """Сохранение конфигурации в агенте"""
        try:
            self.agent_core.meta_config = self._config_to_dict(self.config)
            logger.info("Meta-cognitive configuration saved to agent")
        except Exception as e:
            logger.error(f"Error saving meta-cognitive config: {e}")

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> MetaCognitiveConfig:
        """Конвертация словаря в объект конфигурации"""
        try:
            # Создаем конфигурацию с базовыми значениями
            config = MetaCognitiveConfig()
            
            # Обновляем значения из словаря
            if 'cognitive_load_thresholds' in config_dict:
                thresholds = config_dict['cognitive_load_thresholds']
                config.cognitive_load_thresholds = CognitiveLoadThresholds(
                    low_to_medium=thresholds.get('low_to_medium', 0.3),
                    medium_to_high=thresholds.get('medium_to_high', 0.6),
                    high_to_critical=thresholds.get('high_to_critical', 0.8)
                )
            
            if 'adaptation_settings' in config_dict:
                settings = config_dict['adaptation_settings']
                config.adaptation_settings = AdaptationSettings(
                    enable_auto_adaptation=settings.get('enable_auto_adaptation', True),
                    adaptation_frequency=settings.get('adaptation_frequency', 300),
                    minimum_effectiveness_threshold=settings.get('minimum_effectiveness_threshold', 0.1)
                )
            
            if 'learning_parameters' in config_dict:
                params = config_dict['learning_parameters']
                config.learning_parameters = LearningParameters(
                    learning_rate=params.get('learning_rate', 0.01),
                    memory_decay_rate=params.get('memory_decay_rate', 0.001),
                    experience_importance_threshold=params.get('experience_importance_threshold', 0.7)
                )
            
            if 'self_reflection_settings' in config_dict:
                settings = config_dict['self_reflection_settings']
                config.self_reflection_settings = SelfReflectionSettings(
                    reflection_frequency=settings.get('reflection_frequency', 60),
                    min_confidence_for_reflection=settings.get('min_confidence_for_reflection', 0.4),
                    enable_deep_reflection=settings.get('enable_deep_reflection', True)
                )
            
            if 'monitoring_settings' in config_dict:
                settings = config_dict['monitoring_settings']
                config.monitoring_settings = MonitoringSettings(
                    health_check_interval=settings.get('health_check_interval', 30),
                    performance_monitoring_enabled=settings.get('performance_monitoring_enabled', True),
                    cognitive_load_monitoring_enabled=settings.get('cognitive_load_monitoring_enabled', True)
                )
            
            return config
        except Exception as e:
            logger.error(f"Error converting dict to config: {e}")
            return MetaCognitiveConfig()  # Возвращаем конфигурацию по умолчанию

    def _config_to_dict(self, config: MetaCognitiveConfig) -> Dict[str, Any]:
        """Конвертация объекта конфигурации в словарь"""
        return {
            'cognitive_load_thresholds': asdict(config.cognitive_load_thresholds),
            'adaptation_settings': asdict(config.adaptation_settings),
            'learning_parameters': asdict(config.learning_parameters),
            'self_reflection_settings': asdict(config.self_reflection_settings),
            'monitoring_settings': asdict(config.monitoring_settings),
            'last_updated': datetime.now().isoformat()
        }

    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Получение конфигурации
        
        Args:
            section: Раздел конфигурации (опционально)
            
        Returns:
            Dict: Конфигурация
        """
        full_config = self._config_to_dict(self.config)
        
        if section:
            return full_config.get(section, {})
        return full_config

    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обновление конфигурации
        
        Args:
            updates: Обновления конфигурации
            
        Returns:
            Dict: Результат обновления
        """
        try:
            # Валидация обновлений
            validation_errors = self.validate_config_updates(updates)
            if validation_errors:
                return {
                    'success': False,
                    'errors': validation_errors,
                    'message': 'Configuration validation failed'
                }
            
            # Применение обновлений
            updated_config = self._apply_config_updates(self.config, updates)
            self.config = updated_config
            
            # Сохранение в агенте
            self._save_config_to_agent()
            
            # Применение изменений к активным компонентам
            self._apply_config_changes()
            
            return {
                'success': True,
                'updated_config': self.get_config(),
                'message': 'Configuration updated successfully'
            }
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to update configuration'
            }

    def _apply_config_updates(self, current_config: MetaCognitiveConfig, updates: Dict[str, Any]) -> MetaCognitiveConfig:
        """Применение обновлений к конфигурации"""
        # Создаем новую конфигурацию на основе текущей
        new_config = MetaCognitiveConfig()
        
        # Обновляем пороги когнитивной нагрузки
        if 'cognitive_load_thresholds' in updates:
            thresholds_updates = updates['cognitive_load_thresholds']
            new_config.cognitive_load_thresholds = CognitiveLoadThresholds(
                low_to_medium=thresholds_updates.get('low_to_medium', current_config.cognitive_load_thresholds.low_to_medium),
                medium_to_high=thresholds_updates.get('medium_to_high', current_config.cognitive_load_thresholds.medium_to_high),
                high_to_critical=thresholds_updates.get('high_to_critical', current_config.cognitive_load_thresholds.high_to_critical)
            )
        else:
            new_config.cognitive_load_thresholds = current_config.cognitive_load_thresholds
        
        # Обновляем настройки адаптации
        if 'adaptation_settings' in updates:
            settings_updates = updates['adaptation_settings']
            new_config.adaptation_settings = AdaptationSettings(
                enable_auto_adaptation=settings_updates.get('enable_auto_adaptation', current_config.adaptation_settings.enable_auto_adaptation),
                adaptation_frequency=settings_updates.get('adaptation_frequency', current_config.adaptation_settings.adaptation_frequency),
                minimum_effectiveness_threshold=settings_updates.get('minimum_effectiveness_threshold', current_config.adaptation_settings.minimum_effectiveness_threshold)
            )
        else:
            new_config.adaptation_settings = current_config.adaptation_settings
        
        # Обновляем параметры обучения
        if 'learning_parameters' in updates:
            params_updates = updates['learning_parameters']
            new_config.learning_parameters = LearningParameters(
                learning_rate=params_updates.get('learning_rate', current_config.learning_parameters.learning_rate),
                memory_decay_rate=params_updates.get('memory_decay_rate', current_config.learning_parameters.memory_decay_rate),
                experience_importance_threshold=params_updates.get('experience_importance_threshold', current_config.learning_parameters.experience_importance_threshold)
            )
        else:
            new_config.learning_parameters = current_config.learning_parameters
        
        # Обновляем настройки саморефлексии
        if 'self_reflection_settings' in updates:
            settings_updates = updates['self_reflection_settings']
            new_config.self_reflection_settings = SelfReflectionSettings(
                reflection_frequency=settings_updates.get('reflection_frequency', current_config.self_reflection_settings.reflection_frequency),
                min_confidence_for_reflection=settings_updates.get('min_confidence_for_reflection', current_config.self_reflection_settings.min_confidence_for_reflection),
                enable_deep_reflection=settings_updates.get('enable_deep_reflection', current_config.self_reflection_settings.enable_deep_reflection)
            )
        else:
            new_config.self_reflection_settings = current_config.self_reflection_settings
        
        # Обновляем настройки мониторинга
        if 'monitoring_settings' in updates:
            settings_updates = updates['monitoring_settings']
            new_config.monitoring_settings = MonitoringSettings(
                health_check_interval=settings_updates.get('health_check_interval', current_config.monitoring_settings.health_check_interval),
                performance_monitoring_enabled=settings_updates.get('performance_monitoring_enabled', current_config.monitoring_settings.performance_monitoring_enabled),
                cognitive_load_monitoring_enabled=settings_updates.get('cognitive_load_monitoring_enabled', current_config.monitoring_settings.cognitive_load_monitoring_enabled)
            )
        else:
            new_config.monitoring_settings = current_config.monitoring_settings
        
        return new_config

    def validate_config_updates(self, updates: Dict[str, Any]) -> List[str]:
        """
        Валидация обновлений конфигурации
        
        Args:
            updates: Обновления конфигурации
            
        Returns:
            List[str]: Ошибки валидации
        """
        errors = []
        
        # Валидация порогов когнитивной нагрузки
        if 'cognitive_load_thresholds' in updates:
            thresholds = updates['cognitive_load_thresholds']
            low_med = thresholds.get('low_to_medium', 0.3)
            med_high = thresholds.get('medium_to_high', 0.6)
            high_crit = thresholds.get('high_to_critical', 0.8)
            
            if not (0 <= low_med <= med_high <= high_crit <= 1.0):
                errors.append("Cognitive load thresholds must be in ascending order between 0 and 1")
        
        # Валидация настроек адаптации
        if 'adaptation_settings' in updates:
            settings = updates['adaptation_settings']
            freq = settings.get('adaptation_frequency', 300)
            if freq < 1 or freq > 3600:  # 1 час максимум
                errors.append("Adaptation frequency must be between 1 and 3600 seconds")
            
            threshold = settings.get('minimum_effectiveness_threshold', 0.1)
            if not 0 <= threshold <= 1.0:
                errors.append("Minimum effectiveness threshold must be between 0 and 1")
        
        # Валидация параметров обучения
        if 'learning_parameters' in updates:
            params = updates['learning_parameters']
            lr = params.get('learning_rate', 0.01)
            if not 0 < lr <= 1.0:
                errors.append("Learning rate must be between 0 and 1")
            
            decay = params.get('memory_decay_rate', 0.001)
            if not 0 <= decay <= 0.1:
                errors.append("Memory decay rate must be between 0 and 0.1")
            
            importance = params.get('experience_importance_threshold', 0.7)
            if not 0 <= importance <= 1.0:
                errors.append("Experience importance threshold must be between 0 and 1")
        
        # Валидация настроек саморефлексии
        if 'self_reflection_settings' in updates:
            settings = updates['self_reflection_settings']
            freq = settings.get('reflection_frequency', 60)
            if freq < 1 or freq > 3600:  # 1 час максимум
                errors.append("Reflection frequency must be between 1 and 3600 seconds")
            
            min_conf = settings.get('min_confidence_for_reflection', 0.4)
            if not 0 <= min_conf <= 1.0:
                errors.append("Minimum confidence for reflection must be between 0 and 1")
        
        # Валидация настроек мониторинга
        if 'monitoring_settings' in updates:
            settings = updates['monitoring_settings']
            interval = settings.get('health_check_interval', 30)
            if interval < 1 or interval > 3600:  # 1 час максимум
                errors.append("Health check interval must be between 1 and 3600 seconds")
        
        return errors

    def reset_to_defaults(self) -> Dict[str, Any]:
        """Сброс к значениям по умолчанию"""
        try:
            self.config = self.defaults
            self._save_config_to_agent()
            
            # Применение изменений к активным компонентам
            self._apply_config_changes()
            
            return {
                'success': True,
                'default_config': self.get_config(),
                'message': 'Configuration reset to defaults'
            }
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to reset configuration'
            }

    def _apply_config_changes(self):
        """Применение изменений конфигурации к активным компонентам"""
        try:
            # Применение порогов когнитивной нагрузки
            if (hasattr(self.agent_core, 'cognitive_load_analyzer') and 
                self.agent_core.cognitive_load_analyzer):
                self.agent_core.cognitive_load_analyzer.thresholds.update({
                    'low_to_medium': self.config.cognitive_load_thresholds.low_to_medium,
                    'medium_to_high': self.config.cognitive_load_thresholds.medium_to_high,
                    'high_to_critical': self.config.cognitive_load_thresholds.high_to_critical
                })
            
            # Применение настроек адаптации
            if (hasattr(self.agent_core, 'adaptation_engine') and 
                self.agent_core.adaptation_engine):
                self.agent_core.adaptation_engine.settings.update({
                    'auto_adaptation_enabled': self.config.adaptation_settings.enable_auto_adaptation,
                    'adaptation_frequency': self.config.adaptation_settings.adaptation_frequency,
                    'minimum_effectiveness_threshold': self.config.adaptation_settings.minimum_effectiveness_threshold
                })
            
            # Применение параметров обучения
            if (hasattr(self.agent_core, 'learning_engine') and 
                self.agent_core.learning_engine):
                self.agent_core.learning_engine.parameters.update({
                    'learning_rate': self.config.learning_parameters.learning_rate,
                    'memory_decay_rate': self.config.learning_parameters.memory_decay_rate,
                    'experience_importance_threshold': self.config.learning_parameters.experience_importance_threshold
                })
            
            # Применение настроек саморефлексии
            if (hasattr(self.agent_core, 'reflection_engine') and 
                self.agent_core.reflection_engine):
                self.agent_core.reflection_engine.settings.update({
                    'reflection_frequency': self.config.self_reflection_settings.reflection_frequency,
                    'min_confidence_for_reflection': self.config.self_reflection_settings.min_confidence_for_reflection,
                    'enable_deep_reflection': self.config.self_reflection_settings.enable_deep_reflection
                })
            
            # Применение настроек мониторинга
            if (hasattr(self.agent_core, 'self_monitoring') and 
                self.agent_core.self_monitoring):
                self.agent_core.self_monitoring.settings.update({
                    'health_check_interval': self.config.monitoring_settings.health_check_interval,
                    'performance_monitoring_enabled': self.config.monitoring_settings.performance_monitoring_enabled,
                    'cognitive_load_monitoring_enabled': self.config.monitoring_settings.cognitive_load_monitoring_enabled
                })
            
            logger.info("Configuration changes applied to active components")
        except Exception as e:
            logger.error(f"Error applying configuration changes: {e}")

    def get_health_indicators(self) -> Dict[str, Any]:
        """Получение индикаторов здоровья конфигурации"""
        try:
            # Проверяем, все ли ключевые параметры установлены
            config_dict = self._config_to_dict(self.config)
            
            # Проверка настроек
            settings_health = {
                'cognitive_load_thresholds_valid': self._validate_thresholds(self.config.cognitive_load_thresholds),
                'adaptation_settings_valid': self._validate_adaptation_settings(self.config.adaptation_settings),
                'learning_parameters_valid': self._validate_learning_parameters(self.config.learning_parameters),
                'self_reflection_settings_valid': self._validate_self_reflection_settings(self.config.self_reflection_settings),
                'monitoring_settings_valid': self._validate_monitoring_settings(self.config.monitoring_settings)
            }
            
            # Оценка общего здоровья
            valid_settings = sum(1 for is_valid in settings_health.values() if is_valid)
            total_settings = len(settings_health)
            health_score = valid_settings / total_settings if total_settings > 0 else 0.0
            
            overall_health = 'healthy' if health_score >= 0.9 else 'degraded' if health_score >= 0.7 else 'poor'
            
            return {
                'status': overall_health,
                'health_score': health_score,
                'settings_health': settings_health,
                'configuration_complete': all(settings_health.values()),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting configuration health: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _validate_thresholds(self, thresholds: CognitiveLoadThresholds) -> bool:
        """Проверка валидности порогов"""
        return (0 <= thresholds.low_to_medium <= thresholds.medium_to_high <= 
                thresholds.high_to_critical <= 1.0)

    def _validate_adaptation_settings(self, settings: AdaptationSettings) -> bool:
        """Проверка валидности настроек адаптации"""
        return (1 <= settings.adaptation_frequency <= 3600 and 
                0 <= settings.minimum_effectiveness_threshold <= 1.0)

    def _validate_learning_parameters(self, params: LearningParameters) -> bool:
        """Проверка валидности параметров обучения"""
        return (0 < params.learning_rate <= 1.0 and 
                0 <= params.memory_decay_rate <= 0.1 and 
                0 <= params.experience_importance_threshold <= 1.0)

    def _validate_self_reflection_settings(self, settings: SelfReflectionSettings) -> bool:
        """Проверка валидности настроек саморефлексии"""
        return (1 <= settings.reflection_frequency <= 3600 and 
                0 <= settings.min_confidence_for_reflection <= 1.0)

    def _validate_monitoring_settings(self, settings: MonitoringSettings) -> bool:
        """Проверка валидности настроек мониторинга"""
        return 1 <= settings.health_check_interval <= 3600


# Глобальная функция для инициализации менеджера конфигурации
def initialize_meta_cognitive_config_manager(agent_core: 'AgentCore') -> MetaCognitiveConfigManager:
    """
    Инициализация менеджера мета-когнитивной конфигурации
    
    Args:
        agent_core: Ядро агента
        
    Returns:
        MetaCognitiveConfigManager: Менеджер конфигурации
    """
    config_manager = MetaCognitiveConfigManager(agent_core)
    logger.info("Meta-cognitive configuration manager initialized and attached to agent core")
    return config_manager