"""
Система алертинга для AI Агента
Обработка аномалий и отправка уведомлений через различные каналы
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from .metrics_collector import metrics_collector


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Уровни критичности алертов"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Статусы алертов"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class Alert:
    """Модель алерта"""
    id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    generator_url: str = ""
    fingerprint: str = ""
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "description": self.description,
            "labels": self.labels,
            "annotations": self.annotations,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "generator_url": self.generator_url,
            "fingerprint": self.fingerprint,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


class AlertChannel(Enum):
    """Каналы отправки алертов"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    CONSOLE = "console"


class AlertRule:
    """Правило алертинга"""
    
    def __init__(
        self,
        name: str,
        condition: Callable[[], bool],
        severity: AlertSeverity,
        for_duration: timedelta = timedelta(minutes=1),
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None
    ):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.for_duration = for_duration
        self.labels = labels or {}
        self.annotations = annotations or {}
        self.firing_since: Optional[datetime] = None
        self.last_evaluation = datetime.now()
    
    def evaluate(self) -> bool:
        """Оценка условия алерта"""
        self.last_evaluation = datetime.now()
        condition_result = self.condition()
        
        if condition_result:
            if self.firing_since is None:
                self.firing_since = self.last_evaluation
            elif self.last_evaluation - self.firing_since >= self.for_duration:
                return True
        else:
            self.firing_since = None
        
        return False


class AlertNotifier:
    """Система отправки уведомлений"""
    
    def __init__(self):
        self.email_config = {
            "smtp_server": "localhost",
            "smtp_port": 587,
            "username": "",
            "password": "",
            "from_email": "alerts@ai-agent.local"
        }
        self.slack_webhook_url = ""
        self.pagerduty_integration_key = ""
        self.webhook_url = ""
        
        # Загрузка конфигурации из переменных окружения
        self._load_config_from_env()
    
    def _load_config_from_env(self):
        """Загрузка конфигурации из переменных окружения"""
        import os
        
        self.email_config["smtp_server"] = os.getenv("EMAIL_SMTP_SERVER", self.email_config["smtp_server"])
        self.email_config["smtp_port"] = int(os.getenv("EMAIL_SMTP_PORT", str(self.email_config["smtp_port"])))
        self.email_config["username"] = os.getenv("EMAIL_USERNAME", self.email_config["username"])
        self.email_config["password"] = os.getenv("EMAIL_PASSWORD", self.email_config["password"])
        self.email_config["from_email"] = os.getenv("EMAIL_FROM", self.email_config["from_email"])
        
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", self.slack_webhook_url)
        self.pagerduty_integration_key = os.getenv("PAGERDUTY_INTEGRATION_KEY", self.pagerduty_integration_key)
        self.webhook_url = os.getenv("ALERT_WEBHOOK_URL", self.webhook_url)
    
    async def send_alert(
        self,
        alert: Alert,
        channels: List[AlertChannel] = None
    ) -> Dict[AlertChannel, bool]:
        """Отправка алерта по указанным каналам"""
        if channels is None:
            channels = [AlertChannel.CONSOLE]
        
        results = {}
        
        for channel in channels:
            try:
                if channel == AlertChannel.EMAIL:
                    results[channel] = await self._send_email(alert)
                elif channel == AlertChannel.SLACK:
                    results[channel] = await self._send_slack(alert)
                elif channel == AlertChannel.WEBHOOK:
                    results[channel] = await self._send_webhook(alert)
                elif channel == AlertChannel.PAGERDUTY:
                    results[channel] = await self._send_pagerduty(alert)
                elif channel == AlertChannel.CONSOLE:
                    results[channel] = await self._send_console(alert)
                else:
                    logger.warning(f"Unknown alert channel: {channel}")
                    results[channel] = False
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
                results[channel] = False
        
        return results
    
    async def _send_email(self, alert: Alert) -> bool:
        """Отправка алерта по email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config["from_email"]
            msg['To'] = ', '.join(alert.labels.get('recipients', ['admin@ai-agent.local']))
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.name}"
            
            body = f"""
            Alert: {alert.name}
            Severity: {alert.severity.value}
            Status: {alert.status.value}
            Message: {alert.message}
            Description: {alert.description}
            Time: {alert.start_time.isoformat()}
            
            Labels: {json.dumps(alert.labels, indent=2)}
            Annotations: {json.dumps(alert.annotations, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"])
            server.starttls()
            if self.email_config["username"] and self.email_config["password"]:
                server.login(self.email_config["username"], self.email_config["password"])
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    async def _send_slack(self, alert: Alert) -> bool:
        """Отправка алерта в Slack"""
        if not self.slack_webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False
        
        try:
            color = {
                AlertSeverity.LOW: "#66D3E4",
                AlertSeverity.MEDIUM: "#FFD93D",
                AlertSeverity.HIGH: "#FF9F43",
                AlertSeverity.CRITICAL: "#FF6B6B"
            }.get(alert.severity, "#66D3E4")
            
            payload = {
                "text": f"Alert: {alert.name}",
                "attachments": [
                    {
                        "color": color,
                        "fields": [
                            {
                                "title": "Alert",
                                "value": alert.name,
                                "short": True
                            },
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Status",
                                "value": alert.status.value,
                                "short": True
                            },
                            {
                                "title": "Message",
                                "value": alert.message,
                                "short": False
                            },
                            {
                                "title": "Description",
                                "value": alert.description,
                                "short": False
                            },
                            {
                                "title": "Time",
                                "value": alert.start_time.isoformat(),
                                "short": True
                            }
                        ],
                        "footer": "AI Agent Monitoring",
                        "ts": int(alert.start_time.timestamp())
                    }
                ]
            }
            
            response = requests.post(
                self.slack_webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {alert.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    async def _send_webhook(self, alert: Alert) -> bool:
        """Отправка алерта через вебхук"""
        if not self.webhook_url:
            logger.warning("Webhook URL not configured")
            return False
        
        try:
            payload = alert.to_dict()
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent: {alert.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    async def _send_pagerduty(self, alert: Alert) -> bool:
        """Отправка алерта в PagerDuty"""
        if not self.pagerduty_integration_key:
            logger.warning("PagerDuty integration key not configured")
            return False
        
        try:
            payload = {
                "routing_key": self.pagerduty_integration_key,
                "event_action": "trigger" if alert.status == AlertStatus.FIRING else "resolve",
                "payload": {
                    "summary": f"{alert.name}: {alert.message}",
                    "source": "ai-agent",
                    "severity": "error" if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL] else "warning",
                    "custom_details": {
                        "description": alert.description,
                        "labels": alert.labels,
                        "annotations": alert.annotations
                    }
                }
            }
            
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"PagerDuty alert sent: {alert.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False
    
    async def _send_console(self, alert: Alert) -> bool:
        """Отправка алерта в консоль"""
        logger.critical(f"ALERT [{alert.severity.value.upper()}]: {alert.name} - {alert.message}")
        logger.info(f"Description: {alert.description}")
        logger.info(f"Labels: {alert.labels}")
        logger.info(f"Annotations: {alert.annotations}")
        return True


class AlertManager:
    """Менеджер алертов"""
    
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.notifier = AlertNotifier()
        self.evaluation_interval = 30  # seconds
        self.default_channels = [AlertChannel.CONSOLE]
        
        logger.info("Alert manager initialized")
    
    def add_rule(self, rule: AlertRule):
        """Добавление правила алертинга"""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Удаление правила алертинга"""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        logger.info(f"Removed alert rule: {rule_name}")
    
    async def evaluate_rules(self):
        """Оценка всех правил алертинга"""
        for rule in self.rules:
            try:
                should_fire = rule.evaluate()
                
                alert_id = f"alert_{rule.name}_{hash(str(rule.labels))}"
                
                if should_fire:
                    if alert_id not in self.active_alerts:
                        # Создаем новый алерт
                        alert = Alert(
                            id=alert_id,
                            name=rule.name,
                            severity=rule.severity,
                            status=AlertStatus.FIRING,
                            message=f"Rule '{rule.name}' is firing",
                            description=rule.annotations.get("description", f"Condition in rule '{rule.name}' is met"),
                            labels=rule.labels,
                            annotations=rule.annotations,
                            generator_url=f"http://ai-agent:8000/rules/{rule.name}",
                            fingerprint=alert_id
                        )
                        
                        self.active_alerts[alert_id] = alert
                        logger.info(f"New alert fired: {alert.name}")
                        
                        # Отправляем алерт
                        await self._notify_alert(alert)
                    else:
                        # Алерт уже активен, обновляем время
                        self.active_alerts[alert_id].end_time = datetime.now()
                else:
                    # Правило больше не срабатывает, проверяем, нужно ли закрыть алерт
                    if alert_id in self.active_alerts and self.active_alerts[alert_id].status == AlertStatus.FIRING:
                        # Проверяем, был ли алерт активен дольше минимального времени
                        if rule.firing_since and datetime.now() - rule.firing_since >= rule.for_duration:
                            # Закрываем алерт
                            alert = self.active_alerts[alert_id]
                            alert.status = AlertStatus.RESOLVED
                            alert.resolved_at = datetime.now()
                            logger.info(f"Alert resolved: {alert.name}")
                            
                            # Отправляем уведомление о закрытии алерта
                            await self._notify_alert(alert)
            
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    async def _notify_alert(self, alert: Alert):
        """Отправка уведомления об алерте"""
        # Определяем каналы отправки на основе меток алерта
        channels_str = alert.labels.get('channels', 'console').split(',')
        channels = []
        
        for channel_str in channels_str:
            try:
                channel = AlertChannel(channel_str.strip().upper())
                channels.append(channel)
            except ValueError:
                logger.warning(f"Invalid alert channel: {channel_str}")
        
        if not channels:
            channels = self.default_channels
        
        # Отправляем алерт
        results = await self.notifier.send_alert(alert, channels)
        
        # Логируем результаты
        successful_channels = [ch for ch, success in results.items() if success]
        failed_channels = [ch for ch, success in results.items() if not success]
        
        if successful_channels:
            logger.info(f"Alert {alert.name} sent successfully via: {successful_channels}")
        
        if failed_channels:
            logger.error(f"Alert {alert.name} failed to send via: {failed_channels}")
        
        # Обновляем метрики
        if metrics_collector.enabled:
            metrics_collector.set_health_status(f"alert_{alert.severity.value}", 
                                              0.0 if alert.status == AlertStatus.FIRING else 1.0)
    
    async def start_alert_evaluation(self):
        """Запуск периодической оценки алертов"""
        logger.info(f"Starting alert evaluation loop with interval {self.evaluation_interval}s")
        
        while True:
            try:
                await self.evaluate_rules()
                await asyncio.sleep(self.evaluation_interval)
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(self.evaluation_interval)
    
    def get_active_alerts(self) -> List[Alert]:
        """Получение списка активных алертов"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Получение истории алертов (в реальной системе это хранилось бы в БД)"""
        # В этой реализации возвращаем только активные алерты
        # В реальной системе архив алертов хранился бы в БД
        return list(self.active_alerts.values())[-limit:]


# Примеры правил алертинга для AI Агента
def create_cpu_usage_alert_rule(threshold: float = 90.0) -> AlertRule:
    """Создание правила алерта для CPU usage"""
    def condition():
        # В реальной системе это проверяло бы метрики из Prometheus
        # Здесь используем psutil для демонстрации
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        return cpu_percent > threshold
    
    return AlertRule(
        name="high_cpu_usage",
        condition=condition,
        severity=AlertSeverity.HIGH,
        for_duration=timedelta(minutes=2),
        labels={"team": "ai", "service": "ai-agent", "severity": "high"},
        annotations={
            "summary": "High CPU usage detected",
            "description": f"CPU usage is above {threshold}% for more than 2 minutes"
        }
    )


def create_memory_usage_alert_rule(threshold: float = 90.0) -> AlertRule:
    """Создание правила алерта для Memory usage"""
    def condition():
        import psutil
        memory_percent = psutil.virtual_memory().percent
        return memory_percent > threshold
    
    return AlertRule(
        name="high_memory_usage",
        condition=condition,
        severity=AlertSeverity.HIGH,
        for_duration=timedelta(minutes=2),
        labels={"team": "ai", "service": "ai-agent", "severity": "high"},
        annotations={
            "summary": "High memory usage detected",
            "description": f"Memory usage is above {threshold}% for more than 2 minutes"
        }
    )


def create_cognitive_load_alert_rule(threshold: float = 0.9) -> AlertRule:
    """Создание правила алерта для когнитивной нагрузки"""
    def condition():
        # В реальной системе это проверяло бы метрики когнитивной нагрузки из Prometheus
        # Здесь возвращаем False для демонстрации
        return False  # Пока что возвращаем False, так как нет реальных метрик
    
    return AlertRule(
        name="high_cognitive_load",
        condition=condition,
        severity=AlertSeverity.MEDIUM,
        for_duration=timedelta(minutes=5),
        labels={"team": "ai", "service": "ai-agent", "severity": "medium"},
        annotations={
            "summary": "High cognitive load detected",
            "description": f"Cognitive load is above {threshold} for more than 5 minutes"
        }
    )


def create_error_rate_alert_rule(threshold: float = 0.05) -> AlertRule:
    """Создание правила алерта для ошибок"""
    def condition():
        # В реальной системе это проверяло бы метрики ошибок из Prometheus
        # Здесь возвращаем False для демонстрации
        return False  # Пока что возвращаем False, так как нет реальных метрик
    
    return AlertRule(
        name="high_error_rate",
        condition=condition,
        severity=AlertSeverity.HIGH,
        for_duration=timedelta(minutes=1),
        labels={"team": "ai", "service": "ai-agent", "severity": "high"},
        annotations={
            "summary": "High error rate detected",
            "description": f"Error rate is above {threshold} for more than 1 minute"
        }
    )


# Глобальный менеджер алертов
alert_manager = AlertManager()


def setup_default_alert_rules():
    """Настройка стандартных правил алертинга"""
    # Добавляем стандартные правила
    alert_manager.add_rule(create_cpu_usage_alert_rule(85.0))
    alert_manager.add_rule(create_memory_usage_alert_rule(85.0))
    alert_manager.add_rule(create_cognitive_load_alert_rule(0.85))
    alert_manager.add_rule(create_error_rate_alert_rule(0.05))
    
    logger.info("Default alert rules configured")


if __name__ == "__main__":
    # Пример использования
    import asyncio
    
    async def main():
        # Настройка правил
        setup_default_alert_rules()
        
        # Запуск оценки алертов (в демонстрационных целях запускаем один раз)
        await alert_manager.evaluate_rules()
        
        # Показываем активные алерты
        active_alerts = alert_manager.get_active_alerts()
        print(f"Active alerts: {len(active_alerts)}")
        for alert in active_alerts:
            print(f"  - {alert.name}: {alert.status.value} ({alert.severity.value})")
    
    # asyncio.run(main())