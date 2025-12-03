"""
Конфигурация Prometheus для AI Агента
Генерация конфигурационного файла для сбора метрик
"""

import os
from typing import Dict, Any, List
import yaml
from pathlib import Path


def create_prometheus_config(
    ai_agent_targets: List[str] = None,
    additional_targets: List[Dict[str, Any]] = None,
    alerting_rules: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Создание конфигурации Prometheus
    
    Args:
        ai_agent_targets: Список целей для AI агента
        additional_targets: Дополнительные цели для мониторинга
        alerting_rules: Правила алертинга
    
    Returns:
        Dict с конфигурацией Prometheus
    """
    if ai_agent_targets is None:
        ai_agent_targets = ["ai-agent:8000"]
    
    if additional_targets is None:
        additional_targets = []
    
    if alerting_rules is None:
        alerting_rules = []
    
    config = {
        'global': {
            'scrape_interval': '15s',
            'evaluation_interval': '15s',
            'external_labels': {
                'monitor': 'ai-agent-monitor'
            }
        },
        'rule_files': [
            'alert_rules.yml'
        ],
        'scrape_configs': [
            {
                'job_name': 'prometheus',
                'static_configs': [
                    {
                        'targets': ['localhost:9090']
                    }
                ]
            },
            {
                'job_name': 'ai-agent',
                'scrape_interval': '5s',  # Частое сканирование для высоконагруженного сервиса
                'scrape_timeout': '10s',
                'metrics_path': '/metrics',
                'static_configs': [
                    {
                        'targets': ai_agent_targets,
                        'labels': {
                            'service': 'ai-agent',
                            'team': 'ai'
                        }
                    }
                ],
                'relabel_configs': [
                    {
                        'source_labels': ['__address__'],
                        'target_label': '__tmp_prometheus_agent_address'
                    }
                ]
            },
            {
                'job_name': 'node-exporter',
                'scrape_interval': '5s',
                'static_configs': [
                    {
                        'targets': ['node-exporter:9100'],
                        'labels': {
                            'service': 'node-exporter',
                            'team': 'infrastructure'
                        }
                    }
                ]
            },
            {
                'job_name': 'jaeger',
                'scrape_interval': '10s',
                'static_configs': [
                    {
                        'targets': ['jaeger:14269'],
                        'labels': {
                            'service': 'jaeger',
                            'team': 'tracing'
                        }
                    }
                ]
            }
        ]
    }
    
    # Добавляем дополнительные цели
    config['scrape_configs'].extend(additional_targets)
    
    return config


def create_alert_rules() -> Dict[str, Any]:
    """
    Создание правил алертинга для AI Агента
    
    Returns:
        Dict с правилами алертинга
    """
    rules = {
        'groups': [
            {
                'name': 'ai-agent.rules',
                'rules': [
                    # Правила для проверки здоровья сервиса
                    {
                        'alert': 'AI_Agent_Down',
                        'expr': 'up{job="ai-agent"} == 0',
                        'for': '5m',
                        'labels': {
                            'severity': 'critical'
                        },
                        'annotations': {
                            'summary': 'AI Agent is down',
                            'description': 'AI Agent has been down for more than 5 minutes'
                        }
                    },
                    # Правила для метрик производительности
                    {
                        'alert': 'AI_Agent_High_HTTP_Errors',
                        'expr': 'rate(http_requests_total{job="ai-agent", status=~"5.."}[5m]) / rate(http_requests_total{job="ai-agent"}[5m]) > 0.05',
                        'for': '2m',
                        'labels': {
                            'severity': 'warning'
                        },
                        'annotations': {
                            'summary': 'High error rate on AI Agent',
                            'description': 'AI Agent is experiencing high error rate (> 5%)'
                        }
                    },
                    {
                        'alert': 'AI_Agent_High_Response_Time',
                        'expr': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="ai-agent"}[5m])) > 5',
                        'for': '2m',
                        'labels': {
                            'severity': 'warning'
                        },
                        'annotations': {
                            'summary': 'High response time on AI Agent',
                            'description': '95th percentile response time is above 5 seconds'
                        }
                    },
                    # Правила для когнитивной нагрузки
                    {
                        'alert': 'AI_Agent_High_Cognitive_Load',
                        'expr': 'cognitive_load_current{job="ai-agent"} > 0.9',
                        'for': '5m',
                        'labels': {
                            'severity': 'warning'
                        },
                        'annotations': {
                            'summary': 'High cognitive load on AI Agent',
                            'description': 'AI Agent cognitive load is above 90%'
                        }
                    },
                    # Правила для системных метрик
                    {
                        'alert': 'AI_Agent_High_CPU_Usage',
                        'expr': 'system_cpu_percent{job="ai-agent"} > 90',
                        'for': '5m',
                        'labels': {
                            'severity': 'warning'
                        },
                        'annotations': {
                            'summary': 'High CPU usage on AI Agent',
                            'description': 'AI Agent CPU usage is above 90%'
                        }
                    },
                    {
                        'alert': 'AI_Agent_High_Memory_Usage',
                        'expr': 'system_memory_percent{job="ai-agent"} > 90',
                        'for': '5m',
                        'labels': {
                            'severity': 'warning'
                        },
                        'annotations': {
                            'summary': 'High memory usage on AI Agent',
                            'description': 'AI Agent memory usage is above 90%'
                        }
                    },
                    # Правила для агента
                    {
                        'alert': 'AI_Agent_Low_Success_Rate',
                        'expr': 'learning_success_rate{job="ai-agent"} < 0.5',
                        'for': '10m',
                        'labels': {
                            'severity': 'warning'
                        },
                        'annotations': {
                            'summary': 'Low learning success rate',
                            'description': 'AI Agent learning success rate is below 50%'
                        }
                    },
                    {
                        'alert': 'AI_Agent_High_Tool_Error_Rate',
                        'expr': 'rate(tool_calls_total{status="error"}[5m]) / rate(tool_calls_total[5m]) > 0.1',
                        'for': '2m',
                        'labels': {
                            'severity': 'warning'
                        },
                        'annotations': {
                            'summary': 'High tool error rate',
                            'description': 'AI Agent tool error rate is above 10%'
                        }
                    }
                ]
            }
        ]
    }
    
    return rules


def create_alertmanager_config() -> Dict[str, Any]:
    """
    Создание конфигурации Alertmanager
    
    Returns:
        Dict с конфигурацией Alertmanager
    """
    config = {
        'global': {
            'smtp_smarthost': 'localhost:25',
            'smtp_from': 'alertmanager@ai-agent.local',
            'smtp_require_tls': False
        },
        'route': {
            'group_by': ['alertname', 'service'],
            'group_wait': '30s',
            'group_interval': '5m',
            'repeat_interval': '12h',
            'receiver': 'default-receiver',
            'routes': [
                {
                    'matchers': [
                        'severity="critical"'
                    ],
                    'receiver': 'critical-receiver',
                    'group_interval': '5m',
                    'repeat_interval': '1h'
                },
                {
                    'matchers': [
                        'severity="warning"'
                    ],
                    'receiver': 'warning-receiver',
                    'group_interval': '10m',
                    'repeat_interval': '3h'
                }
            ]
        },
        'receivers': [
            {
                'name': 'default-receiver',
                'email_configs': [
                    {
                        'to': 'admin@ai-agent.local',
                        'send_resolved': True
                    }
                ],
                'webhook_configs': [
                    {
                        'url': 'http://notification-service:8080/alert',
                        'send_resolved': True
                    }
                ]
            },
            {
                'name': 'critical-receiver',
                'email_configs': [
                    {
                        'to': 'admin@ai-agent.local,manager@ai-agent.local',
                        'send_resolved': True
                    }
                ],
                'webhook_configs': [
                    {
                        'url': 'http://notification-service:8080/critical-alert',
                        'send_resolved': True
                    }
                ],
                'pagerduty_configs': [
                    {
                        'routing_key': 'pagerduty-routing-key',
                        'send_resolved': True
                    }
                ]
            },
            {
                'name': 'warning-receiver',
                'email_configs': [
                    {
                        'to': 'admin@ai-agent.local',
                        'send_resolved': True
                    }
                ],
                'webhook_configs': [
                    {
                        'url': 'http://notification-service:8080/warning-alert',
                        'send_resolved': True
                    }
                ]
            }
        ],
        'inhibit_rules': [
            {
                'source_matchers': [
                    'severity="critical"'
                ],
                'target_matchers': [
                    'severity="warning"'
                ],
                'equal': ['alertname', 'service']
            }
        ]
    }
    
    return config


def create_loki_config() -> Dict[str, Any]:
    """
    Создание конфигурации Loki для агрегации логов
    
    Returns:
        Dict с конфигурацией Loki
    """
    config = {
        'auth_enabled': False,
        'server': {
            'http_listen_port': 3100
        },
        'common': {
            'path_prefix': '/tmp/loki',
            'storage': {
                'filesystem': {
                    'chunks_directory': '/tmp/loki/chunks',
                    'rules_directory': '/tmp/loki/rules'
                }
            },
            'replication_factor': 1
        },
        'schema_config': {
            'configs': [
                {
                    'from': '2020-10-24',
                    'store': 'boltdb-shipper',
                    'object_store': 'filesystem',
                    'schema': 'v11',
                    'index': {
                        'prefix': 'index_',
                        'period': '24h'
                    }
                }
            ]
        },
        'ruler': {
            'alertmanager_url': 'http://alertmanager:9093'
        }
    }
    
    return config


def create_promtail_config() -> Dict[str, Any]:
    """
    Создание конфигурации Promtail для отправки логов в Loki
    
    Returns:
        Dict с конфигурацией Promtail
    """
    config = {
        'server': {
            'http_listen_port': 9080,
            'grpc_listen_port': 0
        },
        'positions': {
            'filename': '/tmp/positions.yaml'
        },
        'clients': [
            {
                'url': 'http://loki:3100/loki/api/v1/push'
            }
        ],
        'scrape_configs': [
            {
                'job_name': 'ai-agent-logs',
                'static_configs': [
                    {
                        'targets': ['localhost'],
                        'labels': {
                            'job': 'ai-agent',
                            'host': 'ai-agent-container',
                            '__path__': '/home/app/logs/*.log'
                        }
                    }
                ]
            },
            {
                'job_name': 'system-logs',
                'static_configs': [
                    {
                        'targets': ['localhost'],
                        'labels': {
                            'job': 'system',
                            'host': 'ai-agent-container',
                            '__path__': '/var/log/*.log'
                        }
                    }
                ]
            }
        ]
    }
    
    return config


def save_config_to_file(config: Dict[str, Any], filename: str):
    """
    Сохранение конфигурации в YAML файл
    
    Args:
        config: Конфигурация
        filename: Имя файла
    """
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Configuration saved to {filename}")


def generate_all_configs():
    """Генерация всех конфигурационных файлов"""
    # Создаем директорию для конфигураций
    config_dir = Path("monitoring")
    config_dir.mkdir(exist_ok=True)
    
    # Генерируем конфигурации
    prometheus_config = create_prometheus_config(
        ai_agent_targets=["localhost:8000"]  # Предполагаем, что AI агент запущен на 800 порту
    )
    
    alert_rules = create_alert_rules()
    alertmanager_config = create_alertmanager_config()
    loki_config = create_loki_config()
    promtail_config = create_promtail_config()
    
    # Сохраняем конфигурации
    save_config_to_file(prometheus_config, "monitoring/prometheus.yml")
    save_config_to_file(alert_rules, "monitoring/alert_rules.yml")
    save_config_to_file(alertmanager_config, "monitoring/alertmanager.yml")
    save_config_to_file(loki_config, "monitoring/loki-config.yml")
    save_config_to_file(promtail_config, "monitoring/promtail-config.yml")
    
    print("All configuration files generated successfully!")
    print("Configuration files created:")
    print("- monitoring/prometheus.yml")
    print("- monitoring/alert_rules.yml")
    print("- monitoring/alertmanager.yml")
    print("- monitoring/loki-config.yml")
    print("- monitoring/promtail-config.yml")


if __name__ == "__main__":
    generate_all_configs()