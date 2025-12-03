"""
Генератор Grafana dashboards для мониторинга AI Агента
Создает JSON конфигурации для dashboard с метриками системы
"""

import json
from typing import Dict, Any, List
from datetime import datetime

def create_system_monitoring_dashboard() -> Dict[str, Any]:
    """
    Создание dashboard для системного мониторинга

    Returns:
        Dict с конфигурацией Grafana dashboard
    """
    dashboard = {
        "dashboard": {
            "id": None,
            "title": "AI Agent System Monitoring",
            "tags": ["ai-agent", "system", "monitoring"],
            "timezone": "browser",
            "panels": [],
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "timepicker": {},
            "templating": {
                "list": []
            },
            "annotations": {
                "list": []
            },
            "refresh": "30s",
            "schemaVersion": 27,
            "version": 0,
            "links": []
        }
    }

    panels = []

    # Panel 1: CPU Usage
    panels.append({
        "id": 1,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [{
            "expr": "100 - (avg by(instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU Usage %",
            "refId": "A"
        }],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "yaxes": [{"format": "percent", "label": "CPU %"}]
    })

    # Panel 2: Memory Usage
    panels.append({
        "id": 2,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [{
            "expr": "(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100",
            "legendFormat": "Memory Usage %",
            "refId": "A"
        }],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "yaxes": [{"format": "percent", "label": "Memory %"}]
    })

    # Panel 3: Disk Usage
    panels.append({
        "id": 3,
        "title": "Disk Usage",
        "type": "graph",
        "targets": [{
            "expr": "(1 - node_filesystem_avail_bytes{mountpoint=\"/\"} / node_filesystem_size_bytes{mountpoint=\"/\"}) * 100",
            "legendFormat": "Disk Usage %",
            "refId": "A"
        }],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "yaxes": [{"format": "percent", "label": "Disk %"}]
    })

    # Panel 4: Network Traffic
    panels.append({
        "id": 4,
        "title": "Network Traffic",
        "type": "graph",
        "targets": [
            {
                "expr": "rate(node_network_receive_bytes_total[5m])",
                "legendFormat": "Receive",
                "refId": "A"
            },
            {
                "expr": "rate(node_network_transmit_bytes_total[5m])",
                "legendFormat": "Transmit",
                "refId": "B"
            }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "yaxes": [{"format": "Bps", "label": "Bytes/sec"}]
    })

    dashboard["dashboard"]["panels"] = panels
    return dashboard

def create_application_monitoring_dashboard() -> Dict[str, Any]:
    """
    Создание dashboard для мониторинга приложения

    Returns:
        Dict с конфигурацией Grafana dashboard
    """
    dashboard = {
        "dashboard": {
            "id": None,
            "title": "AI Agent Application Monitoring",
            "tags": ["ai-agent", "application", "monitoring"],
            "timezone": "browser",
            "panels": [],
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "timepicker": {},
            "templating": {
                "list": []
            },
            "annotations": {
                "list": []
            },
            "refresh": "30s",
            "schemaVersion": 27,
            "version": 0,
            "links": []
        }
    }

    panels = []

    # Panel 1: HTTP Request Rate
    panels.append({
        "id": 1,
        "title": "HTTP Request Rate",
        "type": "graph",
        "targets": [{
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "Requests/sec",
            "refId": "A"
        }],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "yaxes": [{"format": "reqps", "label": "Requests/sec"}]
    })

    # Panel 2: HTTP Response Time
    panels.append({
        "id": 2,
        "title": "HTTP Response Time",
        "type": "graph",
        "targets": [{
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95 Response Time",
            "refId": "A"
        }],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "yaxes": [{"format": "s", "label": "Seconds"}]
    })

    # Panel 3: Error Rate
    panels.append({
        "id": 3,
        "title": "HTTP Error Rate",
        "type": "graph",
        "targets": [{
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100",
            "legendFormat": "5xx Error Rate %",
            "refId": "A"
        }],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "yaxes": [{"format": "percent", "label": "Error %"}]
    })

    # Panel 4: Active Connections
    panels.append({
        "id": 4,
        "title": "Active Connections",
        "type": "graph",
        "targets": [{
            "expr": "http_connections_active",
            "legendFormat": "Active Connections",
            "refId": "A"
        }],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "yaxes": [{"format": "short", "label": "Connections"}]
    })

    dashboard["dashboard"]["panels"] = panels
    return dashboard

def create_ai_agent_monitoring_dashboard() -> Dict[str, Any]:
    """
    Создание dashboard для мониторинга AI Агента

    Returns:
        Dict с конфигурацией Grafana dashboard
    """
    dashboard = {
        "dashboard": {
            "id": None,
            "title": "AI Agent Performance Monitoring",
            "tags": ["ai-agent", "performance", "monitoring"],
            "timezone": "browser",
            "panels": [],
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "timepicker": {},
            "templating": {
                "list": []
            },
            "annotations": {
                "list": []
            },
            "refresh": "30s",
            "schemaVersion": 27,
            "version": 0,
            "links": []
        }
    }

    panels = []

    # Panel 1: Agent State Transitions
    panels.append({
        "id": 1,
        "title": "Agent State Transitions",
        "type": "stat",
        "targets": [{
            "expr": "increase(agent_state_transitions_total[1h])",
            "legendFormat": "State Changes",
            "refId": "A"
        }],
        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 0},
        "fieldConfig": {
            "defaults": {
                "mappings": [],
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "red", "value": 80}
                    ]
                }
            }
        }
    })

    # Panel 2: Current Agent State
    panels.append({
        "id": 2,
        "title": "Current Agent State",
        "type": "stat",
        "targets": [{
            "expr": "agent_state_current",
            "legendFormat": "Current State",
            "refId": "A"
        }],
        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 0},
        "fieldConfig": {
            "defaults": {
                "mappings": [
                    {"options": {"IDLE": {"text": "IDLE", "color": "green"}},
                     "options": {"ANALYZING": {"text": "ANALYZING", "color": "blue"}},
                     "options": {"EXECUTING": {"text": "EXECUTING", "color": "orange"}},
                     "options": {"ERROR": {"text": "ERROR", "color": "red"}}}
                ]
            }
        }
    })

    # Panel 3: AI Inference Latency
    panels.append({
        "id": 3,
        "title": "AI Inference Latency",
        "type": "graph",
        "targets": [{
            "expr": "histogram_quantile(0.95, rate(ai_inference_duration_seconds_bucket[5m]))",
            "legendFormat": "P95 Inference Time",
            "refId": "A"
        }],
        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 0},
        "yaxes": [{"format": "s", "label": "Seconds"}]
    })

    # Panel 4: Cache Hit Rate
    panels.append({
        "id": 4,
        "title": "Cache Performance",
        "type": "graph",
        "targets": [
            {
                "expr": "rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m])) * 100",
                "legendFormat": "Cache Hit Rate %",
                "refId": "A"
            }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "yaxes": [{"format": "percent", "label": "Hit Rate %"}]
    })

    # Panel 5: Database Query Performance
    panels.append({
        "id": 5,
        "title": "Database Query Performance",
        "type": "graph",
        "targets": [{
            "expr": "histogram_quantile(0.95, rate(db_query_duration_seconds_bucket[5m]))",
            "legendFormat": "P95 Query Time",
            "refId": "A"
        }],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "yaxes": [{"format": "s", "label": "Seconds"}]
    })

    dashboard["dashboard"]["panels"] = panels
    return dashboard

def create_health_checks_dashboard() -> Dict[str, Any]:
    """
    Создание dashboard для health checks

    Returns:
        Dict с конфигурацией Grafana dashboard
    """
    dashboard = {
        "dashboard": {
            "id": None,
            "title": "AI Agent Health Checks",
            "tags": ["ai-agent", "health", "monitoring"],
            "timezone": "browser",
            "panels": [],
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "timepicker": {},
            "templating": {
                "list": []
            },
            "annotations": {
                "list": []
            },
            "refresh": "30s",
            "schemaVersion": 27,
            "version": 0,
            "links": []
        }
    }

    panels = []

    # Panel 1: Overall Health Status
    panels.append({
        "id": 1,
        "title": "Overall System Health",
        "type": "stat",
        "targets": [{
            "expr": "health_check_status{check=\"overall\"}",
            "legendFormat": "System Health",
            "refId": "A"
        }],
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0},
        "fieldConfig": {
            "defaults": {
                "mappings": [
                    {"options": {"1": {"text": "HEALTHY", "color": "green"}},
                     "options": {"0.5": {"text": "DEGRADED", "color": "orange"}},
                     "options": {"0": {"text": "UNHEALTHY", "color": "red"}}}
                ]
            }
        }
    })

    # Panel 2: Health Check Status Table
    panels.append({
        "id": 2,
        "title": "Health Check Details",
        "type": "table",
        "targets": [{
            "expr": "health_check_status",
            "legendFormat": "{{check}}",
            "refId": "A",
            "format": "table"
        }],
        "gridPos": {"h": 12, "w": 18, "x": 6, "y": 0},
        "fieldConfig": {
            "defaults": {
                "custom": {
                    "align": "auto",
                    "displayMode": "auto"
                }
            },
            "overrides": [
                {
                    "matcher": {"id": "byName", "options": "Status"},
                    "properties": [
                        {
                            "id": "mappings",
                            "value": [
                                {"options": {"1": {"text": "HEALTHY", "color": "green"}},
                                 "options": {"0.5": {"text": "DEGRADED", "color": "orange"}},
                                 "options": {"0": {"text": "UNHEALTHY", "color": "red"}}}
                            ]
                        }
                    ]
                }
            ]
        }
    })

    # Panel 3: Health Check Response Times
    panels.append({
        "id": 3,
        "title": "Health Check Response Times",
        "type": "graph",
        "targets": [{
            "expr": "health_check_duration_seconds",
            "legendFormat": "{{check}}",
            "refId": "A"
        }],
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 12},
        "yaxes": [{"format": "s", "label": "Response Time"}]
    })

    dashboard["dashboard"]["panels"] = panels
    return dashboard

def save_dashboard_to_file(dashboard: Dict[str, Any], filename: str):
    """
    Сохранение dashboard в JSON файл

    Args:
        dashboard: Конфигурация dashboard
        filename: Имя файла
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dashboard, f, indent=2, ensure_ascii=False)

    print(f"Dashboard saved to {filename}")

def generate_all_dashboards():
    """Генерация всех dashboard файлов"""
    dashboards = [
        ("system_monitoring_dashboard.json", create_system_monitoring_dashboard()),
        ("application_monitoring_dashboard.json", create_application_monitoring_dashboard()),
        ("ai_agent_monitoring_dashboard.json", create_ai_agent_monitoring_dashboard()),
        ("health_checks_dashboard.json", create_health_checks_dashboard())
    ]

    # Создаем директорию для dashboard
    import os
    os.makedirs("monitoring/dashboards", exist_ok=True)

    for filename, dashboard in dashboards:
        filepath = f"monitoring/dashboards/{filename}"
        save_dashboard_to_file(dashboard, filepath)

    print("All dashboards generated successfully!")
    print("Import these JSON files into Grafana to create the dashboards.")

if __name__ == "__main__":
    generate_all_dashboards()
