"""
Pipeline Monitoring and Alerting System
Monitors CI/CD pipeline health and sends alerts for failures
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class PipelineMonitor:
    """
    Monitors CI/CD pipeline health and sends alerts for failures
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.email_host = os.getenv("EMAIL_HOST", "localhost")
        self.email_port = int(os.getenv("EMAIL_PORT", "587"))
        self.email_username = os.getenv("EMAIL_USERNAME")
        self.email_password = os.getenv("EMAIL_PASSWORD")
        self.alert_recipients = os.getenv("ALERT_RECIPIENTS", "").split(",")

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for monitoring"""
        logger = logging.getLogger("pipeline_monitor")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def send_slack_alert(self, message: str, job_name: str = "", status: str = "failed"):
        """Send alert to Slack"""
        if not self.slack_webhook_url:
            self.logger.warning("Slack webhook URL not configured")
            return

        color = "#FF0000" if status == "failed" else "#36A64F"
        
        payload = {
            "text": f"Pipeline Alert: {message}",
            "attachments": [
                {
                    "color": color,
                    "fields": [
                        {
                            "title": "Job",
                            "value": job_name or "Unknown",
                            "short": True
                        },
                        {
                            "title": "Status",
                            "value": status,
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": datetime.now().isoformat(),
                            "short": True
                        }
                    ]
                }
            ]
        }

        try:
            response = requests.post(
                self.slack_webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            self.logger.info(f"Slack alert sent successfully: {message}")
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")

    def send_email_alert(self, subject: str, body: str):
        """Send alert via email"""
        if not self.email_username or not self.alert_recipients:
            self.logger.warning("Email configuration incomplete")
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_username
            msg['To'] = ", ".join(self.alert_recipients)
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.email_host, self.email_port)
            server.starttls()
            server.login(self.email_username, self.email_password)
            
            text = msg.as_string()
            server.sendmail(self.email_username, self.alert_recipients, text)
            server.quit()
            
            self.logger.info(f"Email alert sent: {subject}")
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")

    def log_pipeline_event(self, event_data: Dict[str, Any]):
        """Log pipeline event to monitoring system"""
        event_data['timestamp'] = datetime.now().isoformat()
        
        # Log to console
        self.logger.info(f"Pipeline event: {json.dumps(event_data)}")
        
        # In a real implementation, you might send this to a monitoring system like Prometheus
        # or a log aggregation service

    def check_pipeline_health(self) -> Dict[str, Any]:
        """Check overall pipeline health"""
        # This would integrate with your CI/CD system API to get real health data
        # For now, we'll return a mock implementation
        return {
            "status": "healthy",
            "last_run": datetime.now().isoformat(),
            "success_rate": 95.5,
            "avg_duration": 180,  # seconds
            "pending_jobs": 0,
            "failed_jobs": 0
        }

    def monitor_deployment(self, deployment_info: Dict[str, Any]):
        """Monitor deployment status and send alerts if needed"""
        status = deployment_info.get('status', 'unknown')
        environment = deployment_info.get('environment', 'unknown')
        deployment_id = deployment_info.get('id', 'unknown')
        
        if status == 'failed':
            message = f"Deployment failed in {environment} environment"
            self.send_slack_alert(message, f"deployment-{environment}", "failed")
            
            email_body = f"""
            Deployment Alert
            
            Environment: {environment}
            Deployment ID: {deployment_id}
            Status: {status}
            Time: {datetime.now().isoformat()}
            
            Please check the deployment logs for more details.
            """
            self.send_email_alert(f"Deployment Failed: {environment}", email_body)
        
        elif status == 'successful':
            self.logger.info(f"Deployment successful in {environment}")
        else:
            self.logger.info(f"Deployment status: {status} in {environment}")


# Example usage
if __name__ == "__main__":
    monitor = PipelineMonitor()
    
    # Example pipeline event
    event = {
        "job": "test-unit",
        "status": "failed",
        "reason": "Test failure in core module",
        "duration": 120
    }
    
    monitor.log_pipeline_event(event)
    monitor.send_slack_alert("Unit tests failed", "test-unit", "failed")