"""
Alert management for Slack/PagerDuty integration
"""

import requests
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class AlertManager:
    """
    Send alerts to Slack, PagerDuty, or webhook
    """

    def __init__(self, slack_webhook: str = None, pagerduty_key: str = None):
        self.slack_webhook = slack_webhook
        self.pagerduty_key = pagerduty_key

    def send_alert(self, title: str, message: str, severity: str = "warning"):
        """Send alert to configured channels"""

        if self.slack_webhook:
            self._send_slack(title, message, severity)

        if self.pagerduty_key and severity in ["critical", "error"]:
            self._send_pagerduty(title, message, severity)

    def _send_slack(self, title: str, message: str, severity: str):
        """Send Slack notification"""
        color_map = {
            "info": "#36a64f",
            "warning": "#ffcc00",
            "error": "#ff4444",
            "critical": "#ff0000"
        }

        payload = {
            "attachments": [{
                "color": color_map.get(severity, "#36a64f"),
                "title": title,
                "text": message,
                "footer": "Sentiment Analysis MLOps Pipeline",
                "ts": int(datetime.now().timestamp())
            }]
        }

        try:
            response = requests.post(self.slack_webhook, json=payload, timeout=5)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    def _send_pagerduty(self, title: str, message: str, severity: str):
        """Send PagerDuty incident"""
        payload = {
            "routing_key": self.pagerduty_key,
            "event_action": "trigger",
            "title": title,
            "severity": severity,
            "custom_details": {
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
        }

        try:
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=5
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
