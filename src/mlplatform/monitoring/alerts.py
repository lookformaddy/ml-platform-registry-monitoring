"""Structured alerts and pluggable alert sinks."""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    kind: str  # DRIFT | FAIRNESS | ROLLBACK | DEPLOYMENT
    severity: Severity
    model_name: str
    message: str
    version: str = ""
    details: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)


class AlertSink(ABC):
    """Where alerts go. Implement against PagerDuty/Slack/email in production."""

    @abstractmethod
    def emit(self, alert: Alert) -> None:
        ...


class InMemoryAlertSink(AlertSink):
    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        self.alerts.append(alert)

    def by_severity(self, severity: Severity) -> list[Alert]:
        return [a for a in self.alerts if a.severity == severity]
