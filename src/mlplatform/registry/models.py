"""Core registry data model: immutable model versions and stage transitions."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class Stage(str, Enum):
    """Lifecycle stage of a model version."""

    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class StageTransition:
    """One audited stage change."""

    model_name: str
    version: str
    from_stage: Stage
    to_stage: Stage
    actor: str
    reason: str = ""
    at: float = field(default_factory=time.time)


@dataclass
class ModelVersion:
    """An immutable, registered model version and its validation record."""

    name: str
    version: str
    artifact_uri: str
    framework: str = "unknown"
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    description: str = ""
    created_by: str = "system"
    stage: Stage = Stage.NONE
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"
