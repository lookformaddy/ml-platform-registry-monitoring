"""Deployment manager: immediate, canary and shadow release strategies.

Traffic assignment for canaries is a deterministic hash of the request key,
so a given user consistently sees the same variant for the whole rollout —
a precondition for valid canary analysis. The manager coordinates with the
registry: full promotion goes through the registry's governed stage machine
(metric gates + named approver), and the previously serving version is
tracked for one-step rollback.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from ..registry.models import Stage
from ..registry.service import ModelRegistry


class Strategy(str, Enum):
    IMMEDIATE = "immediate"
    CANARY = "canary"
    SHADOW = "shadow"


@dataclass
class Deployment:
    model_name: str
    candidate_version: str
    strategy: Strategy
    steps: tuple[int, ...] = (100,)
    step_index: int = 0
    status: str = "active"  # active | promoted | rolled_back | cancelled
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)

    @property
    def traffic_percent(self) -> int:
        if self.strategy == Strategy.SHADOW:
            return 0
        if self.strategy == Strategy.IMMEDIATE:
            return 100
        return self.steps[min(self.step_index, len(self.steps) - 1)]


def _bucket(request_key: str) -> int:
    digest = hashlib.sha256(request_key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


class DeploymentManager:
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._serving: dict[str, str] = {}            # model -> serving version
        self._previous: dict[str, str | None] = {}    # model -> version before current
        self._active: dict[str, Deployment] = {}      # model -> active deployment
        self._shadow_agreements: dict[str, list[bool]] = {}

    # -- introspection -----------------------------------------------------
    def serving_version(self, model_name: str) -> str | None:
        if model_name not in self._serving:
            prod = self._registry.production_version(model_name)
            if prod is not None:
                self._serving[model_name] = prod.version
        return self._serving.get(model_name)

    def active_deployment(self, model_name: str) -> Deployment | None:
        dep = self._active.get(model_name)
        return dep if dep is not None and dep.status == "active" else None

    # -- deploy --------------------------------------------------------------
    def deploy(
        self,
        model_name: str,
        version: str,
        strategy: Strategy = Strategy.CANARY,
        steps: tuple[int, ...] = (5, 25, 50, 100),
        actor: str = "system",
        approved_by: str | None = None,
    ) -> Deployment:
        self._registry.get(model_name, version)  # must exist
        current = self.serving_version(model_name)
        if strategy != Strategy.IMMEDIATE and current is None:
            raise ValueError(
                f"{model_name}: {strategy.value} deployment requires an existing production version"
            )
        if strategy == Strategy.IMMEDIATE:
            dep = Deployment(model_name=model_name, candidate_version=version,
                             strategy=strategy, steps=(100,), status="promoted")
            self._promote(model_name, version, actor, approved_by)
            return dep
        dep = Deployment(model_name=model_name, candidate_version=version,
                         strategy=strategy, steps=tuple(steps))
        self._active[model_name] = dep
        if strategy == Strategy.SHADOW:
            self._shadow_agreements[model_name] = []
        return dep

    def _promote(self, model_name: str, version: str, actor: str, approved_by: str | None) -> None:
        current = self.serving_version(model_name)
        mv = self._registry.get(model_name, version)
        if mv.stage != Stage.PRODUCTION:
            if mv.stage == Stage.NONE:
                self._registry.transition_stage(model_name, version, Stage.STAGING, actor=actor)
            self._registry.transition_stage(
                model_name, version, Stage.PRODUCTION, actor=actor,
                approved_by=approved_by, reason="deployment promotion",
            )
        self._previous[model_name] = current
        self._serving[model_name] = version

    # -- routing ---------------------------------------------------------------
    def route(self, model_name: str, request_key: str) -> str | None:
        """Return the version that should serve this request."""
        dep = self.active_deployment(model_name)
        if dep is not None and dep.strategy == Strategy.CANARY:
            if _bucket(request_key) < dep.traffic_percent:
                return dep.candidate_version
        return self.serving_version(model_name)

    # -- canary lifecycle --------------------------------------------------------
    def advance(self, model_name: str, actor: str = "system", approved_by: str | None = None) -> Deployment:
        dep = self.active_deployment(model_name)
        if dep is None or dep.strategy != Strategy.CANARY:
            raise ValueError(f"{model_name}: no active canary deployment")
        if dep.step_index < len(dep.steps) - 1:
            dep.step_index += 1
        if dep.traffic_percent >= 100:
            self._promote(model_name, dep.candidate_version, actor, approved_by)
            dep.status = "promoted"
            del self._active[model_name]
        return dep

    # -- shadow lifecycle ----------------------------------------------------------
    def record_shadow(self, model_name: str, production_prediction, candidate_prediction) -> None:
        dep = self.active_deployment(model_name)
        if dep is None or dep.strategy != Strategy.SHADOW:
            raise ValueError(f"{model_name}: no active shadow deployment")
        self._shadow_agreements[model_name].append(production_prediction == candidate_prediction)

    def shadow_agreement(self, model_name: str) -> float:
        samples = self._shadow_agreements.get(model_name, [])
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    def promote_shadow(self, model_name: str, actor: str = "system", approved_by: str | None = None) -> Deployment:
        dep = self.active_deployment(model_name)
        if dep is None or dep.strategy != Strategy.SHADOW:
            raise ValueError(f"{model_name}: no active shadow deployment")
        self._promote(model_name, dep.candidate_version, actor, approved_by)
        dep.status = "promoted"
        del self._active[model_name]
        return dep

    # -- rollback ---------------------------------------------------------------------
    def rollback(self, model_name: str, reason: str = "", actor: str = "system") -> str | None:
        """Restore the previous production version in one step.

        Returns the version now serving, or None if there is nothing to
        roll back to (in which case an active canary/shadow is cancelled).
        """
        dep = self._active.pop(model_name, None)
        if dep is not None and dep.status == "active":
            dep.status = "rolled_back"
        previous = self._previous.get(model_name)
        current = self.serving_version(model_name)
        if previous is None or current is None or previous == current:
            return current
        # Demote the bad version, restore the previous one — audited through
        # the registry stage machine. The previous version was archived when
        # it was superseded, so it comes back via the governed restore path
        # (archived -> staging -> production, with promotion gates applied).
        self._registry.transition_stage(
            model_name, current, Stage.STAGING, actor=actor,
            reason=f"rollback: {reason}" if reason else "rollback",
        )
        prev_mv = self._registry.get(model_name, previous)
        if prev_mv.stage == Stage.ARCHIVED:
            self._registry.transition_stage(
                model_name, previous, Stage.STAGING, actor=actor, reason="rollback restore",
            )
        self._registry.transition_stage(
            model_name, previous, Stage.PRODUCTION, actor=actor,
            approved_by=actor, reason="rollback restore",
        )
        self._previous[model_name] = current  # keep roll-forward possible
        self._serving[model_name] = previous
        return previous
