"""Model registry service: registration, stage machine, promotion gates.

Governance rules enforced here — for every caller, including the deployment
manager:

- Versions are unique per model and identified by semantic versions.
- Stage transitions follow an explicit allowed-transition map.
- Promotion to PRODUCTION must pass the ``PromotionPolicy`` metric gates and
  (by default) requires a named approver.
- Promoting a version archives the version currently in production.
- Every transition is appended to an audit history.
"""
from __future__ import annotations

from .models import ModelVersion, Stage, StageTransition
from .store import InMemoryRegistryStore, RegistryStore


class InvalidTransitionError(ValueError):
    """Raised when a stage transition is not allowed."""


class PromotionGateError(ValueError):
    """Raised when a version fails the production promotion policy."""


ALLOWED_TRANSITIONS: dict[Stage, set[Stage]] = {
    Stage.NONE: {Stage.STAGING, Stage.ARCHIVED},
    Stage.STAGING: {Stage.PRODUCTION, Stage.ARCHIVED, Stage.NONE},
    Stage.PRODUCTION: {Stage.ARCHIVED, Stage.STAGING},  # STAGING = rollback path
    Stage.ARCHIVED: {Stage.STAGING},  # restore path
}


class PromotionPolicy:
    """Gates that a version must pass to reach production.

    ``min_metrics``: validation metrics that must be present and >= threshold.
    ``require_approval``: a named approver is mandatory for production.
    """

    def __init__(self, min_metrics: dict[str, float] | None = None, require_approval: bool = True) -> None:
        self.min_metrics = min_metrics or {}
        self.require_approval = require_approval

    def check(self, mv: ModelVersion, approved_by: str | None) -> None:
        failures: list[str] = []
        for metric, threshold in self.min_metrics.items():
            value = mv.metrics.get(metric)
            if value is None:
                failures.append(f"missing required metric '{metric}'")
            elif value < threshold:
                failures.append(f"metric '{metric}'={value:.4f} below threshold {threshold:.4f}")
        if self.require_approval and not approved_by:
            failures.append("production promotion requires a named approver")
        if failures:
            raise PromotionGateError(f"{mv.key} failed promotion gates: " + "; ".join(failures))


def _parse_version(v: str) -> tuple[int, ...]:
    parts = []
    for piece in v.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


class ModelRegistry:
    def __init__(self, store: RegistryStore | None = None, policy: PromotionPolicy | None = None) -> None:
        self._store = store or InMemoryRegistryStore()
        self._policy = policy or PromotionPolicy()
        self._history: list[StageTransition] = []

    # -- registration ----------------------------------------------------
    def register(
        self,
        name: str,
        artifact_uri: str,
        version: str | None = None,
        framework: str = "unknown",
        metrics: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
        description: str = "",
        created_by: str = "system",
    ) -> ModelVersion:
        if version is None:
            version = self._next_version(name)
        if self._store.get(name, version) is not None:
            raise ValueError(f"model version already registered: {name}@{version}")
        mv = ModelVersion(
            name=name, version=version, artifact_uri=artifact_uri,
            framework=framework, metrics=dict(metrics or {}), tags=dict(tags or {}),
            description=description, created_by=created_by,
        )
        self._store.save(mv)
        return mv

    def _next_version(self, name: str) -> str:
        existing = [_parse_version(mv.version) for mv in self._store.list(name)]
        if not existing:
            return "1.0.0"
        major, minor, patch = (max(existing) + (0, 0, 0))[:3]
        return f"{major}.{minor}.{patch + 1}"

    # -- reads -------------------------------------------------------------
    def get(self, name: str, version: str) -> ModelVersion:
        mv = self._store.get(name, version)
        if mv is None:
            raise KeyError(f"unknown model version: {name}@{version}")
        return mv

    def list_versions(self, name: str, stage: Stage | None = None) -> list[ModelVersion]:
        out = self._store.list(name)
        if stage is not None:
            out = [mv for mv in out if mv.stage == stage]
        return out

    def production_version(self, name: str) -> ModelVersion | None:
        prod = self.list_versions(name, Stage.PRODUCTION)
        return prod[0] if prod else None

    def history(self, name: str | None = None) -> list[StageTransition]:
        return [t for t in self._history if name is None or t.model_name == name]

    # -- stage machine -----------------------------------------------------
    def transition_stage(
        self,
        name: str,
        version: str,
        to_stage: Stage,
        actor: str = "system",
        approved_by: str | None = None,
        reason: str = "",
    ) -> ModelVersion:
        mv = self.get(name, version)
        if mv.stage == to_stage:
            return mv
        if to_stage not in ALLOWED_TRANSITIONS[mv.stage]:
            raise InvalidTransitionError(
                f"{mv.key}: transition {mv.stage.value} -> {to_stage.value} is not allowed"
            )
        if to_stage == Stage.PRODUCTION:
            self._policy.check(mv, approved_by)
            current = self.production_version(name)
            if current is not None and current.version != version:
                self._apply_transition(current, Stage.ARCHIVED, actor, reason="superseded by " + version)
        self._apply_transition(mv, to_stage, actor, reason)
        return mv

    def _apply_transition(self, mv: ModelVersion, to_stage: Stage, actor: str, reason: str) -> None:
        transition = StageTransition(
            model_name=mv.name, version=mv.version,
            from_stage=mv.stage, to_stage=to_stage, actor=actor, reason=reason,
        )
        mv.stage = to_stage
        self._store.save(mv)
        self._history.append(transition)
