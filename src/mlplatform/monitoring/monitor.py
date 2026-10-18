"""Model monitor and rollback controller — the closed safety loop.

The monitor compares live batches against the reference profile captured for
the production version: per-feature drift (PSI/KS) and, when group labels are
available, fairness metrics. Findings become structured alerts on the sinks.

The rollback controller consumes monitor reports: after a configurable number
of *consecutive* critical evaluations for a model, it instructs the
deployment manager to restore the previous production version and records a
ROLLBACK alert, so the whole detect -> decide -> act cycle is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..deployment.manager import DeploymentManager
from .alerts import Alert, AlertSink, Severity
from .drift import DriftReport, detect_drift
from .fairness import FairnessReport, evaluate_fairness


@dataclass
class MonitorConfig:
    psi_warn: float = 0.10
    psi_critical: float = 0.25
    ks_warn: float = 0.10
    ks_critical: float = 0.20
    dpd_warning: float = 0.10
    di_minimum: float = 0.80
    di_critical: float = 0.60
    eod_warning: float = 0.10
    rollback_consecutive_critical: int = 2


@dataclass
class ReferenceProfile:
    model_name: str
    version: str
    features: dict[str, list[float]]


@dataclass
class EvaluationBatch:
    features: dict[str, list[float]] = field(default_factory=dict)
    y_pred: list[int] | None = None
    y_true: list[int] | None = None
    groups: list[str] | None = None


@dataclass
class MonitoringReport:
    model_name: str
    version: str
    drift: list[DriftReport]
    fairness: FairnessReport | None
    alerts: list[Alert]

    @property
    def has_critical(self) -> bool:
        return any(a.severity == Severity.CRITICAL for a in self.alerts)

    @property
    def has_warning(self) -> bool:
        return any(a.severity == Severity.WARNING for a in self.alerts)


class ModelMonitor:
    def __init__(self, config: MonitorConfig | None = None, sinks: list[AlertSink] | None = None) -> None:
        self._config = config or MonitorConfig()
        self._sinks = sinks or []
        self._references: dict[str, ReferenceProfile] = {}

    def set_reference(self, profile: ReferenceProfile) -> None:
        self._references[profile.model_name] = profile

    def evaluate(self, model_name: str, batch: EvaluationBatch) -> MonitoringReport:
        cfg = self._config
        ref = self._references.get(model_name)
        version = ref.version if ref else ""
        alerts: list[Alert] = []

        drift_reports: list[DriftReport] = []
        if ref is not None and batch.features:
            drift_reports = detect_drift(
                ref.features, batch.features,
                psi_warn=cfg.psi_warn, psi_critical=cfg.psi_critical,
                ks_warn=cfg.ks_warn, ks_critical=cfg.ks_critical,
            )
            for rep in drift_reports:
                if rep.severity == "ok":
                    continue
                alerts.append(Alert(
                    kind="DRIFT",
                    severity=Severity.CRITICAL if rep.severity == "critical" else Severity.WARNING,
                    model_name=model_name, version=version,
                    message=f"feature '{rep.feature}' drift {rep.severity}: PSI={rep.psi:.3f}, KS={rep.ks:.3f}",
                    details={"feature": rep.feature, "psi": rep.psi, "ks": rep.ks},
                ))

        fairness_report: FairnessReport | None = None
        if batch.y_pred is not None and batch.groups is not None:
            fairness_report = evaluate_fairness(
                batch.y_pred, batch.groups, y_true=batch.y_true,
                dpd_warning=cfg.dpd_warning, di_minimum=cfg.di_minimum,
                eod_warning=cfg.eod_warning,
            )
            for breach in fairness_report.breaches:
                critical = fairness_report.disparate_impact < cfg.di_critical
                alerts.append(Alert(
                    kind="FAIRNESS",
                    severity=Severity.CRITICAL if critical else Severity.WARNING,
                    model_name=model_name, version=version,
                    message=f"fairness breach: {breach}",
                    details={"disparate_impact": fairness_report.disparate_impact,
                             "demographic_parity_difference": fairness_report.demographic_parity_difference},
                ))

        for alert in alerts:
            for sink in self._sinks:
                sink.emit(alert)
        return MonitoringReport(
            model_name=model_name, version=version,
            drift=drift_reports, fairness=fairness_report, alerts=alerts,
        )


class RollbackController:
    """Triggers automatic rollback after consecutive critical evaluations."""

    def __init__(self, manager: DeploymentManager, config: MonitorConfig | None = None,
                 sinks: list[AlertSink] | None = None) -> None:
        self._manager = manager
        self._config = config or MonitorConfig()
        self._sinks = sinks or []
        self._consecutive: dict[str, int] = {}

    def on_report(self, report: MonitoringReport) -> bool:
        """Consume a monitoring report; return True if a rollback fired."""
        name = report.model_name
        if report.has_critical:
            self._consecutive[name] = self._consecutive.get(name, 0) + 1
        else:
            self._consecutive[name] = 0
        if self._consecutive.get(name, 0) < self._config.rollback_consecutive_critical:
            return False
        restored = self._manager.rollback(
            name, reason="automatic rollback: consecutive critical monitoring breaches",
            actor="rollback-controller",
        )
        self._consecutive[name] = 0
        alert = Alert(
            kind="ROLLBACK", severity=Severity.CRITICAL, model_name=name,
            message=f"automatic rollback executed; serving version is now {restored}",
            details={"restored_version": restored},
        )
        for sink in self._sinks:
            sink.emit(alert)
        return True
