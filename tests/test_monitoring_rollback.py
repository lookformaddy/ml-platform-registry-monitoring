import random

import pytest

from mlplatform.deployment.manager import DeploymentManager, Strategy
from mlplatform.monitoring.alerts import InMemoryAlertSink, Severity
from mlplatform.monitoring.monitor import (
    EvaluationBatch, ModelMonitor, MonitorConfig, ReferenceProfile, RollbackController,
)
from mlplatform.registry.service import ModelRegistry, PromotionPolicy

random.seed(11)


def gauss(mean, sd, n):
    return [random.gauss(mean, sd) for _ in range(n)]


@pytest.fixture()
def platform():
    reg = ModelRegistry(policy=PromotionPolicy(min_metrics={"auc": 0.70}))
    mgr = DeploymentManager(reg)
    sink = InMemoryAlertSink()
    config = MonitorConfig(rollback_consecutive_critical=2)
    monitor = ModelMonitor(config=config, sinks=[sink])
    controller = RollbackController(mgr, config=config, sinks=[sink])
    reg.register("m", "s3://m/v1", version="1.0.0", metrics={"auc": 0.80})
    reg.register("m", "s3://m/v2", version="2.0.0", metrics={"auc": 0.86})
    mgr.deploy("m", "1.0.0", strategy=Strategy.IMMEDIATE, actor="a", approved_by="boss")
    monitor.set_reference(ReferenceProfile("m", "1.0.0", {"x": gauss(0, 1, 800)}))
    mgr.deploy("m", "2.0.0", strategy=Strategy.IMMEDIATE, actor="a", approved_by="boss")
    return mgr, sink, monitor, controller


def clean_batch():
    return EvaluationBatch(features={"x": gauss(0, 1, 800)})


def drifted_batch():
    return EvaluationBatch(features={"x": gauss(5, 1, 800)})


def test_clean_batch_produces_no_alerts(platform):
    _mgr, _sink, monitor, controller = platform
    report = monitor.evaluate("m", clean_batch())
    assert report.alerts == []
    assert controller.on_report(report) is False


def test_drifted_batch_raises_critical_drift_alert(platform):
    _mgr, sink, monitor, _controller = platform
    report = monitor.evaluate("m", drifted_batch())
    assert report.has_critical
    assert any(a.kind == "DRIFT" for a in sink.alerts)


def test_single_critical_does_not_roll_back_but_consecutive_do(platform):
    mgr, sink, monitor, controller = platform
    first = monitor.evaluate("m", drifted_batch())
    assert controller.on_report(first) is False
    assert mgr.serving_version("m") == "2.0.0"
    second = monitor.evaluate("m", drifted_batch())
    assert controller.on_report(second) is True
    assert mgr.serving_version("m") == "1.0.0"
    rollbacks = [a for a in sink.alerts if a.kind == "ROLLBACK"]
    assert len(rollbacks) == 1
    assert rollbacks[0].severity == Severity.CRITICAL


def test_clean_evaluation_resets_consecutive_counter(platform):
    mgr, _sink, monitor, controller = platform
    controller.on_report(monitor.evaluate("m", drifted_batch()))
    controller.on_report(monitor.evaluate("m", clean_batch()))
    controller.on_report(monitor.evaluate("m", drifted_batch()))
    assert mgr.serving_version("m") == "2.0.0"  # never two in a row


def test_biased_batch_raises_fairness_alert(platform):
    _mgr, _sink, monitor, _controller = platform
    groups = ["a"] * 100 + ["b"] * 100
    y_pred = [1] * 90 + [0] * 10 + [1] * 10 + [0] * 90
    report = monitor.evaluate("m", EvaluationBatch(y_pred=y_pred, groups=groups))
    assert report.fairness is not None
    assert any(a.kind == "FAIRNESS" and a.severity == Severity.CRITICAL for a in report.alerts)
