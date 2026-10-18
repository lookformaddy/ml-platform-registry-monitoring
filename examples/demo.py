"""End-to-end demo on synthetic data.

Scenario: a churn model v1 is in production. v2 is registered, passes the
promotion gates, and rolls out as a canary. Then the live feature distribution
shifts hard and predictions become biased against one group. The monitor
raises critical alerts twice in a row — and the platform rolls back to v1
automatically.

Run: PYTHONPATH=src python examples/demo.py   (or: make demo)
"""
from __future__ import annotations

import random

from mlplatform.deployment.manager import DeploymentManager, Strategy
from mlplatform.monitoring.alerts import InMemoryAlertSink
from mlplatform.monitoring.monitor import (
    EvaluationBatch, ModelMonitor, MonitorConfig, ReferenceProfile, RollbackController,
)
from mlplatform.registry.service import ModelRegistry, PromotionPolicy

random.seed(42)


def sample(mean: float, sd: float, n: int) -> list[float]:
    return [random.gauss(mean, sd) for _ in range(n)]


def main() -> None:
    registry = ModelRegistry(policy=PromotionPolicy(min_metrics={"auc": 0.70}))
    manager = DeploymentManager(registry)
    sink = InMemoryAlertSink()
    config = MonitorConfig()
    monitor = ModelMonitor(config=config, sinks=[sink])
    controller = RollbackController(manager, config=config, sinks=[sink])

    print("== Register and ship v1 ==")
    registry.register("churn-model", "s3://models/churn/v1", version="1.0.0",
                      framework="sklearn", metrics={"auc": 0.81}, created_by="ml-team")
    manager.deploy("churn-model", "1.0.0", strategy=Strategy.IMMEDIATE,
                   actor="ml-team", approved_by="director-ml")
    print("serving:", manager.serving_version("churn-model"))

    reference = {"tenure_months": sample(24, 8, 500), "monthly_spend": sample(80, 20, 500)}
    monitor.set_reference(ReferenceProfile("churn-model", "1.0.0", reference))

    print("\n== Register v2 and start a canary ==")
    registry.register("churn-model", "s3://models/churn/v2", version="2.0.0",
                      framework="sklearn", metrics={"auc": 0.84}, created_by="ml-team")
    dep = manager.deploy("churn-model", "2.0.0", strategy=Strategy.CANARY,
                         actor="ml-team", approved_by="director-ml")
    while dep.status == "active":
        manager.advance("churn-model", actor="ml-team", approved_by="director-ml")
        print(f"canary step -> traffic {dep.traffic_percent}%  status={dep.status}")
    print("serving:", manager.serving_version("churn-model"))

    print("\n== Simulate drifted + biased live traffic ==")
    drifted = {"tenure_months": sample(6, 4, 500), "monthly_spend": sample(160, 30, 500)}
    n = 400
    groups = ["group-a"] * (n // 2) + ["group-b"] * (n // 2)
    y_pred = [1] * 160 + [0] * 40 + [1] * 30 + [0] * 170  # group-a favored heavily
    batch = EvaluationBatch(features=drifted, y_pred=y_pred, groups=groups)
    for round_no in (1, 2):
        report = monitor.evaluate("churn-model", batch)
        fired = controller.on_report(report)
        print(f"evaluation {round_no}: critical={report.has_critical} alerts={len(report.alerts)} rollback={fired}")
    print("serving after guardrails:", manager.serving_version("churn-model"))

    print("\n== Alert trail ==")
    for alert in sink.alerts:
        print(f"[{alert.severity.value:>8}] {alert.kind:<8} {alert.message}")


if __name__ == "__main__":
    main()
