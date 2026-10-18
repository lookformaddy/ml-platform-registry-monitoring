import pytest

from mlplatform.deployment.manager import DeploymentManager, Strategy
from mlplatform.registry.models import Stage
from mlplatform.registry.service import ModelRegistry, PromotionPolicy


@pytest.fixture()
def platform():
    reg = ModelRegistry(policy=PromotionPolicy(min_metrics={"auc": 0.70}))
    mgr = DeploymentManager(reg)
    reg.register("m", "s3://m/v1", version="1.0.0", metrics={"auc": 0.80})
    reg.register("m", "s3://m/v2", version="2.0.0", metrics={"auc": 0.85})
    mgr.deploy("m", "1.0.0", strategy=Strategy.IMMEDIATE, actor="a", approved_by="boss")
    return reg, mgr


def test_immediate_deploy_serves_and_registers_production(platform):
    reg, mgr = platform
    assert mgr.serving_version("m") == "1.0.0"
    assert reg.production_version("m").version == "1.0.0"
    assert mgr.route("m", "user-1") == "1.0.0"


def test_canary_requires_existing_production():
    reg = ModelRegistry(policy=PromotionPolicy(require_approval=False))
    mgr = DeploymentManager(reg)
    reg.register("m", "s3://m/v1", version="1.0.0")
    with pytest.raises(ValueError):
        mgr.deploy("m", "1.0.0", strategy=Strategy.CANARY)


def test_canary_traffic_split_is_deterministic_and_roughly_proportional(platform):
    _reg, mgr = platform
    dep = mgr.deploy("m", "2.0.0", strategy=Strategy.CANARY, steps=(20, 100), approved_by="boss")
    assert dep.traffic_percent == 20
    keys = [f"user-{i}" for i in range(2000)]
    first = [mgr.route("m", k) for k in keys]
    second = [mgr.route("m", k) for k in keys]
    assert first == second  # stable assignment
    share = sum(1 for v in first if v == "2.0.0") / len(first)
    assert 0.14 < share < 0.26


def test_canary_advance_to_full_promotes(platform):
    reg, mgr = platform
    dep = mgr.deploy("m", "2.0.0", strategy=Strategy.CANARY, steps=(50, 100),
                     actor="a", approved_by="boss")
    mgr.advance("m", actor="a", approved_by="boss")
    assert dep.status == "promoted"
    assert mgr.serving_version("m") == "2.0.0"
    assert reg.production_version("m").version == "2.0.0"
    assert reg.get("m", "1.0.0").stage == Stage.ARCHIVED


def test_shadow_never_serves_but_scores_agreement(platform):
    _reg, mgr = platform
    mgr.deploy("m", "2.0.0", strategy=Strategy.SHADOW)
    assert mgr.route("m", "user-9") == "1.0.0"
    for i in range(10):
        mgr.record_shadow("m", production_prediction=1, candidate_prediction=1 if i < 8 else 0)
    assert mgr.shadow_agreement("m") == pytest.approx(0.8)


def test_promote_shadow(platform):
    reg, mgr = platform
    mgr.deploy("m", "2.0.0", strategy=Strategy.SHADOW)
    mgr.promote_shadow("m", actor="a", approved_by="boss")
    assert mgr.serving_version("m") == "2.0.0"
    assert reg.production_version("m").version == "2.0.0"


def test_rollback_restores_previous_production(platform):
    reg, mgr = platform
    mgr.deploy("m", "2.0.0", strategy=Strategy.IMMEDIATE, actor="a", approved_by="boss")
    assert mgr.serving_version("m") == "2.0.0"
    restored = mgr.rollback("m", reason="bad metrics", actor="ops")
    assert restored == "1.0.0"
    assert mgr.serving_version("m") == "1.0.0"
    assert reg.production_version("m").version == "1.0.0"
    assert reg.get("m", "2.0.0").stage == Stage.STAGING


def test_rollback_cancels_active_canary(platform):
    _reg, mgr = platform
    mgr.deploy("m", "2.0.0", strategy=Strategy.CANARY, steps=(50, 100), approved_by="boss")
    restored = mgr.rollback("m", reason="canary analysis failed", actor="ops")
    assert restored == "1.0.0"
    assert mgr.active_deployment("m") is None
    assert mgr.route("m", "user-3") == "1.0.0"
