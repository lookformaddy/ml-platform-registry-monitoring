import pytest

from mlplatform.registry.models import Stage
from mlplatform.registry.service import (
    InvalidTransitionError, ModelRegistry, PromotionGateError, PromotionPolicy,
)
from mlplatform.registry.store import JsonFileRegistryStore


def make_registry(**policy_kwargs):
    return ModelRegistry(policy=PromotionPolicy(**policy_kwargs))


def test_register_and_auto_version():
    reg = make_registry()
    v1 = reg.register("m", "s3://m/v1")
    v2 = reg.register("m", "s3://m/v2")
    assert (v1.version, v2.version) == ("1.0.0", "1.0.1")
    assert v1.stage == Stage.NONE


def test_duplicate_version_rejected():
    reg = make_registry()
    reg.register("m", "s3://m/v1", version="1.0.0")
    with pytest.raises(ValueError):
        reg.register("m", "s3://m/v1b", version="1.0.0")


def test_stage_machine_happy_path_and_history():
    reg = make_registry(require_approval=False)
    reg.register("m", "s3://m/v1", version="1.0.0")
    reg.transition_stage("m", "1.0.0", Stage.STAGING, actor="alice")
    reg.transition_stage("m", "1.0.0", Stage.PRODUCTION, actor="alice")
    assert reg.production_version("m").version == "1.0.0"
    history = reg.history("m")
    assert [t.to_stage for t in history] == [Stage.STAGING, Stage.PRODUCTION]


def test_illegal_transition_rejected():
    reg = make_registry(require_approval=False)
    reg.register("m", "s3://m/v1", version="1.0.0")
    with pytest.raises(InvalidTransitionError):
        reg.transition_stage("m", "1.0.0", Stage.PRODUCTION, actor="alice")


def test_promotion_gate_blocks_low_metric():
    reg = make_registry(min_metrics={"auc": 0.75})
    reg.register("m", "s3://m/v1", version="1.0.0", metrics={"auc": 0.71})
    reg.transition_stage("m", "1.0.0", Stage.STAGING, actor="alice")
    with pytest.raises(PromotionGateError):
        reg.transition_stage("m", "1.0.0", Stage.PRODUCTION, actor="alice", approved_by="boss")


def test_promotion_requires_approver():
    reg = make_registry(min_metrics={"auc": 0.75}, require_approval=True)
    reg.register("m", "s3://m/v1", version="1.0.0", metrics={"auc": 0.9})
    reg.transition_stage("m", "1.0.0", Stage.STAGING, actor="alice")
    with pytest.raises(PromotionGateError):
        reg.transition_stage("m", "1.0.0", Stage.PRODUCTION, actor="alice")
    mv = reg.transition_stage("m", "1.0.0", Stage.PRODUCTION, actor="alice", approved_by="boss")
    assert mv.stage == Stage.PRODUCTION


def test_new_production_archives_previous():
    reg = make_registry(require_approval=False)
    for v in ("1.0.0", "2.0.0"):
        reg.register("m", f"s3://m/{v}", version=v)
        reg.transition_stage("m", v, Stage.STAGING, actor="a")
        reg.transition_stage("m", v, Stage.PRODUCTION, actor="a")
    assert reg.get("m", "1.0.0").stage == Stage.ARCHIVED
    assert reg.production_version("m").version == "2.0.0"


def test_json_file_store_roundtrip(tmp_path):
    path = str(tmp_path / "registry.json")
    reg = ModelRegistry(store=JsonFileRegistryStore(path),
                        policy=PromotionPolicy(require_approval=False))
    reg.register("m", "s3://m/v1", version="1.0.0", metrics={"auc": 0.9})
    reg.transition_stage("m", "1.0.0", Stage.STAGING, actor="a")
    # A fresh registry over the same file sees the persisted state.
    reg2 = ModelRegistry(store=JsonFileRegistryStore(path))
    mv = reg2.get("m", "1.0.0")
    assert mv.stage == Stage.STAGING
    assert mv.metrics == {"auc": 0.9}
