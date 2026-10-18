"""Optional FastAPI surface over the registry, deployment and monitoring.

Install with: pip install ".[api]"
Run with:    uvicorn mlplatform.api:app --port 8080

The core library has no web dependency; this module is a thin transport
adapter so the platform can sit behind an internal ML portal or gateway.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .deployment.manager import DeploymentManager, Strategy
from .monitoring.alerts import InMemoryAlertSink
from .monitoring.monitor import EvaluationBatch, ModelMonitor, ReferenceProfile
from .registry.models import Stage
from .registry.service import ModelRegistry, PromotionPolicy
from .registry.store import JsonFileRegistryStore

_store_path = os.environ.get("MLPLATFORM_REGISTRY_PATH")
_store = JsonFileRegistryStore(_store_path) if _store_path else None
registry = ModelRegistry(store=_store, policy=PromotionPolicy(min_metrics={"auc": 0.70}))
manager = DeploymentManager(registry)
sink = InMemoryAlertSink()
monitor = ModelMonitor(sinks=[sink])

app = FastAPI(title="ML Platform — Registry, Deployment & Monitoring")


class RegisterRequest(BaseModel):
    name: str
    artifact_uri: str
    version: str | None = None
    framework: str = "unknown"
    metrics: dict[str, float] = {}
    created_by: str = "api"


class TransitionRequest(BaseModel):
    to_stage: str
    actor: str = "api"
    approved_by: str | None = None
    reason: str = ""


class DeployRequest(BaseModel):
    version: str
    strategy: str = "canary"
    actor: str = "api"
    approved_by: str | None = None


class ReferenceRequest(BaseModel):
    version: str
    features: dict[str, list[float]]


class EvaluateRequest(BaseModel):
    features: dict[str, list[float]] = {}
    y_pred: list[int] | None = None
    y_true: list[int] | None = None
    groups: list[str] | None = None


@app.post("/models")
def register_model(req: RegisterRequest):
    mv = registry.register(req.name, req.artifact_uri, version=req.version,
                           framework=req.framework, metrics=req.metrics,
                           created_by=req.created_by)
    return {"name": mv.name, "version": mv.version, "stage": mv.stage.value}


@app.get("/models/{name}/versions")
def list_versions(name: str):
    return [{"version": mv.version, "stage": mv.stage.value, "metrics": mv.metrics}
            for mv in registry.list_versions(name)]


@app.post("/models/{name}/versions/{version}/transition")
def transition(name: str, version: str, req: TransitionRequest):
    try:
        mv = registry.transition_stage(name, version, Stage(req.to_stage),
                                       actor=req.actor, approved_by=req.approved_by,
                                       reason=req.reason)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"name": mv.name, "version": mv.version, "stage": mv.stage.value}


@app.post("/models/{name}/deployments")
def deploy(name: str, req: DeployRequest):
    try:
        dep = manager.deploy(name, req.version, strategy=Strategy(req.strategy),
                             actor=req.actor, approved_by=req.approved_by)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deployment_id": dep.id, "strategy": dep.strategy.value,
            "traffic_percent": dep.traffic_percent, "status": dep.status}


@app.post("/models/{name}/deployments/advance")
def advance(name: str, actor: str = "api", approved_by: str | None = None):
    try:
        dep = manager.advance(name, actor=actor, approved_by=approved_by)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"traffic_percent": dep.traffic_percent, "status": dep.status,
            "serving": manager.serving_version(name)}


@app.post("/models/{name}/rollback")
def rollback(name: str, reason: str = "manual rollback via API"):
    restored = manager.rollback(name, reason=reason, actor="api")
    return {"serving": restored}


@app.post("/models/{name}/reference")
def set_reference(name: str, req: ReferenceRequest):
    monitor.set_reference(ReferenceProfile(model_name=name, version=req.version, features=req.features))
    return {"ok": True}


@app.post("/models/{name}/evaluate")
def evaluate(name: str, req: EvaluateRequest):
    report = monitor.evaluate(name, EvaluationBatch(
        features=req.features, y_pred=req.y_pred, y_true=req.y_true, groups=req.groups))
    return {"has_critical": report.has_critical, "has_warning": report.has_warning,
            "alerts": [{"kind": a.kind, "severity": a.severity.value, "message": a.message}
                       for a in report.alerts]}
