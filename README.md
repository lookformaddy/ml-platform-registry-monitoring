# ML Platform — Registry, Deployment & Monitoring

![track](https://img.shields.io/badge/track-ml--engineering-blue) ![python](https://img.shields.io/badge/python-3.10%2B-green) ![deps](https://img.shields.io/badge/core%20deps-zero-lightgrey)

An enterprise ML platform core: a **model registry** with versioning and governed
promotion, a **deployment pipeline** supporting immediate, canary and shadow
releases, and **production monitoring** for data drift and bias with
**automatic rollback** when guardrails are breached.

This is the platform layer every enterprise ML organization ends up building
between "a model that trains" and "a model that is trusted in production":
who approved this version, what is it allowed to serve, how do we know it is
still healthy, and how do we get back to the last good version in one step?

## What it demonstrates

- **Governed model registry** — immutable versions, lifecycle stages
  (`none → staging → production → archived`), promotion gates on validation
  metrics, explicit approval for production, a full audit history of every
  transition, and automatic archival of the superseded production version.
- **Pluggable persistence** — registry storage behind an interface with an
  in-memory implementation (tests, local dev) and a JSON file-backed
  implementation (durable, zero infrastructure).
- **Deployment strategies** — immediate replacement, **canary** releases with
  deterministic traffic bucketing and step-wise rollout (5% → 25% → 50% →
  100%), and **shadow** deployments that score live traffic without serving
  it, with an agreement-rate readout before promotion.
- **Drift detection** — Population Stability Index (PSI) and the two-sample
  Kolmogorov–Smirnov statistic per feature, with warning/critical severity
  classification.
- **Bias monitoring** — demographic parity difference, disparate impact
  (four-fifths rule) and equal-opportunity difference across groups.
- **Closed-loop safety** — a monitoring pipeline that emits structured alerts
  and a rollback controller that automatically restores the previous
  production version after consecutive critical breaches.
- **Zero-dependency core** — the library is pure standard library; FastAPI is
  an optional extra for the HTTP API.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the component diagram,
data flow, and design trade-offs.

```
src/mlplatform/
  registry/     models, storage interface + implementations, registry service & promotion gates
  deployment/   deployment manager: canary / shadow / immediate, traffic routing, rollback
  monitoring/   drift (PSI, KS), fairness metrics, alert sinks, monitor, rollback controller
  api.py        optional FastAPI surface over registry, deployment and monitoring
examples/demo.py  end-to-end scenario on synthetic data
tests/            pytest suites for registry, deployment, drift, fairness, monitoring+rollback
```

## Quickstart

```bash
# run the test suite (no installation needed)
python -m pytest -q            # or: make test

# run the end-to-end demo (registers two versions, canaries v2,
# then simulates drift and watches the platform roll back)
make demo

# optional HTTP API
pip install ".[api]"
uvicorn mlplatform.api:app --port 8080
```

## Design decisions & trade-offs

- **Interfaces first, infrastructure last.** Storage, alert sinks and serving
  are protocols/ABCs with local implementations, so the same core can be
  backed by a managed ML platform, a message bus, or a feature-flag system in
  a real deployment without changing business logic.
- **Deterministic traffic bucketing.** Canary assignment hashes the request
  key, so a given user consistently sees the same variant during a rollout —
  a hard requirement for trustworthy canary analysis.
- **Promotion is a policy, not a button.** Moving to production checks
  validation-metric gates and requires a named approver; every transition is
  recorded in an audit history. Convenience never bypasses governance.
- **Monitoring closes the loop.** Drift/bias alerts are not just dashboards:
  consecutive critical breaches trigger an automatic rollback to the previous
  production version and emit a rollback alert for the audit trail.
- **Classical statistics, deliberately.** PSI and KS are the industry-standard
  first line for tabular drift; fairness uses the metrics regulators and model
  risk teams actually ask for (disparate impact / four-fifths rule).

## Track

Part of the **ML Engineering** project series — production ML platform
patterns that enterprise AI organizations interview for.
