# Architecture — ML Platform: Registry, Deployment & Monitoring

## Problem framing

Production ML fails organizationally more often than it fails statistically:
nobody can say which model version is serving, who approved it, whether its
inputs still look like training data, or how to revert it. This platform core
addresses the four control points an enterprise needs around a live model:

1. **Registry** — a single source of truth for model versions and lifecycle.
2. **Deployment** — controlled exposure of new versions (canary / shadow).
3. **Monitoring** — continuous drift and bias surveillance against a reference.
4. **Rollback** — automatic, audited return to the last known-good version.

## Components

```
            +-------------------- Model Registry --------------------+
            |  ModelVersion (immutable)   Stage machine              |
            |  PromotionPolicy (metric gates + approver)           |
            |  RegistryStore (interface)                             |
            |    |- InMemoryRegistryStore                            |
            |    '- JsonFileRegistryStore                            |
            +------------------------+-------------------------------+
                                     | governs
                                     v
            +------------------ Deployment Manager ------------------+
            |  Strategies: IMMEDIATE | CANARY | SHADOW               |
            |  TrafficRouter: sha256(request_key) % 100 < canary %   |
            |  previous-version tracking for one-step rollback       |
            +------------------------+-------------------------------+
                                     | serves versions
                                     v
   live traffic / batches --->  ModelMonitor
            |                     |- DriftDetector: PSI + KS per feature
            |                     |- FairnessEvaluator: DPD, DI, EOD
            |                     '- AlertSink(s): in-memory / callback
            v
   RollbackController: N consecutive critical reports
            -> DeploymentManager.rollback() + ROLLBACK alert (audit)
```

## Data flow

1. A trained model is **registered** as an immutable `ModelVersion` with its
   validation metrics. It starts in stage `none`.
2. Promotion to `staging` is free; promotion to `production` passes the
   `PromotionPolicy` gates (required metrics at or above thresholds) and
   requires a named approver. The previous production version is archived
   automatically. Every transition enters the audit history.
3. A **deployment** exposes a version: immediately, as a canary with stepped
   traffic percentages, or in shadow mode (scored but never served).
4. The **monitor** holds a reference profile (feature samples captured at
   promotion time). Each evaluation batch is compared against it (PSI/KS)
   and, when group labels are present, checked for fairness regressions.
5. Alerts flow to sinks. The **rollback controller** counts consecutive
   critical reports per model; at the configured threshold it rolls the
   deployment back to the previous production version and records a rollback
   alert — closing the detect → decide → act loop.

## Key design choices

| Decision | Rationale | Trade-off |
|---|---|---|
| Pure-stdlib statistics (PSI, KS) | Zero-dependency core; statistics are transparent and auditable for model-risk review | No GPU/distributed scale — acceptable at the control-plane layer |
| Deterministic hash bucketing | Stable per-user variant assignment for valid canary analysis | Bucket distribution is approximate, not exact quotas |
| Stage machine + gates in the registry | Governance cannot be bypassed by any caller, including the deployment manager | Slightly more ceremony for experiments (by design) |
| Interface-backed storage & sinks | Swap local implementations for managed services without touching logic | More upfront abstraction than a demo needs — the point of the exercise |
| Consecutive-breach rollback | Avoids flapping on a single noisy batch while still bounding exposure time | A slow-burn breach below critical never auto-rolls back; it alerts instead |

## Extension points

- `RegistryStore`: implement against a database or an existing ML platform.
- `AlertSink`: implement against PagerDuty/Slack/email.
- Reference profiles: capture from the feature store at promotion time.
- Deployment strategies: add blue/green or A/B with significance gates by
  composing `DeploymentManager.advance()` with monitor readouts.
