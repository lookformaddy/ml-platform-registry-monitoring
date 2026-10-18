"""Data drift detection: Population Stability Index and Kolmogorov–Smirnov.

PSI and the two-sample KS statistic are the industry-standard first line for
tabular feature drift: cheap, explainable, and auditable by model-risk teams.
Implemented in pure standard library so the monitoring core has no numeric
dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_EPS = 1e-4


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("quantile of empty sample")
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _bin_edges(reference: list[float], bins: int) -> list[float]:
    ordered = sorted(reference)
    edges = sorted({_quantile(ordered, i / bins) for i in range(1, bins)})
    return edges


def _proportions(values: list[float], edges: list[float]) -> list[float]:
    counts = [0] * (len(edges) + 1)
    for v in values:
        idx = 0
        while idx < len(edges) and v > edges[idx]:
            idx += 1
        counts[idx] += 1
    total = len(values) or 1
    return [c / total for c in counts]


def population_stability_index(expected: list[float], actual: list[float], bins: int = 10) -> float:
    """PSI between a reference (expected) and current (actual) sample.

    Rule of thumb: < 0.10 no significant shift, 0.10–0.25 moderate shift,
    > 0.25 major shift.
    """
    if not expected or not actual:
        raise ValueError("PSI requires non-empty samples")
    edges = _bin_edges(expected, bins)
    e_props = _proportions(expected, edges)
    a_props = _proportions(actual, edges)
    psi = 0.0
    for e, a in zip(e_props, a_props):
        e = max(e, _EPS)
        a = max(a, _EPS)
        psi += (a - e) * math.log(a / e)
    return psi


def ks_statistic(sample_a: list[float], sample_b: list[float]) -> float:
    """Two-sample Kolmogorov–Smirnov D statistic in [0, 1]."""
    if not sample_a or not sample_b:
        raise ValueError("KS requires non-empty samples")
    a = sorted(sample_a)
    b = sorted(sample_b)
    i = j = 0
    d = 0.0
    # Compare empirical CDFs at each distinct value, advancing both
    # pointers past ties so identical samples yield exactly 0.
    while i < len(a) and j < len(b):
        v = min(a[i], b[j])
        while i < len(a) and a[i] <= v:
            i += 1
        while j < len(b) and b[j] <= v:
            j += 1
        d = max(d, abs(i / len(a) - j / len(b)))
    return d


@dataclass
class DriftReport:
    feature: str
    psi: float
    ks: float
    severity: str  # ok | warning | critical


def _severity(psi: float, ks: float, psi_warn: float, psi_crit: float,
              ks_warn: float, ks_crit: float) -> str:
    if psi >= psi_crit or ks >= ks_crit:
        return "critical"
    if psi >= psi_warn or ks >= ks_warn:
        return "warning"
    return "ok"


def detect_drift(
    reference: dict[str, list[float]],
    current: dict[str, list[float]],
    psi_warn: float = 0.10,
    psi_critical: float = 0.25,
    ks_warn: float = 0.10,
    ks_critical: float = 0.20,
    bins: int = 10,
) -> list[DriftReport]:
    """Compare each feature in ``current`` against its reference distribution."""
    reports: list[DriftReport] = []
    for feature, current_values in current.items():
        ref_values = reference.get(feature)
        if ref_values is None:
            continue
        psi = population_stability_index(ref_values, current_values, bins=bins)
        ks = ks_statistic(ref_values, current_values)
        reports.append(DriftReport(
            feature=feature, psi=psi, ks=ks,
            severity=_severity(psi, ks, psi_warn, psi_critical, ks_warn, ks_critical),
        ))
    return sorted(reports, key=lambda r: r.psi, reverse=True)
