"""Bias / fairness metrics over grouped predictions.

Implements the metrics model-risk and responsible-AI reviews actually ask
for on binary classifiers:

- Demographic parity difference — spread of positive rates across groups.
- Disparate impact — min/max positive-rate ratio (the "four-fifths rule":
  below 0.80 is the classic adverse-impact threshold).
- Equal opportunity difference — spread of true-positive rates across groups.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _group_indices(groups: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        out.setdefault(g, []).append(i)
    return out


def group_positive_rates(y_pred: list[int], groups: list[str]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for g, idxs in _group_indices(groups).items():
        rates[g] = sum(y_pred[i] for i in idxs) / len(idxs)
    return rates


def demographic_parity_difference(y_pred: list[int], groups: list[str]) -> float:
    rates = group_positive_rates(y_pred, groups)
    if not rates:
        return 0.0
    return max(rates.values()) - min(rates.values())


def disparate_impact_ratio(y_pred: list[int], groups: list[str]) -> float:
    rates = group_positive_rates(y_pred, groups)
    if not rates:
        return 1.0
    hi = max(rates.values())
    if hi == 0:
        return 1.0
    return min(rates.values()) / hi


def equal_opportunity_difference(y_true: list[int], y_pred: list[int], groups: list[str]) -> float:
    tprs: list[float] = []
    for _g, idxs in _group_indices(groups).items():
        positives = [i for i in idxs if y_true[i] == 1]
        if not positives:
            continue
        tprs.append(sum(y_pred[i] for i in positives) / len(positives))
    if len(tprs) < 2:
        return 0.0
    return max(tprs) - min(tprs)


@dataclass
class FairnessReport:
    positive_rates: dict[str, float]
    demographic_parity_difference: float
    disparate_impact: float
    equal_opportunity_difference: float | None
    breaches: list[str] = field(default_factory=list)


def evaluate_fairness(
    y_pred: list[int],
    groups: list[str],
    y_true: list[int] | None = None,
    dpd_warning: float = 0.10,
    di_minimum: float = 0.80,
    eod_warning: float = 0.10,
) -> FairnessReport:
    """Evaluate fairness of one prediction batch and list threshold breaches."""
    if len(y_pred) != len(groups):
        raise ValueError("y_pred and groups must have equal length")
    rates = group_positive_rates(y_pred, groups)
    dpd = demographic_parity_difference(y_pred, groups)
    di = disparate_impact_ratio(y_pred, groups)
    eod = None
    if y_true is not None:
        eod = equal_opportunity_difference(y_true, y_pred, groups)
    breaches: list[str] = []
    if dpd > dpd_warning:
        breaches.append(f"demographic parity difference {dpd:.3f} > {dpd_warning}")
    if di < di_minimum:
        breaches.append(f"disparate impact {di:.3f} < {di_minimum} (four-fifths rule)")
    if eod is not None and eod > eod_warning:
        breaches.append(f"equal opportunity difference {eod:.3f} > {eod_warning}")
    return FairnessReport(
        positive_rates=rates, demographic_parity_difference=dpd,
        disparate_impact=di, equal_opportunity_difference=eod, breaches=breaches,
    )
