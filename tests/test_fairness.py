import pytest

from mlplatform.monitoring.fairness import (
    demographic_parity_difference, disparate_impact_ratio,
    equal_opportunity_difference, evaluate_fairness,
)


def test_balanced_predictions_are_fair():
    # Pairs (pred, group) built so both groups see identical prediction patterns.
    pairs = [(1, "a"), (1, "b"), (0, "a"), (0, "b")] * 25
    y_pred = [p for p, _g in pairs]
    groups = [g for _p, g in pairs]
    assert demographic_parity_difference(y_pred, groups) == 0.0
    assert disparate_impact_ratio(y_pred, groups) == 1.0
    report = evaluate_fairness(y_pred, groups)
    assert report.breaches == []


def test_skewed_predictions_breach_four_fifths_rule():
    groups = ["a"] * 100 + ["b"] * 100
    y_pred = [1] * 80 + [0] * 20 + [1] * 20 + [0] * 80
    assert disparate_impact_ratio(y_pred, groups) == pytest.approx(0.25)
    assert demographic_parity_difference(y_pred, groups) == pytest.approx(0.60)
    report = evaluate_fairness(y_pred, groups)
    assert any("four-fifths" in b for b in report.breaches)


def test_equal_opportunity_difference():
    groups = ["a"] * 4 + ["b"] * 4
    y_true = [1, 1, 0, 0, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 0, 1, 0, 0, 0]  # group a TPR 1.0, group b TPR 0.5
    assert equal_opportunity_difference(y_true, y_pred, groups) == pytest.approx(0.5)
    report = evaluate_fairness(y_pred, groups, y_true=y_true)
    assert any("equal opportunity" in b for b in report.breaches)
