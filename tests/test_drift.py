import random

import pytest

from mlplatform.monitoring.drift import detect_drift, ks_statistic, population_stability_index

random.seed(7)


def gauss(mean, sd, n):
    return [random.gauss(mean, sd) for _ in range(n)]


def test_psi_near_zero_for_same_distribution():
    ref = gauss(10, 2, 2000)
    cur = gauss(10, 2, 2000)
    assert population_stability_index(ref, cur) < 0.05


def test_psi_large_for_shifted_distribution():
    ref = gauss(10, 2, 2000)
    cur = gauss(16, 2, 2000)
    assert population_stability_index(ref, cur) > 0.25


def test_ks_bounds_and_shift():
    a = gauss(0, 1, 1000)
    assert ks_statistic(a, list(a)) == 0.0
    assert ks_statistic(a, gauss(3, 1, 1000)) > 0.5


def test_detect_drift_classifies_severity():
    reference = {"x": gauss(0, 1, 1500), "stable": gauss(5, 1, 1500)}
    current = {"x": gauss(4, 1, 1500), "stable": gauss(5, 1, 1500)}
    reports = {r.feature: r for r in detect_drift(reference, current)}
    assert reports["x"].severity == "critical"
    assert reports["stable"].severity == "ok"


def test_empty_samples_rejected():
    with pytest.raises(ValueError):
        population_stability_index([], [1.0])
    with pytest.raises(ValueError):
        ks_statistic([1.0], [])
