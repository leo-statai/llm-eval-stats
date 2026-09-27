import numpy as np
import pytest

from llm_eval_stats import bootstrap_ci, proportion_ci


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    scores = rng.normal(3.5, 1.0, size=200)
    est = bootstrap_ci(scores, seed=1)
    assert est.low < est.value < est.high
    assert est.value == pytest.approx(scores.mean())
    assert est.n == 200


def test_bootstrap_ci_is_reproducible_with_seed():
    scores = np.random.default_rng(0).integers(0, 2, size=100)
    assert bootstrap_ci(scores, seed=42) == bootstrap_ci(scores, seed=42)


def test_bootstrap_ci_constant_scores_collapse():
    est = bootstrap_ci([1, 1, 1, 1])
    assert est.low == est.value == est.high == 1.0


def test_bootstrap_ci_coverage_is_close_to_nominal():
    # Repeated experiments: the 95% interval should cover the true mean ~95% of the time.
    rng = np.random.default_rng(123)
    true_p, hits, runs = 0.7, 0, 200
    for _ in range(runs):
        est = bootstrap_ci(rng.binomial(1, true_p, size=150), n_resamples=2000, seed=rng)
        hits += est.low <= true_p <= est.high
    assert 0.88 <= hits / runs <= 0.99


def test_proportion_ci_wilson_known_values():
    # 8/10 successes, Wilson 95% CI ≈ [0.490, 0.943]
    est = proportion_ci([1] * 8 + [0] * 2)
    assert est.value == pytest.approx(0.8)
    assert est.low == pytest.approx(0.4902, abs=1e-3)
    assert est.high == pytest.approx(0.9433, abs=1e-3)


def test_proportion_ci_rejects_non_binary():
    with pytest.raises(ValueError):
        proportion_ci([0, 1, 2])


@pytest.mark.parametrize("bad", [[], [[1, 2], [3, 4]], [1.0, float("nan")]])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        bootstrap_ci(bad)
