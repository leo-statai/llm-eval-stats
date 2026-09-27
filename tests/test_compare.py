import numpy as np
import pytest
from scipy import stats

from llm_eval_stats import compare_models, compare_paired, holm


def correlated_models(n=400, acc_a=0.70, acc_b=0.78, seed=0):
    """Two models whose correctness depends on a shared item difficulty."""
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(size=n)
    noise = rng.normal(scale=0.7, size=(2, n))
    a = (difficulty + noise[0] < stats.norm.ppf(acc_a)).astype(int)
    b = (difficulty + noise[1] < stats.norm.ppf(acc_b)).astype(int)
    return a, b


def test_mcnemar_matches_exact_binomial():
    a, b = correlated_models()
    res = compare_paired(a, b, seed=0)
    only_a = int(np.sum((a == 1) & (b == 0)))
    only_b = int(np.sum((a == 0) & (b == 1)))
    expected = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue
    assert res.test == "mcnemar"
    assert res.p_value == pytest.approx(expected)
    assert res.difference.value == pytest.approx(a.mean() - b.mean())


def test_detects_real_difference():
    a, b = correlated_models(acc_a=0.65, acc_b=0.80)
    res = compare_paired(a, b, seed=0)
    assert res.significant()
    assert res.difference.high < 0


def test_identical_models_are_not_different():
    a, _ = correlated_models()
    for test in ("mcnemar", "permutation", "wilcoxon"):
        res = compare_paired(a, a, test=test, seed=0)
        assert res.p_value == 1.0
        assert res.difference.value == 0.0


def test_auto_uses_permutation_for_continuous_scores():
    rng = np.random.default_rng(1)
    base = rng.normal(size=100)
    res = compare_paired(base + 0.5, base + rng.normal(scale=0.3, size=100), seed=0)
    assert res.test == "permutation"
    assert res.significant()


def test_mcnemar_rejects_continuous_scores():
    with pytest.raises(ValueError):
        compare_paired([0.2, 0.4], [0.1, 0.9], test="mcnemar")


def test_unequal_lengths_raise():
    with pytest.raises(ValueError):
        compare_paired([1, 0, 1], [1, 0])


def test_holm_known_example():
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02])


def test_compare_models_against_baseline():
    a, b = correlated_models(seed=1)
    c, _ = correlated_models(seed=2)
    results = compare_models({"base": a, "b": b, "c": c}, baseline="base", seed=0)
    assert [(r.model_a, r.model_b) for r in results] == [("b", "base"), ("c", "base")]
    for r in results:
        assert r.p_adjusted >= r.comparison.p_value


def test_compare_models_all_pairs():
    a, b = correlated_models()
    results = compare_models({"x": a, "y": b, "z": a}, seed=0)
    assert len(results) == 3
