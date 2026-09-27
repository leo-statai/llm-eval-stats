"""Paired comparisons between models evaluated on the same items."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from ._utils import as_1d, as_pair, is_binary, rng_from
from .intervals import Estimate, bootstrap_ci

TESTS = ("auto", "mcnemar", "permutation", "wilcoxon")


@dataclass(frozen=True)
class Comparison:
    """Result of comparing model A against model B on the same items.

    ``difference`` is mean(A) - mean(B) with a paired bootstrap CI.
    """

    difference: Estimate
    p_value: float
    test: str
    n: int

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def __str__(self) -> str:
        return f"diff = {self.difference}; p = {self.p_value:.4g} ({self.test})"


@dataclass(frozen=True)
class PairwiseResult:
    model_a: str
    model_b: str
    comparison: Comparison
    p_adjusted: float

    def __str__(self) -> str:
        return (
            f"{self.model_a} vs {self.model_b}: {self.comparison.difference}; "
            f"p = {self.comparison.p_value:.4g}, adjusted p = {self.p_adjusted:.4g}"
        )


def _mean_diff(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.mean(x - y, axis=axis)


def compare_paired(
    a: ArrayLike,
    b: ArrayLike,
    *,
    test: str = "auto",
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | np.random.Generator | None = None,
) -> Comparison:
    """Compare two models scored on the *same* items, respecting the pairing.

    Treating the two score vectors as independent samples ignores that some
    items are hard for every model, which inflates the variance and hides real
    differences. All tests here work on per-item pairs instead.

    ``test``:
        - ``"mcnemar"``: exact McNemar test, for 0/1 correctness.
        - ``"permutation"``: paired sign-flip permutation test of the mean difference.
        - ``"wilcoxon"``: Wilcoxon signed-rank test.
        - ``"auto"``: McNemar for binary scores, permutation otherwise.
    """
    if test not in TESTS:
        raise ValueError(f"test must be one of {TESTS}, got {test!r}")
    x, y = as_pair(a, b)
    rng = rng_from(seed)
    binary = is_binary(x) and is_binary(y)
    if test == "auto":
        test = "mcnemar" if binary else "permutation"

    if test == "mcnemar":
        if not binary:
            raise ValueError("McNemar's test requires 0/1 scores")
        only_a = int(np.sum((x == 1) & (y == 0)))
        only_b = int(np.sum((x == 0) & (y == 1)))
        discordant = only_a + only_b
        p = 1.0 if discordant == 0 else stats.binomtest(only_a, discordant, 0.5).pvalue
    elif test == "permutation":
        if np.all(x == y):
            p = 1.0
        else:
            p = stats.permutation_test(
                (x, y),
                _mean_diff,
                permutation_type="samples",
                vectorized=True,
                n_resamples=n_resamples,
                rng=rng,
            ).pvalue
    else:
        p = 1.0 if np.all(x == y) else stats.wilcoxon(x, y).pvalue

    diff = bootstrap_ci(x - y, confidence=confidence, n_resamples=n_resamples, seed=rng)
    return Comparison(diff, float(p), test, x.size)


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.minimum(1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    out = np.empty(m)
    out[order] = adjusted
    return out


def compare_models(
    scores: Mapping[str, ArrayLike],
    *,
    baseline: str | None = None,
    test: str = "auto",
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | np.random.Generator | None = None,
) -> list[PairwiseResult]:
    """Compare several models on the same items, with Holm-adjusted p-values.

    With ``baseline`` every other model is compared against it; otherwise all
    pairs are compared. Running many pairwise tests without correction makes a
    "significant" winner likely by chance alone.
    """
    names = list(scores)
    if len(names) < 2:
        raise ValueError("need at least two models to compare")
    arrays = {name: as_1d(scores[name], name) for name in names}
    if baseline is not None:
        if baseline not in arrays:
            raise KeyError(f"baseline {baseline!r} not in scores")
        pairs = [(name, baseline) for name in names if name != baseline]
    else:
        pairs = list(combinations(names, 2))

    rng = rng_from(seed)
    comparisons = [
        compare_paired(
            arrays[a],
            arrays[b],
            test=test,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=rng,
        )
        for a, b in pairs
    ]
    adjusted = holm([c.p_value for c in comparisons])
    return [
        PairwiseResult(a, b, c, float(p_adj))
        for (a, b), c, p_adj in zip(pairs, comparisons, adjusted, strict=True)
    ]
