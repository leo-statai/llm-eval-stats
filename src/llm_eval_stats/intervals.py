"""Confidence intervals for a single model's evaluation metric."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from ._utils import as_1d, is_binary, rng_from


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a confidence interval."""

    value: float
    low: float
    high: float
    confidence: float
    n: int
    method: str

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2

    def __str__(self) -> str:
        return (
            f"{self.value:.3f} [{self.low:.3f}, {self.high:.3f}] "
            f"({self.confidence:.0%} CI, n={self.n}, {self.method})"
        )


def bootstrap_ci(
    scores: ArrayLike,
    statistic: Callable[..., float] = np.mean,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    method: str = "BCa",
    seed: int | np.random.Generator | None = None,
) -> Estimate:
    """Bootstrap confidence interval for any statistic of per-item scores.

    ``scores`` holds one value per evaluation item (e.g. 0/1 correctness, a 1-5
    judge rating, a BLEU score). The default statistic is the mean, i.e. the
    benchmark score. ``method`` is passed to :func:`scipy.stats.bootstrap`
    (``"BCa"``, ``"percentile"`` or ``"basic"``).
    """
    x = as_1d(scores)
    value = float(statistic(x))
    if np.ptp(x) == 0:
        # Every resample is identical, so the interval collapses to the point.
        return Estimate(value, value, value, confidence, x.size, method)
    res = stats.bootstrap(
        (x,),
        statistic,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method=method,
        rng=rng_from(seed),
    )
    ci = res.confidence_interval
    return Estimate(value, float(ci.low), float(ci.high), confidence, x.size, method)


def proportion_ci(
    outcomes: ArrayLike,
    *,
    confidence: float = 0.95,
    method: str = "wilson",
) -> Estimate:
    """Closed-form interval for a pass rate / accuracy from 0/1 outcomes.

    Wilson intervals behave well for small ``n`` and for rates near 0 or 1,
    where the naive normal approximation breaks down.
    """
    x = as_1d(outcomes, "outcomes")
    if not is_binary(x):
        raise ValueError("outcomes must contain only 0 and 1")
    k, n = int(x.sum()), x.size
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=confidence, method=method)
    return Estimate(k / n, float(ci.low), float(ci.high), confidence, n, method)
