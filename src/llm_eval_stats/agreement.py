"""Agreement between an LLM judge and human raters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from ._utils import as_pair, rng_from
from .intervals import Estimate, bootstrap_ci

WEIGHTS = (None, "linear", "quadratic")


def cohen_kappa(
    rater_a: ArrayLike,
    rater_b: ArrayLike,
    *,
    weights: str | None = None,
    labels: Sequence[float] | None = None,
) -> float:
    """Cohen's kappa: agreement between two raters corrected for chance.

    Use ``weights="quadratic"`` (or ``"linear"``) for ordinal scales such as
    1-5 ratings, so that a 4-vs-5 disagreement costs less than a 1-vs-5 one.
    ``labels`` fixes the category set and order (defaults to the sorted union
    of observed values).
    """
    if weights not in WEIGHTS:
        raise ValueError(f"weights must be one of {WEIGHTS}, got {weights!r}")
    x, y = as_pair(rater_a, rater_b, ("rater_a", "rater_b"))
    cats = np.unique(np.concatenate([x, y])) if labels is None else np.asarray(labels, float)
    k = cats.size
    if k < 2:
        return 1.0

    index = {c: i for i, c in enumerate(cats)}
    try:
        xi = np.array([index[v] for v in x])
        yi = np.array([index[v] for v in y])
    except KeyError as err:
        raise ValueError(f"rating {err.args[0]} not in labels") from None

    observed = np.zeros((k, k))
    np.add.at(observed, (xi, yi), 1)
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))

    dist = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    if weights is None:
        w = (dist > 0).astype(float)
    elif weights == "linear":
        w = dist
    else:
        w = dist**2

    expected_disagreement = np.sum(w * expected)
    if expected_disagreement == 0:
        # Both raters used a single, identical category: agreement is trivial.
        return 1.0
    return float(1 - np.sum(w * observed) / expected_disagreement)


@dataclass(frozen=True)
class AgreementReport:
    """How well an LLM judge reproduces human ratings on the same items."""

    n: int
    exact_agreement: Estimate
    kappa: Estimate
    spearman: float
    bias: Estimate

    def __str__(self) -> str:
        return (
            f"n = {self.n}\n"
            f"exact agreement : {self.exact_agreement}\n"
            f"kappa           : {self.kappa}\n"
            f"spearman rho    : {self.spearman:.3f}\n"
            f"bias (judge-human): {self.bias}"
        )


def judge_agreement(
    judge: ArrayLike,
    human: ArrayLike,
    *,
    weights: str | None = "quadratic",
    confidence: float = 0.95,
    n_resamples: int = 5_000,
    seed: int | np.random.Generator | None = None,
) -> AgreementReport:
    """Validate an LLM-as-judge against human ratings of the same items.

    Reports exact agreement, (weighted) Cohen's kappa with a bootstrap CI,
    Spearman correlation and the mean bias judge - human (LLM judges are often
    systematically lenient). Kappa is bootstrapped over items with the category
    set held fixed.
    """
    x, y = as_pair(judge, human, ("judge", "human"))
    rng = rng_from(seed)
    labels = np.unique(np.concatenate([x, y]))

    kappa_value = cohen_kappa(x, y, weights=weights, labels=labels)
    if np.all(x == y):
        kappa = Estimate(kappa_value, kappa_value, kappa_value, confidence, x.size, "percentile")
    else:
        res = stats.bootstrap(
            (x, y),
            lambda a, b: cohen_kappa(a, b, weights=weights, labels=labels),
            paired=True,
            vectorized=False,
            confidence_level=confidence,
            n_resamples=n_resamples,
            method="percentile",
            rng=rng,
        )
        ci = res.confidence_interval
        kappa = Estimate(
            kappa_value, float(ci.low), float(ci.high), confidence, x.size, "percentile"
        )

    boot = dict(confidence=confidence, n_resamples=n_resamples, seed=rng)
    exact = bootstrap_ci((x == y).astype(float), **boot)
    bias = bootstrap_ci(x - y, **boot)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        rho = float("nan")
    else:
        rho = float(stats.spearmanr(x, y).statistic)
    return AgreementReport(x.size, exact, kappa, rho, bias)
