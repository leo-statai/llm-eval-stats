"""Sample-size planning: how many eval items do you actually need?"""

from __future__ import annotations

import math

from scipy import stats


def _z(q: float) -> float:
    return float(stats.norm.ppf(q))


def margin_of_error(n: int, *, p: float = 0.5, confidence: float = 0.95) -> float:
    """Half-width of a normal-approximation CI for an accuracy measured on ``n`` items.

    ``p = 0.5`` gives the worst case.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return _z(1 - (1 - confidence) / 2) * math.sqrt(p * (1 - p) / n)


def sample_size_for_margin(margin: float, *, p: float = 0.5, confidence: float = 0.95) -> int:
    """Items needed so an accuracy estimate has at most ``margin`` half-width."""
    if not 0 < margin < 1:
        raise ValueError("margin must be in (0, 1)")
    z = _z(1 - (1 - confidence) / 2)
    return math.ceil(z**2 * p * (1 - p) / margin**2)


def paired_sample_size(
    delta: float,
    discordance: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Items needed to detect an accuracy gap between two models (McNemar test).

    Parameters
    ----------
    delta:
        The accuracy difference you want to detect, e.g. ``0.03`` for 3 points.
    discordance:
        Expected share of items where exactly one model is correct. This is
        what drives the sample size: two models that fail on the *same* items
        are easy to tell apart; models that disagree often need many more items.
        A pilot run on a few hundred items gives a good estimate.

    Uses Connor's (1987) normal approximation for the paired design.
    """
    if not 0 < delta <= discordance <= 1:
        raise ValueError("need 0 < delta <= discordance <= 1")
    za = _z(1 - alpha / 2)
    zb = _z(power)
    n = (za * math.sqrt(discordance) + zb * math.sqrt(discordance - delta**2)) ** 2 / delta**2
    return math.ceil(n)
