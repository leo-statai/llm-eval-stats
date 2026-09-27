from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def as_1d(values: ArrayLike, name: str = "scores") -> np.ndarray:
    """Convert to a finite 1-D float array, raising on empty input or NaNs."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def as_pair(a: ArrayLike, b: ArrayLike, names: Sequence[str] = ("a", "b")):
    """Convert two paired score vectors, checking they have the same length."""
    x, y = as_1d(a, names[0]), as_1d(b, names[1])
    if x.size != y.size:
        raise ValueError(
            f"paired inputs must have the same length: {names[0]}={x.size}, {names[1]}={y.size}"
        )
    return x, y


def is_binary(arr: np.ndarray) -> bool:
    return bool(np.all((arr == 0) | (arr == 1)))


def rng_from(seed: int | np.random.Generator | None) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
