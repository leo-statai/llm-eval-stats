import numpy as np
import pytest

from llm_eval_stats import cohen_kappa, judge_agreement


def test_kappa_textbook_example():
    # 2x2 table [[20, 5], [10, 15]]: po = 0.70, pe = 0.50 -> kappa = 0.40
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_kappa_perfect_and_symmetric():
    x = [1, 2, 3, 4, 5, 3, 2]
    assert cohen_kappa(x, x) == pytest.approx(1.0)
    y = [1, 2, 3, 5, 5, 2, 2]
    assert cohen_kappa(x, y, weights="quadratic") == pytest.approx(
        cohen_kappa(y, x, weights="quadratic")
    )


def test_weighted_kappa_penalises_near_misses_less():
    human = np.array([1, 2, 3, 4, 5] * 20)
    near = np.clip(human + 1, 1, 5)
    far = 6 - human
    assert cohen_kappa(near, human, weights="quadratic") > cohen_kappa(near, human)
    assert cohen_kappa(near, human, weights="quadratic") > cohen_kappa(
        far, human, weights="quadratic"
    )


@pytest.mark.parametrize(
    ("weights", "expected"),
    # Reference values from sklearn.metrics.cohen_kappa_score
    [(None, 0.625), ("linear", 0.810126582278481), ("quadratic", 0.9230769230769231)],
)
def test_kappa_matches_sklearn(weights, expected):
    a = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    b = [1, 2, 3, 5, 5, 2, 2, 3, 3, 5]
    assert cohen_kappa(a, b, weights=weights) == pytest.approx(expected, abs=1e-12)


def test_kappa_unknown_label_raises():
    with pytest.raises(ValueError):
        cohen_kappa([1, 2], [1, 3], labels=[1, 2])


def test_judge_agreement_detects_lenient_judge():
    rng = np.random.default_rng(0)
    human = rng.integers(1, 6, size=300)
    judge = np.clip(human + rng.choice([0, 0, 1], size=300), 1, 5)
    report = judge_agreement(judge, human, n_resamples=1000, seed=0)
    assert report.n == 300
    assert report.bias.low > 0
    assert 0 < report.kappa.low <= report.kappa.value <= report.kappa.high <= 1
    assert report.spearman > 0.8
