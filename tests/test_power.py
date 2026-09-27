import pytest

from llm_eval_stats import margin_of_error, paired_sample_size, sample_size_for_margin


def test_margin_roundtrip():
    n = sample_size_for_margin(0.03)
    assert n == 1068
    assert margin_of_error(n) <= 0.03 < margin_of_error(n - 1)


def test_paired_sample_size_known_value():
    # Connor (1987): delta = 0.05, discordance = 0.20, alpha = 0.05, power = 0.8
    # n = (1.960*sqrt(0.2) + 0.8416*sqrt(0.2 - 0.0025))^2 / 0.0025 ≈ 625.5
    assert paired_sample_size(0.05, 0.20) == 626


def test_more_discordance_needs_more_items():
    assert paired_sample_size(0.03, 0.30) > paired_sample_size(0.03, 0.10)


def test_paired_sample_size_validates():
    with pytest.raises(ValueError):
        paired_sample_size(0.2, 0.1)
