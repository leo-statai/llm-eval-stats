"""Statistically sound LLM evaluation."""

from .agreement import AgreementReport, cohen_kappa, judge_agreement
from .compare import Comparison, PairwiseResult, compare_models, compare_paired, holm
from .intervals import Estimate, bootstrap_ci, proportion_ci
from .power import margin_of_error, paired_sample_size, sample_size_for_margin

__version__ = "0.1.0"

__all__ = [
    "AgreementReport",
    "Comparison",
    "Estimate",
    "PairwiseResult",
    "bootstrap_ci",
    "cohen_kappa",
    "compare_models",
    "compare_paired",
    "holm",
    "judge_agreement",
    "margin_of_error",
    "paired_sample_size",
    "proportion_ci",
    "sample_size_for_margin",
]
