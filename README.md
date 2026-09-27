# llm-eval-stats

**Statistically sound LLM evaluation: confidence intervals, paired model comparisons, LLM-judge agreement and sample-size planning.**

[![CI](https://github.com/leo-statai/llm-eval-stats/actions/workflows/ci.yml/badge.svg)](https://github.com/leo-statai/llm-eval-stats/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

Model B scores 75.7% on your benchmark and model A scores 71.7%. Is B better, or is that noise?

Most LLM leaderboards and internal evals report a single number per model. `llm-eval-stats` adds the statistics that tell you how much to trust that number, using only NumPy and SciPy.

## What it does

| Question | Function | Method |
|---|---|---|
| How precise is this accuracy / mean score? | `bootstrap_ci`, `proportion_ci` | BCa bootstrap; Wilson interval for pass rates |
| Is model B really better than model A? | `compare_paired` | Exact McNemar (0/1), paired permutation or Wilcoxon, plus a paired bootstrap CI of the difference |
| Which of my *k* models differ? | `compare_models` | All pairs or vs. a baseline, Holm-adjusted p-values |
| Can I trust my LLM-as-judge? | `judge_agreement`, `cohen_kappa` | Weighted Cohen's kappa with bootstrap CI, Spearman, judge-minus-human bias |
| How many eval items do I need? | `paired_sample_size`, `sample_size_for_margin` | McNemar power analysis (Connor, 1987); margin of error |

Every comparison is **paired**: both models answer the same items, and treating the two score vectors as independent samples ignores that some questions are hard for every model, which wastes statistical power.

## Install

```bash
pip install git+https://github.com/leo-statai/llm-eval-stats
```

## Quickstart

```python
from llm_eval_stats import bootstrap_ci, compare_paired, judge_agreement, paired_sample_size

# One entry per benchmark question: 1 = correct, 0 = wrong (same questions, same order).
bootstrap_ci(model_a)                 # 0.717 [0.667, 0.767] (95% CI, n=300, BCa)
bootstrap_ci(model_b)                 # 0.757 [0.707, 0.803] (95% CI, n=300, BCa)

result = compare_paired(model_b, model_a)
print(result)                         # diff = 0.040 [-0.007, 0.090]; p = 0.1337 (mcnemar)
result.significant()                  # False: a 4-point gap on 300 items is not conclusive

# How many questions would it take to detect a 4-point gap with 80% power?
paired_sample_size(delta=0.04, discordance=0.18)   # 881

# LLM judge vs human ratings on a 1-5 scale
print(judge_agreement(judge_scores, human_scores))
# exact agreement : 0.595 [0.525, 0.660]
# kappa           : 0.904 [0.882, 0.923]   (quadratic-weighted)
# spearman rho    : 0.901
# bias (judge-human): 0.105 [0.020, 0.195]  -> the judge is slightly lenient
```

The full runnable version, on simulated data and with no API keys needed, is in [`examples/quickstart.py`](examples/quickstart.py).

## Reading the results

- **Report the interval, not only the point.** Two models with overlapping 95% CIs are not necessarily tied, which is why `compare_paired` tests the *paired difference* directly.
- **Discordance drives sample size.** What matters is how often exactly one of the two models is right. Measure it on a pilot run, then use `paired_sample_size` before building a larger benchmark.
- **Correct for multiple comparisons.** Comparing 5 models pairwise means 10 tests. `compare_models` applies Holm's correction so a "winner" isn't declared by chance.
- **Validate the judge before trusting it.** High correlation can hide a systematic bias; `judge_agreement` reports both.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

Kappa values are cross-checked against `scikit-learn`, and bootstrap coverage is checked by simulation.

## Roadmap

- [ ] Krippendorff's alpha for multiple human raters
- [ ] Clustered bootstrap for items grouped by source document or prompt template
- [ ] Pandas / JSONL helpers for common eval-harness outputs
- [ ] CLI: `llm-eval-stats compare results_a.jsonl results_b.jsonl`

## License

MIT © Leonardo Alves
