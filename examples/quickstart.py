"""End-to-end example on simulated eval data (no API keys needed).

Scenario: two LLMs answer the same 300 benchmark questions, and an LLM judge
rates 200 answers on a 1-5 scale that humans also rated.

    python examples/quickstart.py
"""

import numpy as np
from scipy import stats

from llm_eval_stats import (
    bootstrap_ci,
    compare_paired,
    judge_agreement,
    paired_sample_size,
)

rng = np.random.default_rng(0)

# --- 1. Two models on the same 300 questions -------------------------------
# Correctness depends on a shared question difficulty, as in real benchmarks.
n = 300
difficulty = rng.normal(size=n)
scale = np.sqrt(1 + 0.6**2)  # sd of difficulty + noise
model_a = (difficulty + rng.normal(scale=0.6, size=n) < scale * stats.norm.ppf(0.72)).astype(int)
model_b = (difficulty + rng.normal(scale=0.6, size=n) < scale * stats.norm.ppf(0.76)).astype(int)

print("1) Accuracy with 95% confidence intervals")
print("   model A:", bootstrap_ci(model_a, seed=0))
print("   model B:", bootstrap_ci(model_b, seed=0))

print("\n2) Is B really better than A? (paired, McNemar)")
result = compare_paired(model_b, model_a, seed=0)
print("  ", result)
print("   significant at 5%:", result.significant())

# --- 2. How many questions would we need? ---------------------------------
discordance = np.mean(model_a != model_b)
needed = paired_sample_size(delta=0.04, discordance=discordance)
print(
    f"\n3) Items needed to detect a 4-point gap with 80% power "
    f"(observed discordance {discordance:.2f}): {needed}"
)

# --- 3. Can we trust the LLM judge? ---------------------------------------
human = rng.integers(1, 6, size=200)
judge = np.clip(human + rng.choice([-1, 0, 0, 0, 1, 1], size=200), 1, 5)  # a bit lenient

print("\n4) LLM judge vs human ratings")
print(judge_agreement(judge, human, seed=0))
