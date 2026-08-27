#!/usr/bin/env python3
"""
Generic statistical helpers shared by this repository's human-evaluation
scoring scripts.

Moved out of evaluation/e4_pairwise_judging.py (the pairwise LLM-as-judge /
position-bias experiment, retired from the final thesis methodology) so that
still-valid scripts can depend on these two small, generic, judge-unrelated
functions without importing anything about pairwise judging. Behavior and
formulas are unchanged from the originals -- copied verbatim, not
reimplemented.

Used by:
  - evaluation/scripts/e5_final_stats.py
  - evaluation/scripts/run_baseline_v2_human_screening_validation.py

(evaluation/scripts/score_annotation.py also used to import this module, but
was removed during the final-repository cleanup -- it was never used for any
final thesis number.)
"""
from __future__ import annotations

from scipy.stats import beta as beta_dist


def binomial_exact_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson exact CI for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def cohens_kappa_multiclass(labels_a: list[str], labels_b: list[str]) -> float:
    categories = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(labels_a, labels_b) if x == y) / n
    ca = {c: labels_a.count(c) / n for c in categories}
    cb = {c: labels_b.count(c) / n for c in categories}
    pe = sum(ca[c] * cb[c] for c in categories)
    if pe == 1:
        return float("nan")
    return (po - pe) / (1 - pe)
