# E2 — Internal consistency checks (v2)

**Conditions covered:** pipeline, b1, baseline_v2 — five cities, fifteen reports. "Baseline" always means `baseline_v2` here; the retired baseline is not reported (see chapter4, "An earlier baseline").

**Source files (not recomputed for this document):**
- `evaluation/results/e2_summary_v2.csv` — per-(city, system) word counts, raw flag counts, raw defect rate + Poisson CI
- `evaluation/results/e2_flags_v2.csv` — individual flags with hand-verified `VERDICT` (TP/FP) — **pipeline and b1 rows only**; no baseline or baseline_v2 rows exist in this file because zero flags fired on either baseline condition (see §3)

Four checks: **CS-1** (count/percentage pair vs. stated total, off by >0.55pp), **CS-2** (majority/most/over half/nearly all applied to a value ≤50% of total), **CS-4** (aggregate "…account for N%" vs. constituent sum), **CS-5** (claim resting on an invented population/per-capita/census basis). CS-3 (cross-section contradiction) was prototyped and rejected — it fired 121 times on an arithmetically clean report, meaning it was keying on stylistic rephrasing, not real contradiction; not implemented (see `evaluation/e2_internal_consistency.py`'s `CS3_REJECTION_NOTE` constant for the full rejection rationale, which is unaffected by the baseline version and still current).

CS-1/CS-4 were widened after the original run to match count/percentage phrasings baseline-style prose actually uses (e.g. "19 of 47 retained initiatives (40.4%)"), not only the pipeline's "N (P%)" form. The widening left pipeline's own flag count unchanged (17 → 17, same 11 TP / 6 FP) — see `e2_widen_old_vs_new_summary.csv`.

## 1. Raw flags (unverified) per city, per system

| city | system | n_words | n_flags | CS-1 | CS-2 | CS-4 | CS-5 | raw defects/1,000 words |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| barcelona | pipeline | 1308 | 1 | 0 | 0 | 0 | 1 | 0.765 |
| barcelona | b1 | 1287 | 3 | 0 | 2 | 0 | 1 | 2.331 |
| barcelona | baseline_v2 | 1764 | 0 | 0 | 0 | 0 | 0 | 0.000 |
| brighton | pipeline | 1192 | 2 | 0 | 0 | 0 | 2 | 1.678 |
| brighton | b1 | 1357 | 3 | 0 | 0 | 0 | 3 | 2.211 |
| brighton | baseline_v2 | 1929 | 0 | 0 | 0 | 0 | 0 | 0.000 |
| dublin | pipeline | 1255 | 2 | 0 | 1 | 0 | 1 | 1.594 |
| dublin | b1 | 1218 | 2 | 0 | 1 | 0 | 1 | 1.642 |
| dublin | baseline_v2 | 2037 | 0 | 0 | 0 | 0 | 0 | 0.000 |
| london | pipeline | 1244 | 6 | 0 | 1 | 0 | 5 | 4.823 |
| london | b1 | 1259 | 3 | 1 | 0 | 0 | 2 | 2.383 |
| london | baseline_v2 | 1726 | 0 | 0 | 0 | 0 | 0 | 0.000 |
| milan | pipeline | 1319 | 6 | 0 | 3 | 1 | 2 | 4.549 |
| milan | b1 | 1260 | 4 | 0 | 1 | 1 | 2 | 3.175 |
| milan | baseline_v2 | 1639 | 0 | 0 | 0 | 0 | 0 | 0.000 |

Pooled raw: **pipeline** 17/6318 words = 2.691/1,000; **b1** 15/6381 words = 2.351/1,000; **baseline_v2** 0/9095 words = 0.000/1,000.

## 2. Verified (VERDICT=TP only) — the number to cite

Automated flags are candidates, not defects; every flag in `e2_flags_v2.csv` was hand-checked. Rates below use TP only, exact Poisson 95% CI (Garwood method, same as `e2_internal_consistency.poisson_exact_ci`).

| city | pipeline TP/flags | pipeline verified/1,000w | b1 TP/flags | b1 verified/1,000w |
|---|---|---:|---|---:|
| barcelona | 0/1 | 0.000 | 0/3 | 0.000 |
| brighton | 0/2 | 0.000 | 0/3 | 0.000 |
| dublin | 1/2 | 0.797 | 1/2 | 0.821 |
| london | 6/6 | 4.823 | 0/3 | 0.000 |
| milan | 4/6 | 3.033 | 2/4 | 1.587 |
| **pooled** | **11/17** | **1.741 (95% CI 0.869–3.115)** | **3/15** | **0.470 (95% CI 0.097–1.374)** |

**baseline_v2: 0 flags in any city → 0 verified defects → 0.000/1,000 words (95% CI 0.000–0.406, pooled).** This is reported as a genuine, applicable zero — not a case where the checks don't apply. See §3 for why it is empirically, not just structurally, zero.

## 3. Why baseline_v2 scores zero on every check — read this before citing the zero as "cleaner than pipeline"

This is not a case of "the measurement doesn't apply" in the same sense as E1-geo's baseline_v2 gap (§E1-geo of `E1_SUMMARY_v2.md`) — CS-1/CS-4 were specifically widened so they *can* structurally match baseline-style phrasing, and after widening they still found zero qualifying flags in any baseline_v2 report. Three different reasons combine:

- **CS-5 has close to nothing to key on.** Its forbidden-basis phrases ("per capita," "census population," "per 1,000 residents") are artefacts of the pipeline's own Phase-5 `{data_points}` boilerplate (`src/phase_5/text_synthesizer.py:38,109`), which explicitly hands the generator a "Data gap: … requires census population data" line — CS-5 fires when the model echoes or elaborates on that exact phrase. `baseline_v2`'s prompt has no equivalent boilerplate to echo, and the thesis's own characterisation of the baseline (§4.5.2) states it "refused to invent spatial data" — consistent with zero CS-5 hits, but not a structural guarantee against one.
- **CS-2's keyword pattern (majority/most/over half/nearly all near a value ≤50% of total) could in principle fire on baseline_v2 prose** — nothing about the check is pipeline-specific — but none of the 5 baseline_v2 reports happened to use that phrasing incorrectly.
- **CS-1/CS-4 (count/percentage arithmetic) did find matchable phrasings after widening** (the motivating example in the widening script's docstring, "19 of 47 retained initiatives (40.4%)," is a baseline-style sentence) but every such phrasing found in the actual 5 baseline_v2 reports was arithmetically consistent.

Net: baseline_v2's prose states appreciably fewer percentages and derived counts overall than the pipeline's (`baseline_v2_applicability.csv`: 3/5 baseline_v2 reports use any `%` at all, vs. every pipeline report), so there is less arithmetic surface for any of the four checks to find an error in — a smaller-denominator effect, not evidence that baseline_v2's claims are more internally consistent than the pipeline's per-claim. E1 (§4.6.1, `E1_SUMMARY_v2.md`) is the more informative comparison for the two systems' claims overall.

## 4. B1 vs. pipeline

B1's verified rate (0.470/1,000 words) is lower than the pipeline's (1.741/1,000 words). London drives most of the gap: the pipeline's london report carries 6 verified TPs (5× CS-5 — the invented Westminster/Lewisham per-1,000-residents figures — plus 1× CS-2), all keyed to the same "Westminster's 33 FSIs equate to 1.5 initiatives per 1,000 residents (assuming a population of 220,000)" sentence and its Lewisham counterpart; b1's london report has **zero** verified TPs — its two CS-5 hits are generic per-capita/census phrasing that E2's own verification marked FP, not the specific invented figures. Since `{instruction}` (the ablated slot) carries only qualitative GraphRAG background and never numbers (thesis §4.5.4, "A Structural Constraint on the GraphRAG Claim"), this specific invented-population sentence disappearing under ablation is model-generation variance between the two runs, not a mechanism the ablation's own design predicts — worth noting as an observation, not treating as a causal finding this document establishes. Dublin and milan's verified counts are comparable between pipeline and b1 (dublin 1 TP either way; milan 4→2). See `b1_claim_overlap.csv`/`B1_report.md` for the full claim-level detail this summary does not repeat.
