# E6 — Manual error audit (v2)

**Conditions covered: pipeline only.** B1 and baseline_v2 were **not** manually audited — this is a scope boundary of E6 as designed (thesis §4.6.6, "All five pipeline reports were read in full"), not an oversight or a gap this document is filling. Stated explicitly rather than left implicit: this document does not compare pipeline vs. b1 vs. baseline_v2 defect rates, and no future rerun of `e6_manual_audit.py` is needed to bring b1/baseline_v2 "up to date" — they were simply never in scope.

**Source file (not recomputed for this document):** `evaluation/results/manual_error_catalogue.csv`, 31 hand-identified candidate errors from close reading of the five pipeline reports, each with a `VERDICT` (TP/FP/UNCLEAR). This data does not depend on which baseline condition is current — it only concerns the pipeline's own reports — so it required no rerun when baseline_v2 replaced the retired baseline; only the framing below changes.

## 1. Screening outcome

31 candidates: **20 TP, 7 FP, 4 UNCLEAR**. Candidate precision including UNCLEAR as non-error: 20/31 = 64.5%; excluding UNCLEAR: 20/27 = 74.1%.

## 2. Verified errors by city

| city | n_words | n_verified_errors | errors/1,000 words | 95% CI (exact Poisson) |
|---|---:|---:|---:|---:|
| barcelona | 1308 | 2 | 1.529 | 0.185–5.523 |
| brighton | 1192 | 3 | 2.517 | 0.519–7.355 |
| dublin | 1255 | 3 | 2.390 | 0.493–6.986 |
| london | 1244 | 9 | 7.235 | 3.308–13.734 |
| milan | 1319 | 3 | 2.274 | 0.469–6.647 |
| **pooled** | **6318** | **20** | **3.166** | **1.934–4.889** |

## 3. Verified errors by type

| error_type | n |
|---|---:|
| internal_contradiction | 5 |
| derived_incorrect | 4 |
| misattributed_value | 3 |
| invented_basis_and_arithmetic | 2 |
| quantifier_error | 2 |
| count_percent_conflation | 1 |
| chart_text_mismatch | 1 |
| possible_fabrication | 1 |
| unverifiable_self_endorsement | 1 |

## 4. Cross-reference with E2

Match rule: a verified (TP) manual error counts as also caught by E2 if `e2_flags_v2.csv` has a same-city, same-section flag sharing a numeric token with the manual entry's quoted claim.

- **Caught by both E6 and an automated E2 check: 5** (M01 milan/CS-4, M10 london/CS-5, M11 london/CS-5, M13 london/CS-2, M18 dublin/CS-2 — all independently confirmed TP in `e2_flags_v2.csv` too)
- **Manual audit only: 15** (e.g. M07 london's cross-section "nine districts" vs. "three districts" self-contradiction; M05 milan's "none are volunteer-based" vs. "141 non-commercial, encompassing voluntary... work" semantic contradiction; M23 brighton's fabricated named-organisation figure)

This cross-reference is pipeline-only data and is numerically unchanged from the original E6_SUMMARY.md's cross-reference (pipeline's own E2 flag composition — 17 flags, 11 TP/6 FP — is identical between the pre- and post-widening runs; see `e2_widen_old_vs_new_summary.csv`), so the count is carried forward here rather than recomputed.

## 5. Error classes the automated checks structurally cannot reach

Three, each for a different reason — unchanged from the original audit since this is pipeline-only analysis:

- **Cross-section self-contradiction** (CS-3, prototyped and rejected — see `E2_SUMMARY_v2.md` §"CS-3"): M07 london states "nine districts... each host only one FSI" in one paragraph and "three districts... each have only one FSI" two paragraphs later — no single-sentence check can see this.
- **Semantic contradiction with no arithmetic signature**: M05 milan — "none are volunteer-based" against "141 non-commercial, encompassing voluntary... work" in the same paragraph. Neither side is a number CS-1/2/4 can check against a total.
- **Fabricated named-entity figures**: M23 brighton — "The In the Bag Project reaches 10,000+ individuals" is internally consistent (conflicts with nothing else in the report) and was flagged only by checking the figure against `fsi_enriched.jsonl` and finding no supporting record.

## 6. Framing for the thesis

E2's verified automated rate (1.741/1,000 words pipeline-pooled, `E2_SUMMARY_v2.md` §2) is a **lower bound** on defect density — it can only ever report the four syntactic shapes it checks for. E6's manual rate (3.166/1,000 words, 95% CI 1.934–4.889) is roughly 1.8× E2's rate and overlaps its CI; the gap is exactly the three structural classes in §5. Report both, explicitly labelled: E2 = lower bound / automated / reproducible / all-three-conditions; E6 = primary estimate / manual / non-exhaustive / pipeline-only.
