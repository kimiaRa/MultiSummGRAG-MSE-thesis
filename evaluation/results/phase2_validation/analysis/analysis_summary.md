# Phase-2 Classification Human Validation — Analysis

## DATA INTEGRITY

All validation checks in §1 passed before this analysis ran (75/75 rows in both files, matching sample_id sets, 50 core/25 stress overall and 10 core/5 stress per city, no missing human_fsi_status/human_confidence, genuine_fsi rows have both type/op labels and non-genuine rows have neither, no model-label column present in the frozen file, manifest actual_total_records=635 and random_seed=42) -- see the printed validation log for the exact checks run.

## CORE HUMAN STATUS DISTRIBUTION

CORE is 10 uniformly random records per city (city-balanced), NOT a simple random sample of all 635 records -- pooled percentages below describe this 50-row sample, not a precise population estimate.

| status | n (of 50) | % |
|---|---:|---:|
| genuine_fsi | 23 | 46.0 |
| not_an_fsi | 6 | 12.0 |
| insufficient_evidence | 21 | 42.0 |

Per city:

| city | genuine_fsi | not_an_fsi | insufficient_evidence |
|---|---:|---:|---:|
| barcelona | 5 | 0 | 5 |
| brighton | 5 | 0 | 5 |
| dublin | 5 | 1 | 4 |
| london | 5 | 1 | 4 |
| milan | 3 | 4 | 3 |

## FSI-TYPE AGREEMENT

Among 23 CORE genuine_fsi rows (evaluable = CORE + human_fsi_status==genuine_fsi): **10/23 exact matches (43.5%)**, 13 mismatches. Full confusion matrix (long format, every value used as stored, including 'unknown' and any out-of-schema model value) in `core_type_confusion.csv`. Macro-F1 not computed: this repository does not already use a classification-metrics library elsewhere, so per the instruction to avoid it unless trivially available, exact agreement and the confusion matrix are reported instead.

## OPERATIONAL-LEVEL AGREEMENT

Among 23 CORE genuine_fsi rows: **8/23 exact matches (34.8%)**, 15 mismatches. human='unknown' vs model='unknown' counts as an exact match under the same plain-equality rule as every other label pair -- no special-casing was needed or applied. Full confusion matrix in `core_operational_confusion.csv`.

## JOINT AGREEMENT

Among 23 CORE genuine_fsi rows: both fields match 4 (17.4%); type only 6 (26.1%); operational level only 4 (17.4%); neither 9 (39.1%).

## NON-FSI CLASSIFICATION DIAGNOSTIC

CORE rows where the human judged `not_an_fsi` — Phase 2 has no such output class, so these are reported as **non-FSI records receiving downstream classification labels**, not as classification accuracy.

n=6. model_fsi_type: 6 (100.0%) given a specific ordinary type, 0 (0.0%) 'unknown', 0 out-of-schema ([]). model_operational_level: 4 (66.7%) specific, 2 (33.3%) 'unknown', 0 out-of-schema ([]). Row-level detail in `core_status_diagnostics.csv`.

## INSUFFICIENT-EVIDENCE DIAGNOSTIC

CORE rows where the human judged the stored evidence `insufficient_evidence`. Specific model labels here are reported as **model specificity where the human annotator judged stored evidence insufficient** — not automatically an error — especially relevant since the Phase-2 prompt explicitly instructs best-guessing over 'unknown', and a keyword fallback can independently replace 'unknown'.

n=21. model_fsi_type: 13 (61.9%) specific, 8 (38.1%) 'unknown', 0 out-of-schema. model_operational_level: 10 (47.6%) specific, 11 (52.4%) 'unknown', 0 out-of-schema. Row-level detail in `core_status_diagnostics.csv`.

## CONFIDENCE SENSITIVITY

Sensitivity/interpretation aid only -- low-confidence human disagreement is not automatically interpreted as model error.

| confidence | n | type agreement % | op agreement % |
|---|---:|---:|---:|
| high | 11 | 54.5 | 54.5 |
| medium | 12 | 33.3 | 16.7 |
| low | 0 | None | None |

## STRESS-SAMPLE DIAGNOSTICS

STRESS (25 rows, 5/city) was deliberately selected for classifier edge cases (rare types, other, unknown, short-text-but-specific, type/op-unknown mismatches, out-of-schema values). **Every result in this section is diagnostic, not representative, and is never pooled with CORE into one accuracy/agreement number anywhere in this script.** Full grouped breakdown (by stress_reason and by city) in `stress_summary.csv`; full row-level detail in `stress_diagnostics.csv`.

Overall stress human-status counts: genuine_fsi=7, not_an_fsi=7, insufficient_evidence=11.

## OUT-OF-SCHEMA SENTINEL

`"food_service"` is outside the active Phase-2 fsi_type schema (`src/phase_2/classifier.py`'s SYSTEM_PROMPT enum) and demonstrates the absence of output-enum validation in the classifier, regardless of what the human annotator judged this specific record to be.

- `P2-041` (stress): human_fsi_status='genuine_fsi', human_fsi_type='food_gifting', model_fsi_type='food_service', model_operational_level='government_funded'. Full detail in `sentinel_result.csv`.

## INTERPRETATION LIMITS

- CORE is the primary validation sample; STRESS is diagnostic only.
- CORE is 10 uniformly random rows per city -- city-balanced, not proportional to the 635-record corpus (barcelona and london are under-weighted relative to their true share, brighton/dublin/milan over-weighted); pooled CORE percentages describe this 50-row sample, not a precise, unweighted population estimate.
- CORE and STRESS are never pooled into a single accuracy/agreement number anywhere in this script.
- There is one human annotator: these labels are a reference annotation, not unquestionable ground truth.
- No inter-annotator agreement was measured (single annotator, no second pass).
- Several fsi_type/operational_level classes have very small counts within this 50-row sample; per-class conclusions are descriptive, not inferentially powered.
- No FSI-status accuracy, precision/recall/F1, confusion matrix, or kappa was computed anywhere in this script -- Phase 2 has no not_an_fsi/insufficient_evidence output class to compare against, so no such comparison is meaningful.

## THESIS IMPLICATION

**A.** Does CORE support treating model-derived `fsi_type` labels as broadly reliable descriptive annotations? Based on 23 evaluable CORE rows at 43.5% exact agreement -- read this figure together with the confusion matrix and city/confidence breakdowns above before drawing a conclusion; do not overclaim from N=23.

**B.** Does CORE support treating `operational_level` labels as broadly reliable? Based on 23 evaluable CORE rows at 34.8% exact agreement (including unknown-unknown matches) -- same caveat: read alongside the confusion matrix, not as a standalone headline figure.

**C.** How serious is the absence of a `not_an_fsi` class in practice? See the NON-FSI CLASSIFICATION DIAGNOSTIC above: 6/50 CORE rows were judged not_an_fsi by the human annotator, and every one of them nonetheless received an ordinary Phase-2 type/operational-level label with no exclusion mechanism available.

**D.** Which existing report-level conclusions require qualification? Any city-comparison or cross-city narrative built on `type_counts`/`type_pct` or operational-level breakdowns should be read alongside this validation's agreement rates and confusion matrix, not as unqualified fact.

**E.** Is rerunning or correcting all 635 classifications necessary for this thesis, or is transparent limitation + validation sufficient? This is a judgement call for the thesis author to make from the figures above, not a conclusion this script draws automatically.
