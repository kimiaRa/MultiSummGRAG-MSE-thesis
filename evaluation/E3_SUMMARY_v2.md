# E3 — Coverage vs. validity (v2)

**Conditions covered:** pipeline vs. baseline_v2 only. E3 measures URL screening — it has no B1 analogue, because B1 reuses the pipeline's own already-screened `fsi_enriched.jsonl` unchanged (only the Phase-5 prompt's `{instruction}` slot differs); there is no separate B1 screening decision to compare against the ledger. This is stated explicitly rather than shown as a blank row.

**Source files (not recomputed for this document):** `e3_v2_coverage.csv`, `e3_v2_agreement.csv`, `e3_v2_directionality.csv`, `e3_v2_reason_breakdown.csv`, `e3_v2_ledger_validation.csv`, `e3_v2_recall_counter_check_summary.csv`, `e3_v2_url_match_rates.csv` — all produced by `evaluation/scripts/run_baseline_v2_e3.py`, which parses each `report_<city>_baseline_v2.html`'s own Screening Ledger `<table>` directly (not the retired PDF-based extractor).

**Ledger-vs-stated-outcome validation (sanity check before anything else):** for all 5 cities, the number of ledger rows extracted, the retained count, and the discarded count reconcile exactly against each report's own stated "Screening outcome: N of M URLs retained" line — `e3_v2_ledger_validation.csv`, `rows_match_stated_total` and `retained_matches_stated_n` both `True` for barcelona/brighton/dublin/london/milan. URL-normalisation match rate against the raw seed CSV is 100% in every city (`e3_v2_url_match_rates.csv`).

## 1. Coverage rate (surviving / seed)

| city | pipeline surviving/seed | pipeline coverage | baseline_v2 surviving/seed | baseline_v2 coverage |
|---|---|---:|---|---:|
| barcelona | 237/261 | 0.908 | 47/261 | 0.180 |
| brighton | 76/81 | 0.938 | 24/81 | 0.296 |
| dublin | 69/83 | 0.831 | 36/83 | 0.434 |
| london | 106/115 | 0.922 | 33/115 | 0.287 |
| milan | 147/163 | 0.902 | 34/163 | 0.209 |
| **pooled** | **635/703** | **0.903 (95% CI 0.874–0.923)** | **174/703** | **0.248 (95% CI 0.199–0.345)** |

Same 703-row seed list for both systems (thesis §4.5, "the pipeline reads the cleaned corpus... the baseline was produced... from the same seed CSV"). The pipeline retains ~90% of seed URLs (it has no relevance-screening stage after Phase 0's flag-don't-remove step); the agentic baseline retains ~25%, consistent with the thesis's characterisation of it as a genuine screening workflow rather than a one-shot summarisation call.

## 2. Phase-0-vs-ledger agreement (does the pipeline's keep/discard decision match the ledger's independent judgement?)

| city | kept & ledger-retained | kept & ledger-discarded | discarded & ledger-retained | discarded & ledger-discarded | n | raw agreement | Cohen's κ |
|---|---:|---:|---:|---:|---:|---:|---:|
| barcelona | 46 | 191 | 1 | 23 | 261 | 0.264 | 0.033 |
| brighton | 24 | 52 | 0 | 5 | 81 | 0.358 | 0.054 |
| dublin | 36 | 33 | 0 | 14 | 83 | 0.602 | 0.269 |
| london | 33 | 73 | 0 | 9 | 115 | 0.365 | 0.066 |
| milan | 34 | 113 | 0 | 16 | 163 | 0.307 | 0.056 |
| **pooled** | **173** | **462** | **1** | **67** | **703** | **0.341 (95% CI 0.284–0.459)** | **0.064 (95% CI 0.040–0.135)** |

κ near zero (not negative, but far below any conventional "acceptable agreement" threshold) pooled across all five cities: the two systems' screening decisions are close to independent of each other. This is expected given how differently the two decide — Phase 0 removes only confirmed-dead/duplicate/social-media URLs and flags-but-keeps everything else (thesis §4.2.1); the baseline reads and judges each page's actual relevance.

## 3. Directionality of disagreement

| city | kept-but-ledger-discarded | discarded-but-ledger-retained | n disagreements | % of disagreements that are "kept-but-ledger-discarded" |
|---|---:|---:|---:|---:|
| barcelona | 191 | 1 | 192 | 99.5% |
| brighton | 52 | 0 | 52 | 100.0% |
| dublin | 33 | 0 | 33 | 100.0% |
| london | 73 | 0 | 73 | 100.0% |
| milan | 113 | 0 | 113 | 100.0% |
| **pooled** | **462** | **1** | **463** | **99.8% (95% CI 99.56–100%)** |

Disagreement is almost entirely one-directional: essentially every case where the two systems disagree is the pipeline keeping a URL the baseline's ledger discarded, not the reverse. This is the expected shape given §1's coverage gap (~90% vs. ~25%) — it is arithmetically difficult for the reverse direction to be common when the baseline discards so much more than the pipeline does.

## 4. Reason breakdown — why the ledger discarded URLs the pipeline kept

| reason | n_urls_i_kept (pooled) |
|---|---:|
| inaccessible | 150 |
| not_an_fsi | 138 |
| directory_or_policy_page | 84 |
| outside_the_city | 46 |
| duplicate | 17 |
| activity_ended | 27 |

`inaccessible` and `not_an_fsi` together account for the large majority (288/462 = 62.3%) of pipeline-kept, ledger-discarded URLs. This motivates the recall counter-check below: an `inaccessible` reason only counts as a genuine screening advantage for the baseline if the page really was unusable.

## 5. Recall counter-check — was the discarded material really unusable?

Restricted to barcelona and london, the only two cities where `data/<city>/raw/*.html` survived on disk (912 and 540 raw files respectively; brighton/dublin/milan have none).

| city | n eligible (ledger-discarded as `inaccessible`, baseline_v2) | raw file exists | file has >500 chars body text |
|---|---:|---:|---:|
| barcelona | 94 | 51 (54.3%) | 49 (52.1%) |
| london | 27 | 19 (70.4%) | 17 (63.0%) |
| **pooled (2 clusters)** | **121** | **—** | **66 (54.5%, 95% CI 52.1–63.0%)** |

Note the eligible-count column changed between the retired baseline and baseline_v2 (`n_eligible_old_e3_baseline_v1` vs. `n_eligible_baseline_v2` differ in `e3_v2_recall_counter_check_summary.csv`) because baseline_v2's ledger reasons and discard counts are simply different from the retired baseline's — this is expected, not a bug, since it is a different agent run.

Roughly half of the pages the baseline's ledger marked `inaccessible` did in fact have a scraped file with substantial body text — meaning the baseline's `inaccessible` judgement and the pipeline's actual scrape success disagree on close to half of this specific reason-bucket. This is the one measurement designed to give the pipeline a chance to come out ahead on the coverage question, and it does show the baseline's screening is not simply "correct where it discards" — but it covers only 2 of 5 cities (raw HTML availability, not a methodology choice) and only the `inaccessible` bucket, not `not_an_fsi` or the others in §4.

## 6. Reading E3 alongside E1

Coverage and auditability move in opposite directions for the same two systems: the pipeline has near-total coverage (90%) and near-total auditability (98%); baseline_v2 has much lower coverage (25%) and much lower auditability (31%). This is the coverage/validity trade-off the thesis's evaluation design is built to surface (§4.6, "Neither behaviour is simply right: one favours coverage at the cost of precision, the other the reverse") — E3 does not resolve which system is better, only measures what each gives up.
