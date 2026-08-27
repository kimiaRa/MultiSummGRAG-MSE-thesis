# E1 — Claim auditability and E1-geo — geospatial verification (v2)

**Conditions covered:** pipeline (`report_<city>.html`), b1 (`report_<city>_b1.html`), baseline_v2 (`report_<city>_baseline_v2.html`). "Baseline" below always means `baseline_v2`; the earlier, superseded baseline is not reported here (no result in the thesis rests on it — see chapter4, "An earlier baseline"). Five cities: barcelona, brighton, dublin, london, milan. Fifteen reports total.

**Source files (not recomputed for this document):**
- `evaluation/results/e1_summary.csv` — auditability counts per (city, system), baseline_v2 rows used
- `evaluation/results/e1_geo.csv` — E1-geo district checks (pipeline + ground_truth rows only; see §3 note on why its `baseline` rows are not used here)
- `evaluation/results/district_spotcheck.csv` — manual per-initiative district check, 15 initiatives (3/city)

Read-only measurement of whether each system's own numeric claims can be checked against a persistent, machine-readable evidence base that system retained — **not** a measurement of whether those claims are correct. AUDITABLE_DIRECT = value appears verbatim in the evidence base. AUDITABLE_DERIVED = value is an exact sum, or a percentage of a candidate total within 0.55pp. NOT_MACHINE_AUDITABLE = neither. Auditability rate = (direct + derived) / claims.

## 1. Per-city, per-system

| city | system | n_claims | direct | derived | not_auditable | auditability_rate |
|---|---|---:|---:|---:|---:|---:|
| barcelona | pipeline | 86 | 76 | 10 | 0 | 1.000 |
| barcelona | b1 | 86 | 76 | 8 | 2 | 0.977 |
| barcelona | baseline_v2 | 71 | 18 | 3 | 50 | 0.296 |
| brighton | pipeline | 78 | 68 | 9 | 1 | 0.987 |
| brighton | b1 | 84 | 75 | 9 | 0 | 1.000 |
| brighton | baseline_v2 | 75 | 15 | 12 | 48 | 0.360 |
| dublin | pipeline | 77 | 72 | 5 | 0 | 1.000 |
| dublin | b1 | 91 | 84 | 7 | 0 | 1.000 |
| dublin | baseline_v2 | 54 | 6 | 8 | 40 | 0.259 |
| london | pipeline | 99 | 81 | 11 | 7 | 0.929 |
| london | b1 | 86 | 74 | 12 | 0 | 1.000 |
| london | baseline_v2 | 54 | 13 | 5 | 36 | 0.333 |
| milan | pipeline | 73 | 67 | 6 | 0 | 1.000 |
| milan | b1 | 77 | 66 | 11 | 0 | 1.000 |
| milan | baseline_v2 | 45 | 8 | 6 | 31 | 0.311 |

## 2. Pooled (sum of counts across cities, not an average of rates)

| system | n_claims | direct | derived | not_auditable | pooled auditability_rate |
|---|---:|---:|---:|---:|---:|
| pipeline | 413 | 364 | 41 | 8 | 0.9806 |
| b1 | 424 | 375 | 47 | 2 | 0.9953 |
| baseline_v2 | 299 | 60 | 34 | 205 | 0.3144 |

The pipeline and its B1 ablation are architecturally auditable by construction — every prose sentence is generated from a `{data_points}` block built directly from `fsi_enriched.jsonl` (§3.4 of the thesis), so almost every number in the rendered text traces back to that evidence base regardless of which GraphRAG instruction text (if any) accompanied it. `baseline_v2` is architecturally different in kind, not merely lower-scoring: it is a browsing agent whose only retained evidence is its own screening ledger (retain/discard decision + reason, one row per seed URL) — it has no structured per-FSI facts file, so any number in its prose beyond the ledger's own retained/discarded/total counts is, by this metric's definition, not machine-auditable against anything the system itself kept. This is the gap the metric is built to expose (thesis §4.6.1), not a claim that baseline_v2's numbers are wrong.

## 3. E1-geo — geospatial verification

**Applicability note:** `e1_geo.csv`'s `baseline` rows (district_sum_vs_total) were computed against the retired baseline's stated city totals (44/30/38/34/29) and are not used here — no baseline_v2-specific rerun of that check exists yet (unlike E1/E2/E3, which do have baseline_v2 reruns). This is a genuine gap, not a zero: it is reported as **not run for baseline_v2**, rather than silently omitted or shown as a blank/zero. The pipeline-side and ground-truth rows below are unaffected by which baseline is current, since they never depended on baseline text.

**(a) Per-district count vs. point-in-polygon recompute (pipeline only):**

| city | districts checked | exact matches | mismatches |
|---|---:|---:|---:|
| barcelona | 8 | 8 | 0 |
| brighton | 3 | 3 | 0 |
| dublin | 4 | 3 | 1 (Dublin 16: stated 4, recomputed 3) |
| london | 4 | 3 | 1 (Waltham Forest: stated 4, recomputed 2) |
| milan | 4 | 4 | 0 |
| **pooled** | **23** | **21** | **2** |

**(b) District-sum vs. stated city-wide total (pipeline only, with extractor-coverage caveat):**

| city | sum of extracted per-district counts | stated total | delta | districts paired / known |
|---|---:|---:|---:|---|
| barcelona | 169 | 237 | 68 | 8/10 |
| brighton | 76 | 76 | 0 (match) | 3/30 |
| dublin | 36 | 69 | 33 | 4/35 |
| london | 54 | 106 | 52 | 4/33 |
| milan | 73 | 147 | 74 | 4/9 |

These deltas are **not** reported as pipeline errors: coverage is low in four of five cities (as low as 3/30 named districts successfully paired with a nearby stated count by the regex extractor), so a nonzero delta reflects extractor recall as much as report content, exactly as `e1_geo.csv`'s own per-row `note` states. Brighton's exact match at only 3/30 coverage is itself a caution against reading a match as confirmation.

**(c) Ground truth — points falling outside every district polygon (`urls_cleaned.csv` vs. `districts.geojson`):**

| city | n_points_outside_all_districts |
|---|---:|
| barcelona | 35 |
| brighton | 0 |
| dublin | 1 |
| london | 9 |
| milan | 12 |

This is the same shortfall the chart-consistency check (below, and `chart_consistency.csv`) surfaces independently as a bar-chart-sum vs. stated-total mismatch in 4/5 pipeline reports — the two checks agree because they are measuring the same underlying coordinate-precision limitation (thesis §4.8, "Coordinate precision varies by city and source"), not two different defects.

**(d) Manual spot-check — 15 initiatives, 3 per city, page address vs. assigned district (`district_spotcheck.csv`):**

| MATCH | n |
|---|---:|
| yes | 7 |
| no | 4 |
| unclear (no address on page, or multi-district operation) | 4 |

7/11 decidable cases (yes+no, excluding unclear) agree = 63.6%. This is a small, hand-checked sample, not a claim about the whole corpus; the four "no" cases (e.g. D07 Dublin Food Co-op assigned "Dublin 2" but its page address is Dublin 8) are consistent with the coordinate-precision limitation named in (c), not evidence of a geofencing logic error — see thesis §4.8.

## 4. What this does and does not show

Unchanged from the original E1 methodology: this measures verifiability, not correctness (a claim can be auditable and still wrong — see E6 for verified-wrong pipeline claims that were nonetheless traceable to the evidence base); auditability is an upper bound on accuracy, not accuracy itself; and the pipeline/B1 vs. baseline_v2 gap is architectural (what each system kept), not a measure of which system tried harder.
