# B1 — Ablation: {instruction} removed from the Phase 5 prompt

Regenerates the pipeline (qwen3:14b) report for all five cities with the
`{instruction}` block of the Phase 5 prompt (the GraphRAG qualitative
context, `graph_answers[...]["answer"]`) forced to an empty string, holding
everything else fixed: same prompt template, same `{data_points}` built from
the existing `fsi_enriched.jsonl` via `extract_facts()`, same model
(`qwen3:14b`), same `temperature=0`/`num_ctx=16384`, same six sections.
Reuses cached artefacts only — no scraping, no graph rebuild, no Phase 0-4
stage was run.

## How the ablation was implemented

`src/phase_5/text_synthesizer.py` builds each section's `instruction` as
`graph_answers.get(<field>, {}).get("answer", "")`. Rather than reimplement
the prompt (risking drift from the real one), `runs/run_phase5_b1.py` calls
the **exact same** `synthesize_all()` used by the real pipeline
(`runs/run_phase5.py`) but passes `graph_answers={}` — every field lookup
then misses and falls back to `""`, which is precisely the ablation
condition, using the pipeline's own prompt-building code unmodified. The
`digital` section already hardcodes `instruction=""` in the unablated
pipeline, so it acts as a same-prompt control — see "Determinism check"
below.

Outputs written (existing pipeline reports were never opened for writing —
verified by MD5 checksum before/after the run, all ten files identical):

```
data/<city>/output/report_<city>_b1.html
data/<city>/output/report_<city>_b1.pdf
```
for `city` in barcelona, brighton, dublin, london, milan.

## Determinism check (why this matters for reading the diffs below)

Because `digital`'s instruction is `""` in *both* the real pipeline and B1,
its prompt is byte-identical between the two runs. It is therefore a
same-prompt, temperature=0 control on run-to-run reproducibility, not on the
ablation itself. Result: **identical, word-for-word, in 4 of 5 cities**
(barcelona, dublin, london, milan — `char_similarity_ratio = 1.0`,
word-count delta = 0). **Brighton's `digital` section differs** despite the
identical prompt — reworded throughout, though every number it states (76,
35, 35, 15, and the derived remainder) is unchanged between the two
versions. This confirms `temperature=0` on this local Ollama/qwen3:14b setup
is *not* a hard determinism guarantee (plausibly non-deterministic batched
float ops in llama.cpp), but also shows that when it does drift, it drifts
in wording, not in the numbers actually reported — which is the relevant
reassurance for reading the ablated sections' numeric-claim diffs as real
signal rather than pure noise.

## Word counts

| city | pipeline words | B1 words | Δ | Δ% |
|---|---:|---:|---:|---:|
| barcelona | 1,308 | 1,287 | −21 | −1.6% |
| brighton | 1,192 | 1,357 | +165 | +13.8% |
| dublin | 1,255 | 1,218 | −37 | −2.9% |
| london | 1,244 | 1,259 | +15 | +1.2% |
| milan | 1,319 | 1,260 | −59 | −4.5% |

No consistent direction across cities — removing `{instruction}` does not
systematically shorten or lengthen the report. Brighton is the outlier
(+13.8%, every one of its 6 sections grew), all other cities stay within
±5%.

## Per-section diff summary (all 5 cities)

`sim` = character-level `difflib.SequenceMatcher` ratio between the
pipeline and B1 versions of that section (1.0 = identical text, low values
= substantially reworded even where the same facts are covered — expected,
since re-running an LLM writer on a changed prompt rewrites prose from
scratch rather than editing it). `only‑p` / `only‑b1` = count of distinct
numeric tokens (including %) that appear in one version's section text and
not the other's.

| city | section | pipeline wc | B1 wc | Δ | sim | identical | only‑p | only‑b1 |
|---|---|---:|---:|---:|---:|:---:|---:|---:|
| barcelona | geographic | 239 | 245 | +6 | 0.32 | No | 3 | 3 |
| barcelona | types | 229 | 229 | 0 | 0.12 | No | 6 | 3 |
| barcelona | operational | 204 | 187 | −17 | 0.23 | No | 1 | 1 |
| barcelona | reach | 197 | 226 | +29 | 0.20 | No | 0 | 2 |
| barcelona | digital | 210 | 210 | 0 | **1.00** | **Yes** | 0 | 0 |
| barcelona | notable | 229 | 190 | −39 | 0.09 | No | 2 | 0 |
| brighton | geographic | 197 | 222 | +25 | 0.30 | No | 3 | 0 |
| brighton | types | 184 | 220 | +36 | 0.35 | No | 0 | 8 |
| brighton | operational | 195 | 211 | +16 | 0.30 | No | 3 | 0 |
| brighton | reach | 223 | 231 | +8 | 0.11 | No | 3 | 0 |
| brighton | digital | 195 | 228 | +33 | 0.12 | No | 1 | 1 |
| brighton | notable | 198 | 245 | +47 | 0.14 | No | 1 | 0 |
| dublin | geographic | 257 | 219 | −38 | 0.33 | No | 2 | 4 |
| dublin | types | 195 | 191 | −4 | 0.17 | No | 1 | 4 |
| dublin | operational | 206 | 178 | −28 | 0.12 | No | 1 | 0 |
| dublin | reach | 186 | 206 | +20 | 0.28 | No | 1 | 1 |
| dublin | digital | 204 | 204 | 0 | **1.00** | **Yes** | 0 | 0 |
| dublin | notable | 207 | 220 | +13 | 0.12 | No | 1 | 5 |
| london | geographic | 242 | 237 | −5 | 0.16 | No | 12 | 4 |
| london | types | 200 | 216 | +16 | 0.06 | No | 0 | 3 |
| london | operational | 187 | 210 | +23 | 0.14 | No | 0 | 0 |
| london | reach | 205 | 219 | +14 | 0.16 | No | 0 | 0 |
| london | digital | 193 | 193 | 0 | **1.00** | **Yes** | 0 | 0 |
| london | notable | 217 | 184 | −33 | 0.27 | No | 0 | 1 |
| milan | geographic | 241 | 250 | +9 | 0.24 | No | 0 | 2 |
| milan | types | 217 | 203 | −14 | 0.20 | No | 0 | 6 |
| milan | operational | 211 | 181 | −30 | 0.22 | No | 1 | 1 |
| milan | reach | 221 | 217 | −4 | 0.16 | No | 0 | 1 |
| milan | digital | 191 | 191 | 0 | **1.00** | **Yes** | 0 | 0 |
| milan | notable | 238 | 218 | −20 | 0.23 | No | 1 | 3 |

Every non-`digital` section is substantially reworded (`sim` mostly
0.06-0.35 — most sentences change even when they state the same fact), which
is expected: the LLM writer regenerates prose from the changed prompt, it
does not edit the old text. `operational`, `reach`, `types` and `notable`
all have at least one city with zero numeric-claim difference despite heavy
rewording (e.g. london operational/reach, barcelona/london/milan's
untouched-number cases above) — the words change, the figures often do not.

## Do any numeric claims actually differ? Yes — verified, concrete cases

Raw token counts above overstate "meaningful" differences (many are
incidental small integers — list positions, "two paragraphs", etc.). The
cases below were manually read against both HTML files, not just counted.

**1. London geographic — the fabricated per-capita/population arithmetic is entirely absent from B1.**
The pipeline version states: *"Westminster's 33 FSIs equate to 1.5
initiatives per 1,000 residents (assuming a population of 220,000), whereas
Lewisham's single FSI represents 0.004 initiatives per 1,000 residents
(assuming a population of 330,000)"* — this is the exact fabricated-basis
claim independently flagged by E2's CS-5 check and confirmed as a verified
manual-audit TP (`invented_basis_and_arithmetic`, M10/M11 in
`manual_error_catalogue.csv`), explicitly forbidden by the prompt's own
DATA POINTS note ("Data gap: FSI density per capita requires census
population data"). **None of that paragraph — no population figures, no
per-1,000 arithmetic — appears anywhere in the B1 version.** This is the
single clearest case in this ablation where removing `{instruction}`
removes a specific, previously-documented fabrication.

The pipeline version is also internally inconsistent about the green
districts' share of the total, stating both "49.1%" and "54.7%" for the
same three districts (Westminster + Tower Hamlets + Lambeth) in the same
section — a self-contradiction independently confirmed in the E6 manual
audit (M08, M09). B1 states one figure for the same group throughout ("78
FSIs... accounting for 73.6% of all FSIs"), consistently, in both
paragraphs. (This is not a claim that B1's own figures are correct — 78 was
not independently re-verified against `facts` here — only that B1 does not
reproduce *this specific* pipeline self-contradiction.)

**2. Dublin operational — the 46/23 vs 10+7 contradiction is absent from B1.**
Pipeline: *"46 being community-led (grassroots) and the remaining 23
supported by government or local councils"* ... *"Government-funded
initiatives account for 10, and 7 are council-supported"* — 10+7=17, not
23, a verified manual-audit finding (M19, `derived_incorrect`, UNCLEAR
verdict — "no clear way to know which [figure] is wrong, but the text is
contradictory"). B1 states: *"The majority (46) are community-led grassroots
efforts, while 10 are government-funded and 7 receive council support"* —
it does not assert a competing "23" total at all, so this specific
arithmetic contradiction does not arise.

**3. Milan geographic — the flagship 34% Municipio misattribution survives the ablation almost unchanged.**
This is the most-discussed single error in the whole evaluation
(originally a worked example in the now-removed, superseded E1_SUMMARY.md;
independently confirmed by E2's CS-4 check; verified manual-audit TP M01,
`misattributed_value`: 28+22+15=65,
65/147=44.2%, not 34% — 34% is Milan's *FSI-type* "other" category share,
misattached to a district-share claim). **Pipeline:** *"Municipio 1 (28
FSIs), Municipio 5 (22 FSIs), and Municipio 9 (15 FSIs), which collectively
account for 34% of all initiatives."* **B1:** *"Municipio 1 (28 FSIs),
Municipio 5 (22 FSIs), and Municipio 9 (15 FSIs), which together account
for 34% of all FSIs in the city."* Nearly verbatim, same wrong figure. This
is strong evidence that *this particular* error is not sourced from the
GraphRAG `{instruction}` context at all — it reproduces from the
`{data_points}`/prompt structure alone. (B1 additionally states "these three
districts... collectively house 65 FSIs" in the very same paragraph as the
"34%" claim — the correct sum sitting directly next to the wrong
percentage, an even more starkly local inconsistency than the pipeline
version shows.)

**4. Not every difference is an error — some are equivalent restatements.**
Barcelona geographic: pipeline says top-3 districts "account for 44% of all
FSIs" (105/237=44.3%, rounded); B1 says "account for 105 of the total FSIs"
in paragraph 1 and "collectively represent 44.3% of all FSIs" in paragraph
2 — same underlying figure, unrounded, just relocated and reformatted. This
is presentation churn, not a factual disagreement, and is a reminder that
raw "numbers only in one version" counts (the table above) overstate how
many of these are actual claim disagreements versus paraphrase artefacts.

**5. New errors can appear too — the ablation is not strictly a cleanup.**
Milan geographic, B1 only: paragraph 1 states Municipio 4 and Municipio 7
each host "11" FSIs; paragraph 2 of the *same B1 section* states Municipio
4 hosts "8" (matching the pipeline's figure, which is internally
consistent on this point in both its own mentions). B1 introduces a
same-section self-contradiction here that the pipeline version does not
have. Dublin geographic similarly replaces the pipeline's flagged "54%"
green-district claim (M17) not with a correct figure but with a different,
also-unverified "34%" for the top-three-district share (32/69=46.4%,
neither 54% nor 34% matches). Removing `{instruction}` changes *which*
numbers get fabricated in some sections; it does not make the generator
reliably more accurate.

## Bottom line

- The ablation script is a faithful, minimal-diff change: it reuses the
  pipeline's own `synthesize_all()` unmodified, changing only what
  `graph_answers` resolves to, so the prompt template, `{data_points}`,
  model, and temperature are provably identical to the real pipeline run
  (the `digital` control section proves this — identical output in 4/5
  cities under the identical empty-instruction prompt).
- Section-level rewording is substantial everywhere `{instruction}` was
  actually non-empty in the original (`sim` mostly 0.06-0.35), as expected
  from re-generating prose rather than editing it.
- At least one specific, previously-flagged fabrication class (invented
  per-capita/population arithmetic, London geographic) is fully absent from
  the ablated version — a concrete, traceable effect of removing
  `{instruction}`.
- At least one flagship, independently-confirmed misattribution (Milan's
  34% Municipio claim) reproduces almost verbatim without `{instruction}`,
  showing that error is not instruction-sourced.
- Some sections trade one unverified numeric claim for a different one
  rather than resolving it, and B1 introduces at least one fresh
  within-section self-contradiction not present in the pipeline version
  (Milan geographic, Municipio 4: "11" vs "8" in its own two paragraphs).
  This ablation should be read as evidence about *which* errors are
  instruction-linked, not as evidence that removing `{instruction}`
  improves overall report accuracy.

## Files

| File | Contents |
|---|---|
| `runs/run_phase5_b1.py` | Ablation driver — reuses cached artefacts + pipeline's own `synthesize_all()`, `graph_answers={}` |
| `runs/b1_diff.py` | Read-only diff tool: extracts the 6 sections from a pipeline/B1 HTML pair, word counts, numeric-token diff |
| `data/<city>/output/report_<city>_b1.html` / `.pdf` | The 5 ablated reports (new files; existing `report_<city>.html/.pdf` untouched — verified by checksum) |
