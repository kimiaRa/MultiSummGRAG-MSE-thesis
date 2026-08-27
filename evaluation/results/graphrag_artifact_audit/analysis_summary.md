# GraphRAG Artefact Audit

## DATA INTEGRITY

All required artefacts (input/*.txt, kv_store_text_chunks.json, graph_chunk_entity_relation.graphml, kv_store_community_reports.json, phase3_answers.json) were present for all five cities before analysis began. Every file under data/<city>/graphrag/ and each city's phase3_answers.json was SHA-256-hashed before computation started and again after all outputs were written; the run aborted before writing anything if any hash differed (content-based check only -- modification timestamps were never consulted).

Canonical-name source: `nano_graphrag/prompt.py` located via `importlib.util.find_spec` (path not hardcoded, package never imported or executed) at `/home/ubuntu/MT_ZHAW/venv/lib/python3.12/site-packages/nano_graphrag/prompt.py`, SHA-256 = `12dac0c8cfeff179ca614461f82191060c6392875d1dd3a6839171881ca7da97`. All 14 canonical names were confirmed present in the INSTALLED SOURCE TEXT of `PROMPTS["entity_extraction"]` (extracted via static `ast.parse()` + `ast.literal_eval()` on the located assignment node only -- a static source check, not a runtime/execution-based inspection): ALEX (found), CONTROL (found), CRUZ (found), FIRST CONTACT (found), HUMANITY'S RESPONSE (found), INTELLIGENCE (found), JORDAN (found), MERCER (found), OPERATION: DULCE (found), SAM RIVERA (found), TAYLOR (found), THE DEVICE (found), THE TEAM (found), WASHINGTON (found).

## GRAPH STRUCTURE

| city | input_docs | chunks | nodes | edges | isolated % | avg_degree | communities |
|---|---:|---:|---:|---:|---:|---:|---:|
| barcelona | 196 | 220 | 2235 | 2412 | 13.47 | 2.1584 | 324 |
| brighton | 73 | 78 | 596 | 42 | 93.96 | 0.1409 | 5 |
| dublin | 58 | 64 | 747 | 26 | 96.25 | 0.0696 | 3 |
| london | 106 | 128 | 1905 | 2486 | 12.86 | 2.61 | 248 |
| milan | 142 | 180 | 2274 | 3006 | 8.88 | 2.6438 | 376 |

Full metrics (including connected_components, largest_component_pct, self_loops, edges_per_chunk, nodes_per_chunk) in `graph_structure.csv`.

## CROSS-CITY CONTRAST

FACT: edges-per-chunk ranges from dublin (0.4062) to london (19.4219) -- computed directly from GRAPH STRUCTURE above. Cities with low edges-per-chunk also show low `average_degree` and high `isolated_node_pct` in the same table. INTERPRETATION: this pattern is consistent with weaker relationship extraction on those cities' chunks; the cause is NOT established by this script (see INTERPRETATION LIMITS).

## CONFIRMED TEMPLATE LEAKAGE

| city | confirmed_template_leakage | legitimate_name_collision | unclear |
|---|---:|---:|---:|
| barcelona | 0 | 0 | 0 |
| brighton | 14 | 0 | 0 |
| dublin | 13 | 0 | 1 |
| london | 0 | 1 | 0 |
| milan | 0 | 0 | 0 |

Only rows classified `confirmed_template_leakage` (verbatim >=5-word shingle overlap with the live nano_graphrag entity-extraction prompt text -- see `matched_evidence_shingle` column) count as the contamination headline. `legitimate_name_collision` rows (name matches, but description/type traces to a different, real referent) are explicitly NOT counted as leakage. Full per-match detail, including the exact matched name, node description excerpt, and source chunk excerpt where recoverable, in `template_entity_matches.csv`.

## COMMUNITY REPORTS

Reports only whether CONFIRMED template leakage (§CONFIRMED TEMPLATE LEAKAGE) propagated into community reports -- no claim is made about whether a leakage-containing report also mentions a genuine city-corpus entity (a prior version of this check treated any non-canonical-14 node name as "real", which is invalid: derivative leakage entities such as Brighton's "ALEX'S TEAM" also fall outside that 14-name list, so that check could misclassify leakage material itself as a real entity; it has been removed, not replaced).

| city | reports | with confirmed leakage | confirmed names found |
|---|---:|---:|---|
| barcelona | 324 | 0 | - |
| brighton | 5 | 4 | ALEX;CONTROL;CRUZ;FIRST CONTACT;HUMANITY'S RESPONSE;INTELLIGENCE;JORDAN;MERCER;OPERATION: DULCE;SAM RIVERA;TAYLOR;THE DEVICE;THE TEAM;WASHINGTON |
| dublin | 3 | 0 | - |
| london | 248 | 0 | - |
| milan | 376 | 0 | - |

This resolves the prior audit's open question directly (rather than assuming graph-level leakage reached community reports) -- see `community_contamination.csv` for the exact matched template names per city.

## PHASE-3 QUERY ANSWERS

| city | 6/6 present | min length | max length | answers with confirmed leakage |
|---|---|---:|---:|---:|
| barcelona | 6/6 | 2164 | 5416 | 0 |
| brighton | 6/6 | 1660 | 2829 | 0 |
| dublin | 6/6 | 1687 | 3263 | 0 |
| london | 6/6 | 2359 | 6159 | 0 |
| milan | 6/6 | 1965 | 8763 | 0 |

Full per-field detail in `query_answer_summary.csv`. Factual correctness of any answer was NOT evaluated (see INTERPRETATION LIMITS).

## INTERPRETATION LIMITS

**FACT** (directly computed, reported above without qualification): all structural graph metrics, community-report counts, query-answer presence/length, and confirmed template-name matches.

**CONFIRMED DEFECT**: template/example material traced, via verbatim shingle overlap with the live `nano_graphrag/prompt.py` entity-extraction prompt, to actual graph nodes -- see CONFIRMED TEMPLATE LEAKAGE.

**NOT ESTABLISHED by this script**:
- that graph contamination changed any Phase-5 report's content;
- that a sparse graph is necessarily a bad graph;
- why Brighton/Dublin (or any city) show sparser relationship extraction than others;
- the factual correctness of any Phase-3 query answer;
- any single aggregate 'graph quality score' -- none is computed anywhere in this script or its outputs.

## THESIS IMPLICATION

**A.** Did all five cities produce non-empty GraphRAG artefacts? Yes -- every required artefact was present and non-empty for all five cities (see DATA INTEGRITY); query-answer presence: 6/6 for all cities.

**B.** Which cities show strong vs. very sparse relational structure? Dense (avg_degree >= 1.0): barcelona, london, milan. Sparse (avg_degree < 1.0): brighton, dublin.

**C.** Where is template leakage confirmed? brighton, dublin.

**D.** Did confirmed leakage propagate into community reports? brighton.

**E.** Did confirmed leakage propagate into the persisted Phase-3 answers? no city shows this.

**F.** Which structural measurements are suitable for the thesis? nodes/edges/average_degree, isolated_node_pct and largest_component_pct, community_reports count, and edges_per_chunk as the cross-city comparability metric -- all directly computed, reproducible from this script's CSV outputs.

**G.** Does any finding require rerunning GraphRAG? No. Every measurement above came from existing artefacts; sparse relational structure or confirmed template leakage in a given city is a reportable finding, not by itself a reason to rerun -- rerunning would also break comparability with every other evaluation artifact already built against the current graphs.
