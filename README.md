# MultiSummGRAG — MSE Thesis Repository

This repository contains the implementation, stored experimental artefacts, and final evaluation material for the MSE thesis:

**Multimodal Summarization of Topically Related Websites using GraphRAG**

Author: **Kimia Rabishokr**  
Program: **MSE Data Science, ZHAW**  
Submission: **August 2026**

## Overview

MultiSummGRAG is a multimodal, multi-document summarization pipeline for collections of topically related websites. The system was developed in the context of the MediaEval MultiSumm task and was evaluated on Food Sharing Initiative (FSI) corpora from five cities:

- Barcelona
- Brighton & Hove
- Dublin
- London
- Milan

The pipeline combines structured extraction, geographic processing, GraphRAG-derived context, and LLM-based report generation. The thesis focuses specifically on the contribution of the GraphRAG-derived context to the final generated reports.

The central research question is:

> To what extent does GraphRAG-derived context contribute to the generated report content in MultiSummGRAG, and is this contribution sufficient to justify its additional computational and structural complexity?

## Pipeline

The public thesis version of the pipeline contains **Phases 0–5**.

### Phase 0 — URL Cleaning and Initial Filtering

Prepares the input URL lists by handling invalid or unsupported URLs, duplicates, missing coordinates, and related preprocessing.

Relevant code:

```text
src/phase_0/
```

### Phase 1 — Web Extraction and Record Construction

Processes the retained URLs, extracts webpage content and metadata, resolves geographic information, and constructs the stored FSI records.

Relevant code:

```text
src/phase_1/
```

### Phase 2 — Classification and Geographic Assignment

Enriches the retained records with model-derived categorical information such as FSI type and operational model and assigns geographic areas where possible.

Relevant code:

```text
src/phase_2/
```

### Phase 3 — GraphRAG

Constructs the GraphRAG corpus and graph artefacts and stores the GraphRAG query answers used as qualitative context during report generation.

Relevant code:

```text
src/phase_3/
```

Important persisted artefacts include:

```text
data/<city>/graphrag/input/
data/<city>/graphrag/kv_store_text_chunks.json
data/<city>/graphrag/graph_chunk_entity_relation.graphml
data/<city>/graphrag/kv_store_community_reports.json
data/<city>/output/phase3_answers.json
```

### Phase 4 — Structured Aggregation

Aggregates the extracted and enriched information into the structured data used by the final report-generation stage.

Relevant code:

```text
src/phase_4/
```

### Phase 5 — Multimodal Report Generation

Generates the final city-level multimodal reports using structured evidence and, in the Full condition, the persisted GraphRAG-derived context.

Relevant code:

```text
src/phase_5/
```

Canonical stored reports are located under:

```text
data/<city>/output/
```

For each city, the repository retains:

```text
report_<city>.html
report_<city>_b1.html
report_<city>_baseline_v2.html
```

PDF versions are also retained as human-browsable snapshots.

## Experimental Conditions

### Full

The complete MultiSummGRAG pipeline. Phase 5 receives both:

- structured pipeline-derived evidence; and
- the persisted GraphRAG-derived instruction/context.

### No-GraphRAG (B1)

A controlled ablation of the Full system. It uses the same structured evidence, Phase 5 template, model configuration, and section structure, but the GraphRAG-derived instruction is removed.

Relevant code:

```text
runs/run_phase5_b1.py
```

### Commercial Baseline (`baseline_v2`)

A separately generated commercial LLM workflow is retained as a **secondary practical comparison**.

It is not a causal baseline for GraphRAG because it differs from MultiSummGRAG in several ways, including retrieval process, evidence access, generator model, prompt, and use of live web browsing.

Only the artefacts explicitly labelled **`baseline_v2`** belong to the final commercial-baseline comparison.

Relevant provenance files:

```text
evaluation/baseline_v2_prompt.txt
evaluation/baseline_v2_provenance.md
```

## Final Evaluation

The thesis does not use a single overall quality score. Instead, the evaluation combines a controlled GraphRAG ablation with supporting checks over stored artefacts.

### 1. GraphRAG Artefact Analysis

The persisted GraphRAG artefacts are analysed without rebuilding or re-querying the graph.

The analysis covers graph size and connectivity, isolated nodes, largest connected component, community structure, prompt-example/template leakage in graph entities, and propagation checks into community reports and stored Phase 3 answers.

Relevant code and results:

```text
evaluation/analyze_graphrag_artifacts.py
evaluation/results/graphrag_artifact_audit/
```

### 2. Full vs No-GraphRAG Ablation

The primary experiment compares the Full and No-GraphRAG conditions while holding the remaining Phase 5 inputs and configuration fixed.

Relevant code and results:

```text
runs/run_phase5_b1.py
evaluation/scripts/run_b1_claim_overlap.py
evaluation/results/b1_claim_overlap.csv
evaluation/results/b1_comparison.csv
```

### 3. Replication and Generation Stability

Three additional Full and three additional No-GraphRAG outputs were generated per city to distinguish the between-condition effect from ordinary Phase 5 variation.

Relevant code and results:

```text
evaluation/run_b1_replication.py
evaluation/analyze_b1_replication.py
evaluation/results/b1_replication/
```

### 4. Numeric Auditability

Numeric claims in the reports are checked against the structured evidence available to Phase 5.

Relevant code and results:

```text
evaluation/e1_evidence_consistency.py
evaluation/results/e1_claims.csv
evaluation/results/e1_summary.csv
```

### 5. Internal Consistency

Rule-based candidate inconsistencies are identified and manually verified.

Relevant code and results:

```text
evaluation/e2_internal_consistency.py
evaluation/results/e2_flags.csv
evaluation/results/e2_summary.csv
evaluation/E2_SUMMARY_v2.md
```

### 6. Manual Full-Report Error Review

A manual review of the original Full reports was used as a diagnostic check for issues that narrow automated rules could miss.

Relevant material:

```text
evaluation/scripts/e6_manual_audit.py
evaluation/results/manual_error_catalogue.csv
evaluation/results/e6_manual_audit.csv
evaluation/E6_SUMMARY_v2.md
```

### 7. Chart–Text Consistency

The district and FSI-type charts were manually cross-checked against the corresponding report values.

For the Full and No-GraphRAG conditions, chart values are traceable to the deterministic structured facts. For `baseline_v2`, the values were inspected from the rendered report artefacts.

Relevant result:

```text
evaluation/results/chart_consistency.csv
```

### 8. Geospatial Validation

The evaluation includes deterministic geographic checks and a separate manual district-assignment spot check.

Relevant material:

```text
evaluation/results/e1_geo.csv
evaluation/scripts/run_district_spotcheck_sample.py
evaluation/results/district_spotcheck.csv
```

### 9. Phase 2 Human Validation

A blinded human annotation was used to assess agreement with selected Phase 2 categorical outputs.

This is reported as **agreement**, not classifier accuracy or ground truth.

Relevant code and results:

```text
evaluation/create_phase2_validation_sample.py
evaluation/analyze_phase2_validation.py
evaluation/results/phase2_validation/
```

### 10. Commercial-Baseline Screening Comparison

Phase 0 retention decisions are compared descriptively with the semantic screening decisions made by `baseline_v2`.

Because the two workflows perform different filtering tasks, this comparison is not interpreted as precision, recall, or classifier accuracy.

Relevant code and results:

```text
evaluation/e3_coverage_validity.py
evaluation/scripts/run_baseline_v2_e3.py
evaluation/results/e3_v2_*.csv
evaluation/E3_SUMMARY_v2.md
```

### 11. Human Screening Reproducibility for `baseline_v2`

The existing blinded human screening decisions were rejoined against the final `baseline_v2` Screening Ledger.

The preserved result is:

- 40 sampled URLs;
- 35 decidable after excluding 3 `UNREACHABLE_NOW` and 2 `UNCLEAR`;
- 25/35 agreement;
- 71.4% raw agreement;
- Cohen's κ = 0.432.

Relevant code and results:

```text
evaluation/results/e5_set_b.csv
evaluation/scripts/run_baseline_v2_human_screening_validation.py
evaluation/results/baseline_v2_human_screening_validation.csv
evaluation/results/baseline_v2_human_screening_validation_summary.md
```

### 12. Semantic-Slot Human Check

A 25-claim human check evaluates whether extracted numeric claims were associated with the correct structured evidence slots.

All 25 sampled claims were labelled `CORRECT_SLOT`.

Relevant material:

```text
evaluation/results/e5_set_c.csv
evaluation/scripts/e5_final_stats.py
evaluation/results/e5_final_stats.csv
evaluation/results/e5_final_stats.md
```

### 13. Multimodal Output Audit

The stored Full and `baseline_v2` reports are audited for multimodal content and traceability using the persisted HTML artefacts.

Relevant code and results:

```text
evaluation/analyze_multimodal_output.py
evaluation/results/multimodal_output_audit/
```

## Repository Structure

```text
.
├── analysis/              Corpus-level analysis and supporting tables/figures
├── config/                City-specific pipeline configuration
├── data/                  Stored corpora, intermediate artefacts, and reports
├── evaluation/            Final thesis evaluation code, annotations, and results
├── runs/                  Pipeline and ablation entry points
├── src/                   MultiSummGRAG implementation, Phases 0–5
├── tests/                 Phase 0–5 tests and fixtures
├── requirements.txt
└── README.md
```

## Data and Stored Artefacts

The repository intentionally keeps a substantial amount of the experimental data used in the thesis, including raw scraped webpages where retained, GraphRAG input documents, GraphML graphs, community reports, structured intermediate artefacts, final Full/No-GraphRAG/`baseline_v2` reports, human annotations, and final evaluation outputs.

Some large or regenerable local caches are intentionally excluded through `.gitignore`, including:

```text
data/*/output/images/
data/*/graphrag/vdb_entities.json
data/*/graphrag/kv_store_full_docs.json
data/*/graphrag/kv_store_llm_response_cache.json
```

These exclusions do not affect the stored artefacts required by the final thesis evaluation.

## Reproducibility

Most final evaluation analyses operate directly on stored artefacts and can therefore be reproduced without re-running GraphRAG or regenerating the reports.

Important caveats:

- GraphRAG indexing was computationally expensive and is not required to reproduce the final evaluation.
- The original GraphRAG artefacts are treated as frozen experimental outputs.
- The replication experiment records the model digest used for its generation, but deterministic decoding does not guarantee byte-identical outputs across different hardware/software environments.
- The commercial `baseline_v2` reports are fixed external artefacts and are not expected to be regenerable from this repository.
- Several validation components intentionally contain human judgment and are reproducible as stored annotation/result artefacts rather than as fully automated processes.
- Historical absolute paths inside frozen experimental artefacts are retained as provenance and should not be interpreted as current runtime paths.

## Environment

Dependencies are listed in:

```text
requirements.txt
```

The GraphRAG implementation uses:

```text
nano-graphrag==0.0.8.2
```

The recorded replication experiment used a local `qwen3:14b` model build. See the replication manifest under:

```text
evaluation/results/b1_replication/
```

for the recorded generation metadata.

## Running the Pipeline

The consolidated pipeline entry point is:

```bash
python runs/run_pipeline.py <city>
```

The public thesis version runs Phases 1–5 and no longer includes the retired Phase 6 evaluation stage.

Individual phases can also be run through the scripts under `runs/`.

Re-running the full pipeline may require local services, model weights, network access, and substantial compute. For reproducing the thesis evaluation, use the stored artefacts instead of rebuilding the pipeline unless regeneration is specifically required.

## Running Final Evaluation Scripts

Examples of deterministic or stored-artefact analyses include:

```bash
python evaluation/analyze_graphrag_artifacts.py
python evaluation/analyze_b1_replication.py
python evaluation/analyze_multimodal_output.py
python evaluation/scripts/run_baseline_v2_human_screening_validation.py
```

Some scripts may require the environment specified in `requirements.txt`.

Do not overwrite frozen experimental artefacts when reproducing analyses.

## Scope of This Release

This repository corresponds to the **final thesis methodology**.

The following earlier or exploratory approaches are intentionally **not part of this release**:

- the retired pre-`baseline_v2` commercial baseline results;
- the former Phase 6 automatic evaluation stage;
- the abandoned `phase_6_v2` implementation;
- the pairwise LLM-as-judge / position-bias experiment;
- exploratory GraphRAG context-uptake analyses.

These were excluded because they are not used as evidence in the final thesis.

## Main Finding

Within the evaluated five-city setting, GraphRAG-derived context changes the composition of generated reports beyond the ordinary variation observed across repeated Phase 5 generations.

However, the evaluation does not establish a corresponding measurable improvement sufficient to justify the additional computational and structural complexity of GraphRAG in the current MultiSummGRAG implementation.

This conclusion is bounded to the evaluated corpora, implementation, and report properties measured in the thesis. Qualitative dimensions such as readability, thematic organization, usefulness, and cross-document synthesis were not evaluated through a controlled human study.

## Citation

If you use this repository, please cite the corresponding MSE thesis.

A formal citation and permanent release identifier can be added here once the final thesis repository release/tag is published.

## License

No explicit software or data license is currently declared.

The repository contains research code as well as stored webpage-derived artefacts. Reuse of third-party webpage content remains subject to the rights and terms of the original sources.
