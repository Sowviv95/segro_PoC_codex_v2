# SEGRO Evidence-First Working Sprint Plan

Plan version: 1.2

Last updated: 2026-07-29

Current sprint status: Sprint 3 complete. The repository contains dictionary ingestion, source-pack
discovery/registration, bounded ZIP handling, duplicate grouping and document classification.
Extraction is not implemented, and Sprint 4 has not started.

## Purpose And Principles

Build a modular evidence-first extraction engine that can process a selected data dictionary, an
arbitrary supported asset source pack and asset/customer configuration without prior-result
dependence.

Principles:

- evidence indexing before extraction;
- retrieval separate from extraction;
- bounded evidence bundles;
- capability-neutral extractor interfaces;
- independent validation before value promotion;
- local-first operation with optional hosted capabilities;
- no Enfield-specific rules in the core architecture.

## Sprint Sequence

| Sprint | Objective | Scope | Completion gate |
|---:|---|---|---|
| 2 | Dictionary ingestion and target normalization | Parse selected dictionary formats into normalized target specifications. | Synthetic dictionary fixtures produce validated target spec sets. |
| 3 | Source ingestion and document classification | Register source packs, hash files and classify documents. | Generic manifests and document classifications serialize and validate. |
| 4 | Parsing and page-level evidence classification | Add parser interfaces and page/evidence-type classification contracts. | Synthetic parsed pages carry text/table/drawing/image evidence labels. |
| 5 | Hierarchical PageIndex-style evidence index | Build provider-neutral hierarchy from parsed sources. | Synthetic long-manual hierarchy validates with rebuild metadata. |
| 6 | Node enrichment and domain/component classification | Enrich nodes with domain, component and manufacturer/model signals. | Domain/component classifier results attach to hierarchy nodes. |
| 7 | Hybrid retrieval, reranking, and evidence bundles | Retrieve/rerank documents, sections and pages into bounded bundles. | Evidence bundles meet target budgets and preserve provenance. |
| 8 | Route classification and observability implementation | Classify extraction routes and emit run/stage telemetry. | Route decisions and telemetry are serialized for each bundle. |
| 9 | Deterministic and structured-table extraction | Implement first deterministic/table capabilities on bounded bundles. | First working extraction on synthetic fixtures. |
| 10 | Text LLM extraction | Add mocked/tested text LLM capability behind hosted-call guardrails. | LLM candidates are schema-validated with no unmocked test calls. |
| 11 | VLM and OCR-assisted extraction | Add visual/OCR capability interfaces and bounded evidence use. | Synthetic drawings/scans produce candidate results through fakes/mocks. |
| 12 | Independent validation and promotion | Implement evidence, schema, unit, component and conflict validation. | Candidates are promoted only through validation decisions. |
| 13 | End-to-end orchestration and target-aligned output | Wire clean flow and output alignment. | Extraction engine engineering-complete on synthetic fixtures. |
| 14 | Synthetic clean-room benchmark | Build broader synthetic benchmark across mixed source packs. | Benchmark reports coverage, retrieval, validation and cost telemetry. |
| 15 | Customer output ingestion and comparison normalization | Ingest customer outputs for evaluation-only comparison. | Customer outputs are normalized outside the core extraction flow. |
| 16 | Enfield retrieval and classifier evaluation | Evaluate retrieval/classifiers on Enfield without fixed answers. | Retrieval and classifier metrics are reported without extraction claims. |
| 17 | Full Enfield clean extraction and customer comparison | Run clean extraction and compare with customer outputs. | First full Enfield extraction and comparison report. |
| 18 | Accuracy, coverage, and cost adjudication | Adjudicate Enfield values, coverage gaps and cost. | Accuracy- and cost-adjudicated Enfield result. |
| 19 | Operational hardening and portfolio readiness | Harden config, telemetry, packaging, runners and portfolio handling. | Portfolio-ready operational release candidate. |

## Milestone Summary

- First working extraction: Sprint 9.
- Extraction engine engineering-complete: Sprint 13.
- Customer outputs available for evaluation: Sprint 15.
- First full Enfield extraction and comparison: Sprint 17.
- Accuracy- and cost-adjudicated Enfield result: Sprint 18.

## Change Log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-07-29 | Initial authoritative Sprint 2-19 working plan recorded at Foundation Sprint 1 closure. |
| 1.1 | 2026-07-29 | Sprint 2 dictionary ingestion and target normalization marked complete; Sprint 3 not started. |
| 1.2 | 2026-07-29 | Sprint 3 source ingestion and document classification marked complete; Sprint 4 not started. |

This file is the authoritative working plan and must be revised whenever sprint scope or ordering
changes.
