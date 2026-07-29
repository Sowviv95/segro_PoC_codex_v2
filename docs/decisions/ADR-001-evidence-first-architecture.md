# ADR-001 Evidence-First Architecture

## Status

Accepted for Foundation Sprint 1.

## Context

The historical implementation proved that SEGRO asset data can be extracted, reviewed and delivered, but deterministic extraction became the dominant operational route and retrieval, extraction and validation concerns were too tightly coupled. The new repository must support arbitrary supported source packs, combined manuals, supplier manuals, certificates, drawings, spreadsheets, scanned files, ZIPs and missing or additional categories.

## Decision

Use an evidence-first architecture:

- normalize dictionary rows into target specifications;
- ingest sources into a generic source registry;
- build hierarchical evidence indexes;
- retrieve and rerank evidence independently of extraction;
- pass bounded evidence bundles to capability-specific extractors;
- validate candidate values independently before QC and output alignment.

## Alternatives Considered

- Fixed Part 1-6 orchestration as the core abstraction: rejected because source packs may be differently named, combined or missing categories.
- Deterministic extraction as a universal default: rejected because it overfits known layouts and underuses LLM/VLM/OCR/table capabilities.
- Flat chunk search as the only retrieval method: rejected because manuals, drawings, sheets and tables need document and section context.
- Prior-result dependence in clean extraction: rejected because new assets must work without reviewed historical outputs.
- Sending whole manuals to LLMs or VLMs: rejected because it is costly, poorly bounded and hard to validate.
- Promotion without independent validation: rejected because extraction confidence is not the same as evidence-grounded correctness.

## Consequences

The foundation is more contract-driven than feature-complete. Sprint 1 produces schemas, protocols, CLI shape and tests, not extraction coverage. Future implementation can add retrieval and extraction engines without changing the core bundle and validation contracts.

## Guardrails

- Enfield is an evaluation dataset, not the architecture.
- API keys remain environment-only.
- Hosted model calls require explicit future implementation and tests with mocks/fakes.
- Evidence bundles must stay capability-neutral.
- Retrieval outputs must remain reusable by validation and human review.

## Implications for Testing

Foundation tests focus on validation, serialization, protocol compatibility, telemetry aggregation and CLI behavior. Future tests should add fixture-driven dictionary/source ingestion, retrieval quality, capability routing, validation independence and output alignment.

## Relationship to Historical Implementation

Historical source manifests, PageIndex models, extraction envelopes, confidence/telemetry fields, pricing configuration and synthetic fixtures informed this design. Historical deterministic orchestration, Enfield-specific routing, fixed part assumptions and reviewed-result dependence are intentionally not ported.
