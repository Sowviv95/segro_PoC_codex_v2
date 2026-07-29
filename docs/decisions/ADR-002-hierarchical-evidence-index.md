# ADR-002 Hierarchical Evidence Index

## Status

Accepted for Foundation Sprint 1 closure.

## Context

SEGRO source packs can include long O&M manuals, combined manuals, supplier appendices,
certificates, drawings, spreadsheets, scanned files, ZIP archives and weakly organized content.
Some manuals may exceed 2,000 pages. The engine needs bounded evidence retrieval without assuming
manual Part 1-6 naming, fixed page mappings, a specific embedding provider, a vector database or a
model provider.

Flat chunk retrieval is useful as a baseline and fallback, but it is not sufficient as the only
indexing abstraction. It loses document, section, subsection, page/sheet and evidence-unit context;
it struggles with repeated supplier content; and it can return plausible matches from the wrong
component, table, certificate or drawing.

## Decision

Implement an independent PageIndex-style hierarchical evidence index. Adapt selected concepts from
PageIndex-style systems and from the historical repository's simple page index, but do not add the
external PageIndex package as a dependency.

The hierarchy is:

```text
source pack
  document
    section
      subsection
        page or sheet
          text block
          table
          drawing
          image region
```

Bookmarks and tables of contents should seed hierarchy where available. Where explicit structure is
absent, hierarchy may be inferred from headings, page labels, layout, repeated headers/footers, table
captions, drawing titles, OCR signals and classifier output. Low-confidence boundaries must be
represented explicitly with warnings rather than hidden.

## Consequences

The architecture can navigate from target specification to document, section, page/sheet and precise
evidence unit. The same index can support retrieval, reranking, evidence bundles, extraction,
validation and human review.

The implementation must also support fallback retrieval against page-only or flat candidates when
hierarchy quality is weak. Incorrect section boundaries are a known risk and must not be allowed to
silently suppress relevant pages.

## Guardrails

- No dependency on the external PageIndex package.
- No dependency on a specific vector database, embedding provider or model provider.
- Source hashes drive incremental rebuild decisions.
- Provider-neutral persistence stores metadata and references, not provider-specific assumptions.
- Whole manuals must not be sent to LLMs or VLMs.

## Observability

Hierarchy construction and retrieval must report document counts, page/sheet counts, bookmark/TOC
usage, inferred-node counts, low-confidence boundaries, OCR-required pages, table/drawing indicators,
duration, warnings, source-hash cache hits, rebuild counts and fallback-retrieval usage.

## Later Baseline Comparison

A later sprint must compare hierarchical retrieval with a flat-retrieval baseline on synthetic clean
room packs and Enfield evaluation data. Metrics should include recall at bundle size, wrong-component
rate, bundle token/page budget, retrieval latency, rebuild cost, validation rejection rate and human
review burden.

## Alternatives Considered

- Flat chunk retrieval only: rejected as the sole abstraction because it loses source hierarchy and
  component context.
- External PageIndex dependency: rejected for foundation because it would couple contracts and
  persistence too early.
- Provider-specific vector index: rejected because evidence contracts must outlive provider choices.
- Manual-specific fixed sections: rejected because arbitrary source packs may be combined, renamed or
  missing expected categories.

## Recommendation

Adopt an independent PageIndex-style hierarchy and adapt selected concepts. Keep the architecture
evidence-based by validating it against flat retrieval in a later sprint before claiming operational
superiority.
