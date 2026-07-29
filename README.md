# SEGRO Evidence-First Extraction

This repository is the clean foundation for the SEGRO evidence-first asset data extraction solution.

The package is independent from the historical repository at `D:\Segro_PoC_codex`. That repository may be inspected for reference only; it is not a runtime dependency and is not imported.

## Evidence-First Principles

The engine is being designed around bounded evidence before extraction:

1. Normalize selected data dictionary rows into target specifications.
2. Ingest and classify arbitrary asset source packs.
3. Build hierarchical document, section, page, table, drawing and region indexes.
4. Retrieve and rerank evidence separately from extraction.
5. Build bounded, capability-neutral evidence bundles.
6. Route bundles to deterministic, table, text LLM, VLM, OCR or human review capabilities.
7. Validate candidate values independently before output alignment.

Extraction is not implemented in Foundation Sprint 1.

## Setup

Use Python 3.11 or later. A local virtual environment is expected at `.venv`.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The `.env` file is local and excluded from Git. Keep API keys environment-only. Do not commit customer files, source packs, generated indexes, caches or outputs.

## CLI

```powershell
python -m segro_evidence_extraction --help
python -m segro_evidence_extraction config-check --config-path configs/default.yaml
python -m segro_evidence_extraction dictionary inspect `
  --dictionary-path "data/input/data_dictionary/SEGRO_Extraction_Template.xlsx" `
  --sheet-name "Extraction Template"
python -m segro_evidence_extraction dictionary validate `
  --dictionary-path "data/input/data_dictionary/SEGRO_Extraction_Template.xlsx" `
  --sheet-name "Extraction Template" `
  --mapping-config "configs/dictionaries/segro_extraction_template_v1.yaml" `
  --output-dir "output/sprint2_dictionary_validation"
python -m segro_evidence_extraction extract `
  --dictionary-path "data/input/data_dictionary/SEGRO_Extraction_Template.xlsx" `
  --asset-config "configs/assets/example.yaml" `
  --output-dir "output/example"
```

The `extract` command validates arguments and exits non-zero because the pipeline is intentionally unimplemented in this sprint.

## Development

```powershell
python -m ruff check .
python -m mypy src
python -m pytest
```

## Local Data

Local inputs are excluded from Git:

- `data/input/data_dictionary/`
- `data/input/source_packs/`
- `data/cache/`
- `data/indexes/`
- `output/`

Customer files must remain outside version control.

## Documentation

- [Architecture](docs/architecture.md)
- [ADR-001 Evidence-First Architecture](docs/decisions/ADR-001-evidence-first-architecture.md)
- [Historical Reuse Assessment](docs/reuse-assessment.md)

## Current Status

Implemented:

- Python `src` package foundation
- safe configuration loading
- XLSX and CSV dictionary ingestion
- configurable dictionary column mapping
- deterministic target normalization artifacts
- target specification models
- source registry models
- hierarchical evidence index contracts
- capability-neutral evidence bundle contracts
- extraction capability protocol
- candidate extraction result models
- independent validation result models
- provider-neutral telemetry models
- CLI skeleton
- focused tests

Not implemented:

- source-pack parsing
- OCR, embedding, retrieval or reranking engines
- deterministic, table, LLM, VLM or OCR extraction
- output template writing
