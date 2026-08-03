# Unit Runner V1 Requirements

## Proposed Command Shape

```powershell
python -m segro_evidence_extraction run-unit `
  --input-dir "<unit-source-directory>" `
  --dictionary "<extraction-template>" `
  --output-dir "<unit-output-directory>"
```

## Required Stages

- source registration
- document classification
- page-index/cache preparation
- evidence retrieval
- batch construction
- extraction
- validation
- adjudication
- internal export
- customer-candidate export
- execution summary

## Operational Features

- dry-run mode
- resume support
- bounded target mode
- no-model validation mode
- configuration file support
- deterministic run identity
- PowerShell-friendly logging
- deterministic exit codes
- stage manifests
- token and model usage reporting
- failure recovery
- no silent partial success

## Exit-Code Expectations

- `0`: completed successfully
- `1`: runtime failure
- `2`: contract or validation failure
- `3`: unsupported or unavailable stage
- `4`: partial result blocked and explicitly recorded

## Preparation Notes

The runner must treat source registration, evidence preparation, extraction, adjudication and handoff export as separately resumable stages. Runtime artifacts should include input hashes, stage manifests, invocation settings, model usage, token estimates or usage, and failure records.

The runner must support bounded execution from a prepared selected batch without repeating target selection, evidence mapping, parsing, retrieval, OCR, VLM escalation or cache expansion unless the operator explicitly enables those stages.
