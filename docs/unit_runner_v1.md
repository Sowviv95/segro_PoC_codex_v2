# Unit Runner V1

Unit Runner V1 runs the existing bounded Enfield Unit 1 workflow from one
PowerShell-friendly command. It consumes an already prepared selected batch and
does not rebuild source coverage, PageIndex artifacts, retrieval results,
evidence mappings or target selections.

## Modes

```powershell
python -m segro_evidence_extraction run-unit --config "config\enfield_unit1_runner_v1.json" --contract-only
python -m segro_evidence_extraction run-unit --config "config\enfield_unit1_runner_v1.json" --dry-run
python -m segro_evidence_extraction run-unit --config "config\enfield_unit1_runner_v1.json" --execute
```

`contract-only` validates the runner config, inputs, selected batch, extraction
contract readiness and frozen handoff contract availability. It writes skipped
manifests for extraction, adjudication and export, and makes zero model calls.

`dry-run` validates config, inputs and extraction preflight. It makes zero model
calls and reports whether the unit is execution-ready.

`execute` runs the configured selected batch through the existing bounded
extraction, deterministic adjudication, downstream handoff export and frozen
contract validation.

## PowerShell Wrapper

```powershell
.\scripts\run_enfield_unit1_v1.ps1 -ContractOnly
.\scripts\run_enfield_unit1_v1.ps1 -DryRun
.\scripts\run_enfield_unit1_v1.ps1 -Execute
.\scripts\run_enfield_unit1_v1.ps1 -Execute -Resume
```

The wrapper verifies the Git branch, verifies the config exists, uses a local
virtual environment only when required runtime packages are installed, sets
`PYTHONPATH=src`, and exits with the Python process exit code.

## Configuration

`config/enfield_unit1_runner_v1.json` is validated first against
`schemas/unit_runner/segro_unit_runner_v1.schema.json`, then semantically against
repository and artifact state. The config records the unit ID, optional asset
record key, selected batch, source registry, cache root, dictionary artifact,
output root, model provider/name, expected branch, customer caveat policy, retry
limit and frozen contract versions. It must not contain API keys or secrets.

## Exit Codes

- `0`: success
- `2`: configuration or JSON Schema validation failure
- `3`: repository/input/artifact validation failure
- `4`: preflight failure
- `5`: extraction/provider failure
- `6`: adjudication failure
- `7`: export failure
- `8`: frozen contract validation failure
- `9`: resume-state conflict

## Output Layout

Runner outputs are written under:

```text
output/unit_runs/<unit_id>/<run_id>/
```

Each run writes `run_manifest.json`, `config_snapshot.json`, stage manifests,
`execution_summary.json` and `run_log.jsonl`. Execute mode also writes nested
bounded extraction, adjudication and downstream handoff export packages.

## Run Identity

The run ID is deterministic over stable inputs: unit ID, runner/config version,
config hash, selected-batch hash, dictionary hash, frozen contract versions and
mode. Timestamps are recorded separately and are not part of the deterministic
identity.

## Resume Rules

Resume reuses completed stages only when fingerprints still match. Safe resume
paths are:

- preflight complete and extraction not started: start extraction;
- extraction complete with all expected raw responses: resume from adjudication;
- adjudication complete: resume from export;
- export complete: resume from contract validation;
- all stages complete and fingerprints match: reuse the completed run.

The runner fails with exit code `9` when provider state is ambiguous: partial
responses, duplicate/missing responses, response-count mismatches, changed input
fingerprints, corrupted manifests or artifacts from another run ID.

## Troubleshooting

If the wrapper cannot import the package from `.venv`, it falls back to the
current Python and sets `PYTHONPATH=src`. Hosted model credentials are read via
the existing environment mechanism; never place secrets in the runner config.
