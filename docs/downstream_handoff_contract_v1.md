# Downstream Handoff Contract V1

## Scope

This document freezes the V1 contracts for:

- `segro_internal_evidence_rich_handoff_v1`
- `segro_customer_candidate_handoff_v1`
- downstream reason codes
- validation and compatibility behaviour
- artifact versioning
- missing metadata handling

The V1 exporter remains a deterministic handoff producer. It is not the final SEGRO customer-format transformation.

## Contract Artifacts

Tracked contract artifacts live under `schemas/downstream_handoff/`:

- `segro_internal_evidence_rich_handoff_v1.schema.json`
- `segro_customer_candidate_handoff_v1.schema.json`
- `reason_codes_v1.json`
- `contract_manifest_v1.json`

The reusable validator is `src/segro_evidence_extraction/downstream_handoff_contract_v1.py`.

## Internal Evidence-Rich Handoff

The internal handoff retains complete evidence, provenance, validation, model, adjudication and repair details for every adjudicated target, including accepted, caveated, abstained and rejected records.

Always present and non-null:

- `schema_version`
- `trace_id`
- `target_id`
- `requirement_id`
- `field_name`
- `final_decision`
- `reason_code`
- `promotable`
- `promotion_status`
- `evidence_containment_status`
- `event_validation_status`
- `dictionary_validation_status`
- `extraction_run_id`
- `adjudication_run_id`
- `input_provenance`
- `source_id`
- `source_file`
- `page_number`

Always present but nullable:

- `model`
- `source_path`

Required for successful outcomes:

- `display_value`
- `final_value`
- `raw_model_value`
- `evidence_value`

Required for accepted-with-caveat outcomes:

- either `checkpoint_caveat` or `extraction_caveat`

Required for rejected or abstained outcomes:

- `decision_reason`
- no promoted `display_value`
- no promoted `final_value`

Optional metadata may be added under V1 only when additive and nullable.

## Customer Candidate Handoff

The customer candidate handoff is a thin traceable candidate record. It must not fabricate customer mappings.

Always present:

- `schema_version`
- `trace_id`
- `asset_record_key`
- `requirement_id`
- `target_id`
- `field_name`
- `customer_field_label`
- `display_value`
- `status`
- `reason_code`
- `caveat`
- `source_reference`
- `internal_record_identity`
- `transformation_status`

Nullable mapping fields must remain present and null until authoritative mappings exist:

- `asset_record_key`
- `customer_field_label`
- `source_reference.source_url`

Successful records may carry `display_value`. Abstained and rejected records must not carry a promoted value.

## Decisions And Reason Codes

Allowed decisions:

- `accepted`
- `accepted_with_caveat`
- `abstained`
- `rejected`

Allowed reason codes:

- `accepted`
- `accepted_with_caveat`
- `dictionary_value_not_supported`
- `insufficient_attribute_evidence`
- `evidence_containment_failed`
- `response_schema_invalid`
- `wrong_event`
- `wrong_system`
- `component_only`

Valid relationships are defined in `reason_codes_v1.json`. Unknown reason codes and incompatible decision/reason combinations are errors.

## Missing Metadata Policy

`model` is required but nullable. Null is valid only when unavailable from authoritative upstream artifacts. The validator emits `missing_model_metadata`. Future extraction workflows must propagate provider and model metadata into final adjudication. Old artifacts must not be backfilled with guessed values.

`source_path` is required but nullable. Null is valid only when `source_id` and `source_file` remain present. The validator emits `missing_source_path`. Future ingestion and adjudication workflows must propagate the registered source path. Paths and document URLs must not be invented.

## Compatibility Policy

Schema definitions, contract manifests, validators and tests are tracked in Git. Full unit-specific runtime exports are not tracked by default. Small sanitized golden fixtures may be tracked.

Every export records schema version, contract version, exporter version, input hashes, generation timestamp or deterministic run identity where available.

Breaking changes require a new major schema version. Additive nullable fields may be introduced under V1 compatibility rules. Removal, renaming or semantic repurposing of fields requires a new major version.

## Handoff Layers

The internal evidence-rich handoff is the authoritative downstream audit layer.

The customer candidate handoff is a thin candidate value layer with traceability to the internal record.

The final customer-format transformation is not implemented in V1. It requires authoritative customer field labels, asset/entity record keys, ordering/grouping, date display policy, boolean rendering policy, caveat visibility policy and source URL/document-link mappings.
