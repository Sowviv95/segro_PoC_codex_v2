"""Final adjudication for Evidence-First Batch V1 artifacts."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.target_semantics import derive_target_intent
from segro_evidence_extraction.vertical_slice import _atomic_write_json

DEFAULT_FINAL_ADJUDICATION_OUTPUT_DIR = Path(
    "output/enfield_unit1_evidence_first_batch_v1_final_adjudication"
)
DEFAULT_BATCH_V1_INPUT_DIR = Path("output/enfield_unit1_evidence_first_batch_v1")
DEFAULT_DIAGNOSTIC_INPUT_DIR = Path(
    "output/enfield_unit1_evidence_first_batch_v1_retrieval_diagnostic"
)
DEFAULT_REEXTRACT_V1_INPUT_DIR = Path("output/enfield_unit1_evidence_first_batch_v1_reextract_v1")
DEFAULT_REEXTRACT_V2_INPUT_DIR = Path("output/enfield_unit1_evidence_first_batch_v1_reextract_v2")
DEFAULT_ESCALATION_INPUT_DIR = Path("output/enfield_unit1_evidence_escalation_v1")
DEFAULT_ESCALATION_TEXT_INPUT_DIR = Path(
    "output/enfield_unit1_evidence_escalation_text_reextract_v1"
)

EXPECTED_BATCH_TARGET_COUNT = 75
ACCEPTED_DISPOSITIONS = {
    "accepted",
    "accepted_after_normalization",
    "accepted_with_dictionary_caveat",
}
NON_NULL_EXTRACTED_STATUS = "extracted"

FinalDisposition = Literal[
    "accepted",
    "accepted_after_normalization",
    "accepted_with_dictionary_caveat",
    "manual_visual_review",
    "unsupported_in_supplied_sources",
    "dictionary_target_ambiguous",
    "component_present_attribute_absent",
    "insufficient_evidence",
    "conflicting_evidence",
    "invalid_prior_extraction",
]


class AdjudicationInputs(StrictBaseModel):
    batch_v1_dir: Path = DEFAULT_BATCH_V1_INPUT_DIR
    diagnostic_dir: Path = DEFAULT_DIAGNOSTIC_INPUT_DIR
    reextract_v1_dir: Path = DEFAULT_REEXTRACT_V1_INPUT_DIR
    reextract_v2_dir: Path = DEFAULT_REEXTRACT_V2_INPUT_DIR
    escalation_dir: Path = DEFAULT_ESCALATION_INPUT_DIR
    escalation_text_dir: Path = DEFAULT_ESCALATION_TEXT_INPUT_DIR


class ArtifactDecision(StrictBaseModel):
    stage: str
    artifact: str
    target_row_id: str
    status: str
    value: Any = None
    evidence_reference: str | None = None
    reason: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class FinalAdjudicationResult(StrictBaseModel):
    output_dir: str
    final_target_dispositions: list[dict[str, Any]]
    accepted_values: list[dict[str, Any]]
    manual_review_queue: list[dict[str, Any]]
    unsupported_targets: list[dict[str, Any]]
    ambiguous_targets: list[dict[str, Any]]
    rejected_prior_extractions: list[dict[str, Any]]
    evidence_audit: list[dict[str, Any]]
    disposition_trace: list[dict[str, Any]]
    final_metrics: dict[str, Any]


def run_batch_v1_final_adjudication(
    *,
    batch_v1_dir: Path = DEFAULT_BATCH_V1_INPUT_DIR,
    diagnostic_dir: Path = DEFAULT_DIAGNOSTIC_INPUT_DIR,
    reextract_v1_dir: Path = DEFAULT_REEXTRACT_V1_INPUT_DIR,
    reextract_v2_dir: Path = DEFAULT_REEXTRACT_V2_INPUT_DIR,
    escalation_dir: Path = DEFAULT_ESCALATION_INPUT_DIR,
    escalation_text_dir: Path = DEFAULT_ESCALATION_TEXT_INPUT_DIR,
    output_dir: Path = DEFAULT_FINAL_ADJUDICATION_OUTPUT_DIR,
    expected_target_count: int = EXPECTED_BATCH_TARGET_COUNT,
) -> FinalAdjudicationResult:
    """Consolidate all prior Batch V1 decisions into one final delivery pack."""

    inputs = AdjudicationInputs(
        batch_v1_dir=batch_v1_dir,
        diagnostic_dir=diagnostic_dir,
        reextract_v1_dir=reextract_v1_dir,
        reextract_v2_dir=reextract_v2_dir,
        escalation_dir=escalation_dir,
        escalation_text_dir=escalation_text_dir,
    )
    artifacts = LoadedArtifacts.load(inputs)
    records = adjudicate_loaded_artifacts(artifacts, expected_target_count=expected_target_count)
    write_final_adjudication_outputs(records, artifacts, output_dir)
    metrics = build_final_metrics(records, artifacts)
    result = FinalAdjudicationResult(
        output_dir=str(output_dir),
        final_target_dispositions=records,
        accepted_values=accepted_values(records),
        manual_review_queue=manual_review_queue(records),
        unsupported_targets=unsupported_targets(records),
        ambiguous_targets=ambiguous_targets(records),
        rejected_prior_extractions=rejected_prior_extractions(records, artifacts),
        evidence_audit=evidence_audit(records, artifacts),
        disposition_trace=disposition_trace(records, artifacts),
        final_metrics=metrics,
    )
    # Write metrics last from the returned object to ensure CLI and artifacts agree.
    _atomic_write_json(output_dir / "final_metrics.json", metrics)
    return result


class LoadedArtifacts:
    def __init__(self, inputs: AdjudicationInputs) -> None:
        self.inputs = inputs
        self.file_inventory = inventory_input_files(inputs)
        self.selected_targets_raw = _read_list(inputs.batch_v1_dir / "selected_targets.json")
        self.selected_targets = [
            TargetSpecification.model_validate(item["target"])
            for item in self.selected_targets_raw
        ]
        self.targets_by_id = {target.target_row_id: target for target in self.selected_targets}
        self.selected_raw_by_id = {
            item["target"]["target_row_id"]: item for item in self.selected_targets_raw
        }
        self.batch_extractions = _by_id(
            _read_list(inputs.batch_v1_dir / "extraction_results.json"),
            "target_row_id",
            "Batch V1 extraction results",
        )
        self.batch_validation = _by_id(
            _read_list(inputs.batch_v1_dir / "validation_results.json"),
            "target_row_id",
            "Batch V1 validation results",
        )
        self.batch_evidence_validation = _by_id(
            _read_list(inputs.batch_v1_dir / "evidence_validation.json"),
            "target_row_id",
            "Batch V1 evidence validation",
        )
        self.batch_shape_validation = _by_id(
            _read_list(inputs.batch_v1_dir / "shape_validation.json"),
            "target_row_id",
            "Batch V1 shape validation",
        )
        self.batch_schema = _by_id(
            _read_list(inputs.batch_v1_dir / "schema_compatibility_results.json"),
            "target_row_id",
            "Batch V1 schema compatibility",
        )
        self.batch_normalized = _by_id(
            _read_list(inputs.batch_v1_dir / "normalized_values.json"),
            "target_row_id",
            "Batch V1 normalized values",
        )
        self.batch_retrieval = _by_id(
            _read_list(inputs.batch_v1_dir / "retrieval_results.json"),
            "target_row_id",
            "Batch V1 retrieval results",
        )
        self.batch_evidence_spans = _by_id(
            _read_list(inputs.batch_v1_dir / "evidence_spans.json"),
            "span_id",
            "Batch V1 evidence spans",
        )
        self.diagnostic_eligibility = _by_id(
            _read_list(inputs.diagnostic_dir / "retrieval_eligibility.json"),
            "target_row_id",
            "retrieval eligibility",
        )
        self.diagnostic_audit = _by_id(
            _read_list(inputs.diagnostic_dir / "extracted_result_attribute_audit.json"),
            "target_row_id",
            "extracted result attribute audit",
            allow_missing_for_targets=True,
        )
        self.diagnostic_current = _by_id(
            _read_list(inputs.diagnostic_dir / "target_failure_classification.json"),
            "target_row_id",
            "target failure classification",
            allow_missing_for_targets=True,
        )
        self.target_semantics = _by_id(
            _read_list(inputs.diagnostic_dir / "target_semantics.json"),
            "target_row_id",
            "target semantics",
            allow_missing_for_targets=True,
        )
        self.reextract_v1 = _by_id(
            _read_list(inputs.reextract_v1_dir / "extraction_results.json"),
            "target_row_id",
            "re-extraction V1 results",
            allow_missing_for_targets=True,
        )
        self.reextract_v2 = _by_id(
            _read_list(inputs.reextract_v2_dir / "extraction_results.json"),
            "target_row_id",
            "re-extraction V2 results",
            allow_missing_for_targets=True,
        )
        self.escalation_decisions = _by_id(
            _read_list(inputs.escalation_dir / "escalation_decisions.json"),
            "target_row_id",
            "evidence escalation decisions",
            allow_missing_for_targets=True,
        )
        self.visual_triage = _by_id(
            _read_list(inputs.escalation_dir / "visual_page_triage.json"),
            "target_row_id",
            "visual page triage",
            allow_missing_for_targets=True,
        )
        self.reextract_readiness = _by_id(
            _read_list(inputs.escalation_dir / "reextract_readiness.json"),
            "target_row_id",
            "re-extraction readiness",
            allow_missing_for_targets=True,
        )
        self.text_reextract = _by_id(
            _read_list(inputs.escalation_text_dir / "extraction_results.json"),
            "target_row_id",
            "strict text re-extraction results",
            allow_missing_for_targets=True,
        )
        self.text_preflight = _read_json(inputs.escalation_text_dir / "preflight_report.json")
        self.approved_evidence = _read_list(inputs.escalation_text_dir / "approved_evidence.json")

    @classmethod
    def load(cls, inputs: AdjudicationInputs) -> LoadedArtifacts:
        required = [
            inputs.batch_v1_dir / "selected_targets.json",
            inputs.batch_v1_dir / "extraction_results.json",
            inputs.batch_v1_dir / "validation_results.json",
            inputs.batch_v1_dir / "evidence_validation.json",
            inputs.batch_v1_dir / "shape_validation.json",
            inputs.batch_v1_dir / "schema_compatibility_results.json",
            inputs.batch_v1_dir / "normalized_values.json",
            inputs.batch_v1_dir / "retrieval_results.json",
            inputs.batch_v1_dir / "evidence_spans.json",
            inputs.diagnostic_dir / "retrieval_eligibility.json",
            inputs.diagnostic_dir / "extracted_result_attribute_audit.json",
            inputs.diagnostic_dir / "target_failure_classification.json",
            inputs.diagnostic_dir / "target_semantics.json",
            inputs.reextract_v1_dir / "extraction_results.json",
            inputs.reextract_v2_dir / "extraction_results.json",
            inputs.escalation_dir / "escalation_decisions.json",
            inputs.escalation_dir / "visual_page_triage.json",
            inputs.escalation_dir / "reextract_readiness.json",
            inputs.escalation_text_dir / "extraction_results.json",
            inputs.escalation_text_dir / "preflight_report.json",
            inputs.escalation_text_dir / "approved_evidence.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            msg = f"Final adjudication inputs missing: {missing}"
            raise FileNotFoundError(msg)
        return cls(inputs)


def adjudicate_loaded_artifacts(
    artifacts: LoadedArtifacts,
    *,
    expected_target_count: int = EXPECTED_BATCH_TARGET_COUNT,
) -> list[dict[str, Any]]:
    validate_target_coverage(artifacts, expected_target_count=expected_target_count)
    rows: list[dict[str, Any]] = []
    for target in sorted_targets(artifacts.selected_targets):
        target_id = target.target_row_id
        disposition, winning_artifact, precedence_rule, rationale = decide_target(
            target_id, artifacts
        )
        row = build_final_record(
            target=target,
            artifacts=artifacts,
            final_disposition=disposition,
            winning_artifact=winning_artifact,
            precedence_rule=precedence_rule,
            decision_rationale=rationale,
        )
        rows.append(row)
    validate_final_records(rows, artifacts)
    return rows


def decide_target(
    target_id: str,
    artifacts: LoadedArtifacts,
) -> tuple[FinalDisposition, str, str, str]:
    batch = artifacts.batch_extractions[target_id]
    validation = artifacts.batch_validation[target_id]
    eligibility = artifacts.diagnostic_eligibility[target_id]
    audit = artifacts.diagnostic_audit.get(target_id)
    escalation = artifacts.escalation_decisions.get(target_id)
    text_reextract = artifacts.text_reextract.get(target_id)
    reextract_v2 = artifacts.reextract_v2.get(target_id)

    if text_reextract and text_reextract.get("status") == "insufficient_evidence":
        return (
            "insufficient_evidence",
            "output/enfield_unit1_evidence_escalation_text_reextract_v1/extraction_results.json",
            "strict_local_evidence_gate_overrides_prior_attempts",
            str(
                text_reextract.get("ambiguity_or_caveat")
                or "Strict text re-extraction found no locally grounded requested-attribute span."
            ),
        )

    if escalation and escalation.get("final_status") == "manual_visual_review":
        return (
            "manual_visual_review",
            "output/enfield_unit1_evidence_escalation_v1/escalation_decisions.json",
            "manual_visual_review_routing_overrides_generic_text_abstention",
            str(
                escalation.get("recommended_next_action")
                or escalation.get("notes")
                or "Evidence escalation routed this target to manual visual review."
            ),
        )

    if reextract_v2 and reextract_v2.get("status") == "insufficient_evidence":
        return (
            "insufficient_evidence",
            "output/enfield_unit1_evidence_first_batch_v1_reextract_v2/extraction_results.json",
            "structured_model_abstention_overrides_unsupported_prior_value",
            (
                "Strict structured re-extraction abstained; available evidence did not "
                "prove the requested attribute."
            ),
        )

    if audit and audit.get("audit_classification") == "correctly_supported":
        return accepted_from_validation(validation, audit, with_caveat=False)

    if audit and audit.get("audit_classification") == "supported_with_caveat":
        return accepted_from_validation(validation, audit, with_caveat=True)

    if eligibility.get("status") == "component_present_attribute_absent":
        return (
            "component_present_attribute_absent",
            "output/enfield_unit1_evidence_first_batch_v1_retrieval_diagnostic/retrieval_eligibility.json",
            "component_present_attribute_absent_overrides_generic_unsupported",
            str(eligibility.get("reason") or "Component present but requested attribute absent."),
        )

    if eligibility.get("status") in {"dictionary_target_ambiguous", "eligible_with_ambiguity"}:
        return (
            "dictionary_target_ambiguous",
            "output/enfield_unit1_evidence_first_batch_v1_retrieval_diagnostic/retrieval_eligibility.json",
            "dictionary_ambiguity_remains_visible",
            str(
                eligibility.get("reason")
                or "Target definition contains ambiguity that prevents final acceptance."
            ),
        )

    if eligibility.get("status") == "conflicting_evidence":
        return (
            "conflicting_evidence",
            "output/enfield_unit1_evidence_first_batch_v1_retrieval_diagnostic/retrieval_eligibility.json",
            "conflicting_evidence_is_not_silently_resolved",
            str(eligibility.get("reason") or "Diagnostic recorded conflicting evidence."),
        )

    if eligibility.get("status") == "unsupported_in_available_sources":
        return (
            "unsupported_in_supplied_sources",
            "output/enfield_unit1_evidence_first_batch_v1_retrieval_diagnostic/retrieval_eligibility.json",
            "later_diagnostic_overrides_batch_v1_status",
            str(eligibility.get("reason") or "No requested-attribute support in supplied sources."),
        )

    if batch.get("status") == "insufficient_evidence":
        return (
            "insufficient_evidence",
            "output/enfield_unit1_evidence_first_batch_v1/extraction_results.json",
            "abstention_preferred_to_speculative_acceptance",
            str(
                batch.get("ambiguity_or_caveat")
                or "Batch V1 found insufficient evidence and no later artifact superseded it."
            ),
        )

    return (
        "invalid_prior_extraction",
        "output/enfield_unit1_evidence_first_batch_v1_retrieval_diagnostic/extracted_result_attribute_audit.json",
        "unsupported_prior_extraction_is_invalidated",
        str(
            audit.get("explanation")
            if audit
            else "Original extraction was not confirmed by later requested-attribute audit."
        ),
    )


def accepted_from_validation(
    validation: dict[str, Any],
    audit: dict[str, Any],
    *,
    with_caveat: bool,
) -> tuple[FinalDisposition, str, str, str]:
    status = str(validation.get("status") or "")
    if with_caveat or status == "valid_with_dictionary_caveat":
        disposition: FinalDisposition = "accepted_with_dictionary_caveat"
    elif status == "valid_after_normalization":
        disposition = "accepted_after_normalization"
    else:
        disposition = "accepted"
    return (
        disposition,
        "output/enfield_unit1_evidence_first_batch_v1_retrieval_diagnostic/extracted_result_attribute_audit.json",
        "strict_requested_attribute_support_allows_acceptance",
        str(audit.get("explanation") or "Diagnostic confirmed requested-attribute support."),
    )


def build_final_record(
    *,
    target: TargetSpecification,
    artifacts: LoadedArtifacts,
    final_disposition: FinalDisposition,
    winning_artifact: str,
    precedence_rule: str,
    decision_rationale: str,
) -> dict[str, Any]:
    target_id = target.target_row_id
    batch_extraction = artifacts.batch_extractions[target_id]
    batch_validation = artifacts.batch_validation[target_id]
    evidence_validation = artifacts.batch_evidence_validation[target_id]
    shape_validation = artifacts.batch_shape_validation[target_id]
    schema = artifacts.batch_schema[target_id]
    normalized = artifacts.batch_normalized[target_id]
    eligibility = artifacts.diagnostic_eligibility[target_id]
    audit = artifacts.diagnostic_audit.get(target_id)
    intent = artifacts.target_semantics.get(target_id) or derive_target_intent(target).model_dump(
        mode="json"
    )
    accepted = final_disposition in ACCEPTED_DISPOSITIONS
    provenance = target.source_dictionary_provenance
    source, page, node, excerpt, span_ref = accepted_evidence_fields(
        batch_extraction,
        artifacts,
    ) if accepted else ("", "", "", "", "")
    final_evidence_value, normalized_value, final_accepted_value, display_value = (
        materialize_accepted_values(
            target=target,
            intent=intent,
            extraction=batch_extraction,
            normalized=normalized,
            evidence_excerpt=str(excerpt),
        )
        if accepted
        else (None, None, None, None)
    )
    decision_rationale = accepted_value_rationale(
        base_rationale=decision_rationale,
        evidence_value=final_evidence_value,
        accepted_value=final_accepted_value,
        normalized_value=normalized_value,
    )
    record = {
        "target_id": target_id,
        "dictionary_row": provenance.row_number if provenance else None,
        "domain": target.metadata.get("domain") or target.sub_domain,
        "sub_domain": target.sub_domain,
        "field_name": target.expected_field,
        "definition": target.requirement_text,
        "datatype": target.expected_data_type,
        "unit": target.unit,
        "value_shape": intent.get("value_shape_family"),
        "requested_attribute_semantics": {
            "primary_component": intent.get("primary_component"),
            "requested_attribute": intent.get("requested_attribute"),
            "component_terms": intent.get("component_terms", []),
            "attribute_terms": intent.get("attribute_terms", []),
            "ambiguity_notes": intent.get("ambiguity_notes", []),
        },
        "prior_statuses": prior_statuses(target_id, artifacts),
        "latest_evidence_decision": latest_evidence_decision(target_id, artifacts),
        "final_disposition": final_disposition,
        "final_evidence_value": final_evidence_value,
        "final_accepted_value": final_accepted_value,
        "normalized_value": normalized_value,
        "display_value": display_value,
        "canonical_evidence_span_id": span_ref,
        "source_document": source,
        "page": page,
        "hierarchy_node": node,
        "evidence_excerpt": excerpt,
        "evidence_validation_status": evidence_validation.get("status"),
        "value_shape_validation_status": shape_validation.get("status"),
        "dictionary_compatibility_status": schema.get("compatibility"),
        "final_review_status": batch_validation.get("status") if accepted else final_disposition,
        "winning_decision_artifact": winning_artifact,
        "precedence_rule": precedence_rule,
        "decision_rationale": decision_rationale,
        "manual_review_reason": manual_review_reason(target_id, artifacts)
        if final_disposition == "manual_visual_review"
        else None,
        "unsupported_reason": decision_rationale
        if final_disposition
        in {
            "unsupported_in_supplied_sources",
            "component_present_attribute_absent",
            "insufficient_evidence",
        }
        else None,
        "ambiguity_reason": decision_rationale
        if final_disposition in {"dictionary_target_ambiguous", "conflicting_evidence"}
        else None,
        "diagnostic_attribute_support_audit": audit,
        "retrieval_eligibility": eligibility,
        "decision_source_artifact": winning_artifact,
    }
    if accepted:
        ensure_complete_accepted_record(record)
    return record


def accepted_evidence_fields(
    extraction: dict[str, Any],
    artifacts: LoadedArtifacts,
) -> tuple[str, int | str, str, str, str]:
    span_ids = extraction.get("supporting_span_ids") or []
    span_id = str(span_ids[0]) if span_ids else ""
    span = artifacts.batch_evidence_spans.get(span_id, {}) if span_id else {}
    source = extraction.get("source_file") or span.get("source_file") or ""
    page = extraction.get("page_number") or span.get("page_number") or ""
    node = extraction.get("hierarchy_node_id") or span.get("hierarchy_node_id") or ""
    excerpt = extraction.get("supporting_evidence_excerpt") or span.get("text") or ""
    return str(source), page, str(node), str(excerpt), span_id


def materialize_accepted_values(
    *,
    target: TargetSpecification,
    intent: dict[str, Any],
    extraction: dict[str, Any],
    normalized: dict[str, Any],
    evidence_excerpt: str,
) -> tuple[Any, Any, Any, Any]:
    """Separate verbatim evidence from the typed accepted value."""

    evidence_value = extraction.get("evidence_value")
    normalized_value = normalized.get("normalized_value")
    display_value = extraction.get("display_value")
    value_shape = str(intent.get("value_shape_family") or "")
    if value_shape == "integer_count" and isinstance(normalized_value, int):
        return evidence_value, normalized_value, normalized_value, str(normalized_value)

    accepted_value = evidence_value
    component_value = component_specific_description(
        intent=intent,
        evidence_value=evidence_value,
        evidence_excerpt=evidence_excerpt,
    )
    if component_value is not None:
        accepted_value = component_value
        normalized_value = component_value
        display_value = component_value
    elif normalized_value not in {None, ""} and str(target.expected_data_type) != "string":
        accepted_value = normalized_value
    return evidence_value, normalized_value, accepted_value, display_value


def component_specific_description(
    *,
    intent: dict[str, Any],
    evidence_value: Any,
    evidence_excerpt: str,
) -> str | None:
    requested_attribute = str(intent.get("requested_attribute") or "")
    component = str(intent.get("primary_component") or "").lower()
    if requested_attribute not in {"description", "type"} or not component:
        return None
    if isinstance(evidence_value, str) and component in evidence_value.lower():
        return None
    return component_description_phrase(component=component, evidence_excerpt=evidence_excerpt)


def component_description_phrase(*, component: str, evidence_excerpt: str) -> str | None:
    normalized_excerpt = " ".join(evidence_excerpt.split())
    if component == "roof":
        return roof_specific_description(normalized_excerpt)
    if component == "wall":
        return local_component_phrase(
            component_pattern=r"wall(?:\s+panels?)?",
            evidence_excerpt=normalized_excerpt,
        )
    return local_component_phrase(
        component_pattern=re.escape(component),
        evidence_excerpt=normalized_excerpt,
    )


def roof_specific_description(normalized_excerpt: str) -> str | None:
    match = re.search(
        r"(?:with|incorporating|including|comprising)\s+(?:a\s+|an\s+|the\s+)?"
        r"(?P<phrase>[A-Za-z][A-Za-z -]{0,80}\s+roof\b"
        r"(?:\s+and\s+[A-Za-z][A-Za-z -]{0,50}\s+rooflights\b|,\s*rooflights\b)?)",
        normalized_excerpt,
        flags=re.IGNORECASE,
    )
    if not match:
        return local_component_phrase(
            component_pattern=r"roof(?:lights)?",
            evidence_excerpt=normalized_excerpt,
        )
    phrase = " ".join(match.group("phrase").split())
    phrase = phrase.replace(", rooflights", " with rooflights")
    phrase = re.sub(r"\s+elevations\s+and\s+roof\b", " roof", phrase, flags=re.IGNORECASE)
    return normalized_phrase(phrase)


def local_component_phrase(*, component_pattern: str, evidence_excerpt: str) -> str | None:
    pattern = (
        r"(?:with|incorporating|including|comprising|and)\s+"
        r"(?:a\s+|an\s+|the\s+)?"
        rf"(?P<phrase>[A-Za-z][A-Za-z -]{{0,80}}\s+{component_pattern})"
    )
    match = re.search(pattern, evidence_excerpt, flags=re.IGNORECASE)
    if not match:
        return None
    phrase = " ".join(match.group("phrase").split())
    if not component_phrase_is_value_bearing(phrase):
        return None
    return normalized_phrase(phrase)


def component_phrase_is_value_bearing(phrase: str) -> bool:
    text = phrase.lower()
    if any(term in text for term in ["maintenance", "inspection", "cleaning", "required"]):
        return False
    return any(
        term in text
        for term in [
            "aluminium",
            "asbestos",
            "clad",
            "cladding",
            "composite",
            "concrete",
            "framed",
            "glulam",
            "insulated",
            "metal",
            "panel",
            "profiled",
            "polycarbonate",
            "reinforced",
            "seam",
            "standing",
            "timber",
            "vertically",
        ]
    )


def normalized_phrase(phrase: str) -> str:
    phrase = re.sub(r"^(?:with|incorporating|including|comprising|and)\s+", "", phrase)
    phrase = phrase.strip(" ,.")
    return phrase[0].upper() + phrase[1:] if phrase else phrase


def accepted_value_rationale(
    *,
    base_rationale: str,
    evidence_value: Any,
    accepted_value: Any,
    normalized_value: Any,
) -> str:
    if accepted_value == evidence_value:
        return base_rationale
    if accepted_value == normalized_value and not isinstance(accepted_value, str):
        return (
            f"{base_rationale} Final accepted value uses the typed normalized value; "
            "the verbatim evidence value is retained separately."
        )
    return (
        f"{base_rationale} Final accepted value is restricted to the component-specific "
        "phrase directly supported by the canonical evidence; the broader prior value "
        "is retained as the verbatim evidence value."
    )


def prior_statuses(target_id: str, artifacts: LoadedArtifacts) -> dict[str, Any]:
    return {
        "batch_v1_extraction_status": artifacts.batch_extractions[target_id].get("status"),
        "batch_v1_validation_status": artifacts.batch_validation[target_id].get("status"),
        "batch_v1_extraction_value": artifacts.batch_extractions[target_id].get(
            "evidence_value"
        ),
        "diagnostic_eligibility": artifacts.diagnostic_eligibility[target_id].get("status"),
        "diagnostic_support_classification": artifacts.diagnostic_eligibility[target_id].get(
            "support_classification"
        ),
        "extracted_result_audit": (
            artifacts.diagnostic_audit[target_id].get("audit_classification")
            if target_id in artifacts.diagnostic_audit
            else None
        ),
        "reextract_v1_status": artifacts.reextract_v1.get(target_id, {}).get("status"),
        "reextract_v2_status": artifacts.reextract_v2.get(target_id, {}).get("status"),
        "escalation_status": artifacts.escalation_decisions.get(target_id, {}).get(
            "final_status"
        ),
        "strict_text_reextract_status": artifacts.text_reextract.get(target_id, {}).get(
            "status"
        ),
    }


def latest_evidence_decision(target_id: str, artifacts: LoadedArtifacts) -> dict[str, Any]:
    if target_id in artifacts.text_reextract:
        return {
            "stage": "strict_text_reextract",
            "status": artifacts.text_reextract[target_id].get("status"),
            "reason": artifacts.text_reextract[target_id].get("ambiguity_or_caveat"),
        }
    if target_id in artifacts.escalation_decisions:
        escalation = artifacts.escalation_decisions[target_id]
        return {
            "stage": "evidence_escalation",
            "status": escalation.get("final_status"),
            "reason": escalation.get("notes"),
            "candidate_page_or_range": escalation.get("candidate_page_or_range"),
            "evidence_type": escalation.get("evidence_type"),
        }
    if target_id in artifacts.reextract_v2:
        return {
            "stage": "reextract_v2",
            "status": artifacts.reextract_v2[target_id].get("status"),
            "reason": artifacts.reextract_v2[target_id].get("ambiguity_or_caveat"),
        }
    return {
        "stage": "retrieval_diagnostic",
        "status": artifacts.diagnostic_eligibility[target_id].get("status"),
        "reason": artifacts.diagnostic_eligibility[target_id].get("reason"),
    }


def manual_review_reason(target_id: str, artifacts: LoadedArtifacts) -> str:
    escalation = artifacts.escalation_decisions.get(target_id, {})
    triage = artifacts.visual_triage.get(target_id, {})
    return " ".join(
        item
        for item in [
            str(escalation.get("recommended_next_action") or ""),
            str(escalation.get("notes") or ""),
            str(triage.get("explanation") or ""),
            "No VLM call was made in final adjudication.",
        ]
        if item
    )


def validate_target_coverage(
    artifacts: LoadedArtifacts,
    *,
    expected_target_count: int,
) -> None:
    target_ids = [target.target_row_id for target in artifacts.selected_targets]
    duplicates = sorted(target_id for target_id, count in Counter(target_ids).items() if count > 1)
    if duplicates:
        msg = f"Duplicate selected target IDs: {duplicates}"
        raise ValueError(msg)
    if len(target_ids) != expected_target_count:
        msg = f"Expected {expected_target_count} targets, found {len(target_ids)}"
        raise ValueError(msg)
    expected = set(target_ids)
    coverage_sources = {
        "batch_extractions": set(artifacts.batch_extractions),
        "batch_validation": set(artifacts.batch_validation),
        "batch_evidence_validation": set(artifacts.batch_evidence_validation),
        "batch_shape_validation": set(artifacts.batch_shape_validation),
        "batch_schema": set(artifacts.batch_schema),
        "batch_normalized": set(artifacts.batch_normalized),
        "batch_retrieval": set(artifacts.batch_retrieval),
        "diagnostic_eligibility": set(artifacts.diagnostic_eligibility),
    }
    for label, ids in coverage_sources.items():
        missing = sorted(expected - ids)
        unexpected = sorted(ids - expected)
        if missing or unexpected:
            msg = f"{label} target coverage mismatch; missing={missing}; unexpected={unexpected}"
            raise ValueError(msg)


def validate_final_records(records: list[dict[str, Any]], artifacts: LoadedArtifacts) -> None:
    ids = [str(record["target_id"]) for record in records]
    duplicates = sorted(target_id for target_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        msg = f"Duplicate final target IDs: {duplicates}"
        raise ValueError(msg)
    expected = set(artifacts.targets_by_id)
    actual = set(ids)
    if expected != actual:
        msg = (
            "Final target coverage mismatch; "
            f"missing={sorted(expected - actual)}; "
            f"unexpected={sorted(actual - expected)}"
        )
        raise ValueError(msg)
    for record in records:
        if record["final_disposition"] in ACCEPTED_DISPOSITIONS:
            ensure_complete_accepted_record(record)
    accepted_ids = {
        record["target_id"]
        for record in records
        if record["final_disposition"] in ACCEPTED_DISPOSITIONS
    }
    unsupported_ids = {record["target_id"] for record in unsupported_targets(records)}
    if accepted_ids & unsupported_ids:
        overlap = sorted(accepted_ids & unsupported_ids)
        msg = f"Targets overlap between accepted and unsupported: {overlap}"
        raise ValueError(msg)


def ensure_complete_accepted_record(record: dict[str, Any]) -> None:
    required = [
        "target_id",
        "dictionary_row",
        "domain",
        "sub_domain",
        "field_name",
        "requested_attribute_semantics",
        "final_accepted_value",
        "source_document",
        "page",
        "hierarchy_node",
        "evidence_excerpt",
        "evidence_validation_status",
        "value_shape_validation_status",
        "dictionary_compatibility_status",
        "final_review_status",
        "winning_decision_artifact",
        "decision_rationale",
    ]
    missing = [field for field in required if _is_missing_required_value(record.get(field))]
    if missing:
        msg = f"Accepted record {record.get('target_id')} lacks required fields: {missing}"
        raise ValueError(msg)
    if not record.get("canonical_evidence_span_id"):
        msg = f"Accepted record {record.get('target_id')} lacks canonical evidence span ID"
        raise ValueError(msg)


def _is_missing_required_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (dict, list)):
        return len(value) == 0
    return False


def write_final_adjudication_outputs(
    records: list[dict[str, Any]],
    artifacts: LoadedArtifacts,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted = accepted_values(records)
    manual = manual_review_queue(records)
    unsupported = unsupported_targets(records)
    ambiguous = ambiguous_targets(records)
    rejected = rejected_prior_extractions(records, artifacts)
    audit = evidence_audit(records, artifacts)
    trace = disposition_trace(records, artifacts)
    metrics = build_final_metrics(records, artifacts)
    _atomic_write_json(output_dir / "input_file_inventory.json", artifacts.file_inventory)
    _atomic_write_json(output_dir / "final_target_dispositions.json", records)
    _atomic_write_json(output_dir / "accepted_values.json", accepted)
    _atomic_write_json(output_dir / "manual_review_queue.json", manual)
    _atomic_write_json(output_dir / "unsupported_targets.json", unsupported)
    _atomic_write_json(output_dir / "ambiguous_targets.json", ambiguous)
    _atomic_write_json(output_dir / "rejected_prior_extractions.json", rejected)
    _atomic_write_json(output_dir / "evidence_audit.json", audit)
    _atomic_write_json(output_dir / "disposition_trace.json", trace)
    _atomic_write_json(output_dir / "final_metrics.json", metrics)
    write_dict_csv(output_dir / "accepted_values.csv", accepted, accepted_csv_fields())
    write_dict_csv(output_dir / "manual_review_queue.csv", manual, manual_review_csv_fields())
    write_dict_csv(
        output_dir / "final_adjudication_review.csv",
        final_review_rows(records),
        final_review_csv_fields(),
    )
    (output_dir / "final_adjudication_summary.md").write_text(
        final_summary_markdown(metrics), encoding="utf-8"
    )
    (output_dir / "lessons_for_next_batch.md").write_text(
        lessons_markdown(records, metrics), encoding="utf-8"
    )


def accepted_values(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "target_id": record["target_id"],
            "dictionary_row": record["dictionary_row"],
            "domain": record["domain"],
            "sub_domain": record["sub_domain"],
            "field_name": record["field_name"],
            "definition": record["definition"],
            "datatype": record["datatype"],
            "unit": record["unit"],
            "value_shape": record["value_shape"],
            "requested_attribute_semantics": record["requested_attribute_semantics"],
            "final_disposition": record["final_disposition"],
            "final_evidence_value": record["final_evidence_value"],
            "final_accepted_value": record["final_accepted_value"],
            "normalized_value": record["normalized_value"],
            "display_value": record["display_value"],
            "canonical_evidence_span_id": record["canonical_evidence_span_id"],
            "source_document": record["source_document"],
            "page": record["page"],
            "hierarchy_node": record["hierarchy_node"],
            "evidence_excerpt": record["evidence_excerpt"],
            "evidence_validation_status": record["evidence_validation_status"],
            "value_shape_validation_status": record["value_shape_validation_status"],
            "dictionary_compatibility_status": record["dictionary_compatibility_status"],
            "final_review_status": record["final_review_status"],
            "decision_source_artifact": record["decision_source_artifact"],
            "decision_rationale": record["decision_rationale"],
        }
        for record in records
        if record["final_disposition"] in ACCEPTED_DISPOSITIONS
    ]


def manual_review_queue(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "target_id": record["target_id"],
            "dictionary_row": record["dictionary_row"],
            "domain": record["domain"],
            "sub_domain": record["sub_domain"],
            "field_name": record["field_name"],
            "definition": record["definition"],
            "value_shape": record["value_shape"],
            "source_file": record["latest_evidence_decision"].get("source_file", ""),
            "candidate_page_or_pages": record["latest_evidence_decision"].get(
                "candidate_page_or_range", ""
            )
            or record["retrieval_eligibility"].get("page_number", ""),
            "expected_visual_signal": record["manual_review_reason"],
            "reason_text_extraction_is_inadequate": record["manual_review_reason"],
            "no_vlm_call_made_note": "No VLM call was made in final adjudication.",
            "decision_source_artifact": record["decision_source_artifact"],
        }
        for record in records
        if record["final_disposition"] == "manual_visual_review"
    ]


def unsupported_targets(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    statuses = {
        "unsupported_in_supplied_sources",
        "component_present_attribute_absent",
        "insufficient_evidence",
        "invalid_prior_extraction",
    }
    return [record for record in records if record["final_disposition"] in statuses]


def ambiguous_targets(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["final_disposition"] in {"dictionary_target_ambiguous", "conflicting_evidence"}
    ]


def rejected_prior_extractions(
    records: list[dict[str, Any]],
    artifacts: LoadedArtifacts,
) -> list[dict[str, Any]]:
    final_by_id = {record["target_id"]: record for record in records}
    rejected: list[dict[str, Any]] = []
    for target_id, extraction in artifacts.batch_extractions.items():
        if extraction.get("status") != NON_NULL_EXTRACTED_STATUS:
            continue
        final = final_by_id[target_id]
        if final["final_disposition"] in ACCEPTED_DISPOSITIONS:
            continue
        audit = artifacts.diagnostic_audit.get(target_id, {})
        rejected.append(
            {
                "target_id": target_id,
                "field_name": final["field_name"],
                "earlier_value": extraction.get("evidence_value"),
                "earlier_status": extraction.get("status"),
                "earlier_evidence_reference": ";".join(
                    str(item) for item in extraction.get("supporting_span_ids", [])
                ),
                "later_audit_finding": audit.get("audit_classification")
                or final["latest_evidence_decision"].get("status"),
                "final_disposition": final["final_disposition"],
                "rejection_reason": final["decision_rationale"],
                "decision_artifact_that_invalidated_it": final["decision_source_artifact"],
            }
        )
    return sort_records(rejected)


def evidence_audit(
    records: list[dict[str, Any]],
    artifacts: LoadedArtifacts,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    approved_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in artifacts.approved_evidence:
        approved_by_target[str(item["target_row_id"])].append(item)
    for record in records:
        target_id = record["target_id"]
        eligibility = artifacts.diagnostic_eligibility[target_id]
        audit = artifacts.diagnostic_audit.get(target_id, {})
        text_evidence = approved_by_target.get(target_id, [])
        rows.append(
            {
                "target_id": target_id,
                "field_name": record["field_name"],
                "component_support": component_support(eligibility, audit),
                "requested_attribute_support": requested_attribute_support(
                    record, eligibility, audit
                ),
                "value_support": bool(record["final_accepted_value"]),
                "unit_support": (
                    bool(record.get("unit")) if record["final_accepted_value"] else False
                ),
                "source_locality": source_locality(record, eligibility, text_evidence),
                "canonical_span_availability": bool(record["canonical_evidence_span_id"]),
                "evidence_conflict": record["final_disposition"] == "conflicting_evidence",
                "dictionary_compatibility": record["dictionary_compatibility_status"],
                "value_shape_compatibility": record["value_shape_validation_status"],
                "final_evidence_sufficiency": record["final_disposition"]
                in ACCEPTED_DISPOSITIONS,
                "support_classification": eligibility.get("support_classification"),
                "audit_classification": audit.get("audit_classification"),
                "evidence_issues": evidence_issues(target_id, artifacts),
            }
        )
    return rows


def disposition_trace(
    records: list[dict[str, Any]],
    artifacts: LoadedArtifacts,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        target_id = record["target_id"]
        stages: list[dict[str, Any]] = [
            trace_entry(
                "Batch V1",
                "output/enfield_unit1_evidence_first_batch_v1/extraction_results.json",
                artifacts.batch_extractions[target_id],
            ),
            trace_entry(
                "Batch V1 validation",
                "output/enfield_unit1_evidence_first_batch_v1/validation_results.json",
                artifacts.batch_validation[target_id],
            ),
            trace_entry(
                "retrieval diagnostic",
                "output/enfield_unit1_evidence_first_batch_v1_retrieval_diagnostic/retrieval_eligibility.json",
                artifacts.diagnostic_eligibility[target_id],
            ),
        ]
        optional = [
            (
                "extracted-result audit",
                "output/enfield_unit1_evidence_first_batch_v1_retrieval_diagnostic/extracted_result_attribute_audit.json",
                artifacts.diagnostic_audit,
            ),
            (
                "re-extraction V1",
                "output/enfield_unit1_evidence_first_batch_v1_reextract_v1/extraction_results.json",
                artifacts.reextract_v1,
            ),
            (
                "re-extraction V2",
                "output/enfield_unit1_evidence_first_batch_v1_reextract_v2/extraction_results.json",
                artifacts.reextract_v2,
            ),
            (
                "evidence escalation",
                "output/enfield_unit1_evidence_escalation_v1/escalation_decisions.json",
                artifacts.escalation_decisions,
            ),
            (
                "strict text re-extraction",
                "output/enfield_unit1_evidence_escalation_text_reextract_v1/extraction_results.json",
                artifacts.text_reextract,
            ),
        ]
        for stage, artifact, mapping in optional:
            if target_id in mapping:
                stages.append(trace_entry(stage, artifact, mapping[target_id]))
        stages.append(
            {
                "stage": "final adjudication",
                "artifact": "final_target_dispositions.json",
                "status": record["final_disposition"],
                "reason": record["decision_rationale"],
            }
        )
        rows.append({"target_id": target_id, "field_name": record["field_name"], "trace": stages})
    return rows


def trace_entry(stage: str, artifact: str, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": stage,
        "artifact": artifact,
        "status": raw.get("status") or raw.get("final_status") or raw.get("audit_classification"),
        "value": raw.get("evidence_value"),
        "reason": raw.get("reason") or raw.get("explanation") or raw.get("ambiguity_or_caveat"),
    }


def build_final_metrics(
    records: list[dict[str, Any]],
    artifacts: LoadedArtifacts,
) -> dict[str, Any]:
    disposition_counts = Counter(str(record["final_disposition"]) for record in records)
    accepted = sum(disposition_counts[item] for item in ACCEPTED_DISPOSITIONS)
    original_non_null = [
        item
        for item in artifacts.batch_extractions.values()
        if item.get("status") == NON_NULL_EXTRACTED_STATUS
    ]
    rejected = rejected_prior_extractions(records, artifacts)
    retained_original_ids = {
        record["target_id"]
        for record in records
        if record["final_disposition"] in ACCEPTED_DISPOSITIONS
        and artifacts.batch_extractions[record["target_id"]].get("status")
        == NON_NULL_EXTRACTED_STATUS
    }
    return {
        "total_targets": len(records),
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "accepted_count": disposition_counts["accepted"],
        "accepted_after_normalization_count": disposition_counts[
            "accepted_after_normalization"
        ],
        "accepted_with_dictionary_caveat_count": disposition_counts[
            "accepted_with_dictionary_caveat"
        ],
        "total_accepted_count": accepted,
        "manual_visual_review_count": disposition_counts["manual_visual_review"],
        "unsupported_in_supplied_sources_count": disposition_counts[
            "unsupported_in_supplied_sources"
        ],
        "dictionary_target_ambiguous_count": disposition_counts[
            "dictionary_target_ambiguous"
        ],
        "component_present_attribute_absent_count": disposition_counts[
            "component_present_attribute_absent"
        ],
        "insufficient_evidence_count": disposition_counts["insufficient_evidence"],
        "conflicting_evidence_count": disposition_counts["conflicting_evidence"],
        "invalid_prior_extraction_final_count": disposition_counts["invalid_prior_extraction"],
        "rejected_prior_extraction_count": len(rejected),
        "acceptance_by_domain": acceptance_distribution(records, "domain"),
        "acceptance_by_sub_domain": acceptance_distribution(records, "sub_domain"),
        "acceptance_by_value_shape_family": acceptance_distribution(records, "value_shape"),
        "final_dispositions_by_value_shape_family": nested_distribution(
            records, "value_shape", "final_disposition"
        ),
        "evidence_failure_distribution": evidence_failure_distribution(records),
        "dictionary_metadata_issue_distribution": dictionary_issue_distribution(records),
        "prior_batch_v1_status_versus_final_disposition": prior_vs_final(records),
        "original_non_null_extractions": len(original_non_null),
        "original_non_null_extractions_retained": len(retained_original_ids),
        "original_non_null_extractions_rejected": len(rejected),
        "original_non_null_extractions_retained_percentage": round(
            (len(retained_original_ids) / len(original_non_null) * 100)
            if original_non_null
            else 0,
            2,
        ),
        "targets_with_complete_canonical_evidence_provenance": sum(
            1
            for record in records
            if record["final_disposition"] in ACCEPTED_DISPOSITIONS
            and bool(record["canonical_evidence_span_id"])
        ),
        "targets_lacking_canonical_requested_attribute_support": sum(
            1 for record in records if record["final_disposition"] not in ACCEPTED_DISPOSITIONS
        ),
        "input_file_inventory": artifacts.file_inventory,
    }


def acceptance_distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(record.get(key) or "")
                for record in records
                if record["final_disposition"] in ACCEPTED_DISPOSITIONS
            ).items()
        )
    )


def nested_distribution(
    records: list[dict[str, Any]],
    group_key: str,
    count_key: str,
) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        result[str(record.get(group_key) or "")][str(record[count_key])] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(result.items())}


def evidence_failure_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    failures = Counter(
        str(record["final_disposition"])
        for record in records
        if record["final_disposition"] not in ACCEPTED_DISPOSITIONS
        and record["final_disposition"] != "manual_visual_review"
    )
    return dict(sorted(failures.items()))


def dictionary_issue_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        if record["final_disposition"] == "dictionary_target_ambiguous":
            counter["dictionary_target_ambiguous"] += 1
        if record["dictionary_compatibility_status"] not in {
            "compatible",
            "compatible_after_normalization",
            "not_evaluated",
            "unit_not_applicable",
        }:
            counter[str(record["dictionary_compatibility_status"])] += 1
    return dict(sorted(counter.items()))


def prior_vs_final(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        prior = str(record["prior_statuses"].get("batch_v1_extraction_status"))
        result[prior][str(record["final_disposition"])] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(result.items())}


def final_review_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "target ID": record["target_id"],
                "dictionary row": record["dictionary_row"],
                "domain": record["domain"],
                "sub-domain": record["sub_domain"],
                "field name": record["field_name"],
                "definition": record["definition"],
                "datatype": record["datatype"],
                "unit": record["unit"],
                "value shape": record["value_shape"],
                "Batch V1 extraction value": record["prior_statuses"].get(
                    "batch_v1_extraction_value"
                ),
                "Batch V1 status": record["prior_statuses"].get(
                    "batch_v1_extraction_status"
                ),
                "diagnostic attribute-support audit": (
                    record["diagnostic_attribute_support_audit"].get("audit_classification")
                    if record.get("diagnostic_attribute_support_audit")
                    else record["retrieval_eligibility"].get("support_classification")
                ),
                "latest evidence decision": record["latest_evidence_decision"].get("status"),
                "final evidence value": record["final_evidence_value"],
                "final accepted value": record["final_accepted_value"],
                "normalized value": record["normalized_value"],
                "final disposition": record["final_disposition"],
                "source file": record["source_document"],
                "page": record["page"],
                "hierarchy node": record["hierarchy_node"],
                "canonical evidence excerpt": record["evidence_excerpt"],
                "dictionary compatibility": record["dictionary_compatibility_status"],
                "manual-review reason": record["manual_review_reason"],
                "unsupported reason": record["unsupported_reason"],
                "decision artifact/source": record["decision_source_artifact"],
                "decision rationale": record["decision_rationale"],
            }
        )
    return rows


def final_summary_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# Evidence-First Batch V1 Final Adjudication",
        "",
        f"- Total targets adjudicated: {metrics['total_targets']}",
        f"- Original Batch V1 non-null extractions: {metrics['original_non_null_extractions']}",
        f"- Retained accepted values: {metrics['total_accepted_count']}",
        f"- Rejected prior non-null extractions: {metrics['rejected_prior_extraction_count']}",
        f"- Manual visual review: {metrics['manual_visual_review_count']}",
        f"- Unsupported in supplied sources: {metrics['unsupported_in_supplied_sources_count']}",
        "- Component present but attribute absent: "
        f"{metrics['component_present_attribute_absent_count']}",
        f"- Insufficient evidence: {metrics['insufficient_evidence_count']}",
        f"- Dictionary target ambiguous: {metrics['dictionary_target_ambiguous_count']}",
        f"- Conflicting evidence: {metrics['conflicting_evidence_count']}",
        "",
        (
            "The initial Batch V1 `extracted` label is not authoritative. Final "
            "acceptance is limited to records with later requested-attribute support "
            "and complete canonical evidence provenance."
        ),
        "",
        "## Dispositions",
        "",
    ]
    for key, value in metrics["disposition_counts"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def lessons_markdown(records: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    accepted = [
        record for record in records if record["final_disposition"] in ACCEPTED_DISPOSITIONS
    ]
    manual = [record for record in records if record["final_disposition"] == "manual_visual_review"]
    unsupported = [
        record
        for record in records
        if record["final_disposition"]
        in {"unsupported_in_supplied_sources", "component_present_attribute_absent"}
    ]
    ambiguous = [
        record for record in records if record["final_disposition"] == "dictionary_target_ambiguous"
    ]
    lines = [
        "# Lessons For Next Batch",
        "",
        "## Target Families Worth Selecting",
        "",
    ]
    lines.extend(f"- {family}" for family in summarized_families(accepted))
    lines.extend(
        [
            "",
            "## Target Families To Exclude Without Additional Sources",
            "",
        ]
    )
    lines.extend(f"- {family}" for family in summarized_families(unsupported))
    lines.extend(
        [
            "",
            "## Route To Drawings Or Manual Visual Review",
            "",
        ]
    )
    lines.extend(f"- {family}" for family in summarized_families(manual))
    lines.extend(
        [
            "",
            "## Dictionary Fields Needing Customer Clarification",
            "",
        ]
    )
    lines.extend(f"- {family}" for family in summarized_families(ambiguous))
    lines.extend(
        [
            "",
            "## Metrics Basis",
            "",
            f"- Final accepted values: {metrics['total_accepted_count']}",
            f"- Manual visual-review targets: {metrics['manual_visual_review_count']}",
            f"- Dictionary ambiguous targets: {metrics['dictionary_target_ambiguous_count']}",
        ]
    )
    return "\n".join(lines) + "\n"


def summarized_families(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return ["No targets in this final adjudication subset."]
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        key = f"{record['sub_domain']} / {record['value_shape']}"
        grouped[key].append(str(record["field_name"]))
    return [
        f"{key}: {', '.join(sorted(values)[:6])}"
        for key, values in sorted(grouped.items())
    ]


def write_dict_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def accepted_csv_fields() -> list[str]:
    return [
        "target_id",
        "dictionary_row",
        "domain",
        "sub_domain",
        "field_name",
        "definition",
        "datatype",
        "unit",
        "value_shape",
        "final_disposition",
        "final_evidence_value",
        "final_accepted_value",
        "normalized_value",
        "display_value",
        "canonical_evidence_span_id",
        "source_document",
        "page",
        "hierarchy_node",
        "evidence_excerpt",
        "evidence_validation_status",
        "value_shape_validation_status",
        "dictionary_compatibility_status",
        "final_review_status",
        "decision_source_artifact",
        "decision_rationale",
    ]


def manual_review_csv_fields() -> list[str]:
    return [
        "target_id",
        "dictionary_row",
        "domain",
        "sub_domain",
        "field_name",
        "definition",
        "value_shape",
        "source_file",
        "candidate_page_or_pages",
        "expected_visual_signal",
        "reason_text_extraction_is_inadequate",
        "no_vlm_call_made_note",
        "decision_source_artifact",
    ]


def final_review_csv_fields() -> list[str]:
    return [
        "target ID",
        "dictionary row",
        "domain",
        "sub-domain",
        "field name",
        "definition",
        "datatype",
        "unit",
        "value shape",
        "Batch V1 extraction value",
        "Batch V1 status",
        "diagnostic attribute-support audit",
        "latest evidence decision",
        "final evidence value",
        "final accepted value",
        "normalized value",
        "final disposition",
        "source file",
        "page",
        "hierarchy node",
        "canonical evidence excerpt",
        "dictionary compatibility",
        "manual-review reason",
        "unsupported reason",
        "decision artifact/source",
        "decision rationale",
    ]


def component_support(
    eligibility: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    support = audit.get("support_classification") or eligibility.get("support_classification")
    if support in {"supports_requested_attribute", "component_only"}:
        return "present"
    if support == "attribute_only":
        return "not_supported"
    return str(support or "unknown")


def requested_attribute_support(
    record: dict[str, Any],
    eligibility: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    if record["final_disposition"] in ACCEPTED_DISPOSITIONS:
        return "supported"
    support = audit.get("support_classification") or eligibility.get("support_classification")
    if support == "component_only":
        return "absent"
    return str(support or "not_supported")


def source_locality(
    record: dict[str, Any],
    eligibility: dict[str, Any],
    text_evidence: list[dict[str, Any]],
) -> str:
    if record["source_document"] and record["page"]:
        return f"{record['source_document']}:{record['page']}"
    if eligibility.get("source_id") and eligibility.get("page_number"):
        return f"{eligibility['source_id']}:{eligibility['page_number']}"
    if text_evidence:
        first = text_evidence[0]
        return f"{first.get('source_id')}:{first.get('page_number')}"
    return "not_available"


def evidence_issues(target_id: str, artifacts: LoadedArtifacts) -> list[str]:
    issues: list[str] = []
    for mapping in [
        artifacts.batch_evidence_validation,
        artifacts.batch_shape_validation,
        artifacts.batch_schema,
    ]:
        raw = mapping.get(target_id, {})
        issues.extend(str(item) for item in raw.get("issues", []))
    issues.extend(
        str(item)
        for item in artifacts.text_reextract.get(target_id, {}).get("issues", [])
    )
    return issues


def sorted_targets(targets: list[TargetSpecification]) -> list[TargetSpecification]:
    return sorted(
        targets,
        key=lambda target: (
            target.source_dictionary_provenance.row_number
            if target.source_dictionary_provenance
            else 999999,
            target.target_row_id,
        ),
    )


def sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (record.get("dictionary_row") or 999999, record["target_id"]),
    )


def inventory_input_files(inputs: AdjudicationInputs) -> dict[str, list[str]]:
    return {
        label: [
            str(path.relative_to(root))
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and ".w" not in path.parts
            and "worker_runs" not in path.parts
            and "page_probe_runs" not in path.parts
        ]
        for label, root in [
            ("batch_v1", inputs.batch_v1_dir),
            ("retrieval_diagnostic", inputs.diagnostic_dir),
            ("reextract_v1", inputs.reextract_v1_dir),
            ("reextract_v2", inputs.reextract_v2_dir),
            ("evidence_escalation", inputs.escalation_dir),
            ("strict_text_reextract", inputs.escalation_text_dir),
        ]
    }


def _by_id(
    rows: list[dict[str, Any]],
    key: str,
    label: str,
    *,
    allow_missing_for_targets: bool = False,
) -> dict[str, dict[str, Any]]:
    _ = allow_missing_for_targets
    result: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        row_id = str(row[key])
        if row_id in result:
            duplicates.append(row_id)
        result[row_id] = row
    if duplicates:
        msg = f"Duplicate IDs in {label}: {sorted(set(duplicates))}"
        raise ValueError(msg)
    return result


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_list(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, list):
        msg = f"Expected list JSON in {path}"
        raise ValueError(msg)
    return [dict(item) for item in data]
