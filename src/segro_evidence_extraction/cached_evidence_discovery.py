"""Planning-only discovery over already cached evidence pages."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, field_validator

from segro_evidence_extraction.cluster_manifest import EvidenceClusterManifest
from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.source import SourceRegistryEntry
from segro_evidence_extraction.unit_runner_v1 import validate_json_schema_subset

DISCOVERY_REQUEST_SCHEMA_VERSION = "segro_cached_evidence_family_discovery_request_v1"
DISCOVERY_REQUEST_CONTRACT_VERSION = "1.0.0"
DEFAULT_DISCOVERY_REQUEST_SCHEMA_PATH = Path(
    "schemas/cached_evidence_discovery/"
    "segro_cached_evidence_family_discovery_request_v1.schema.json"
)

VALUE_SHAPE_PATTERNS = {
    "date": [r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", r"\b\d{4}-\d{2}-\d{2}\b"],
    "integer_count": [r"\b\d+\b"],
    "decimal_measurement": [r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", r"\b\d+(?:\.\d+)?\b"],
    "identifier_or_reference": [r"\b[A-Z]{1,8}[-/]?[A-Z0-9]{2,}(?:[-/][A-Z0-9]{2,})*\b"],
    "categorical": [r"\b[A-Z][A-Za-z0-9&./-]{2,}\b"],
    "short_text": [r"\b[A-Z][A-Za-z0-9&./-]{2,}\b"],
    "descriptive_text": [r"\b[A-Z][A-Za-z0-9&./-]{2,}\b"],
}

INDICATOR_TERMS = {
    "schedule": ["schedule", "accommodation", "table", "area", "total"],
    "certificate": ["certificate", "certified", "inspection", "approval"],
    "form": ["form", "reference", "date", "signed"],
}


class CachedEvidenceDiscoveryError(ValueError):
    """Raised when cached evidence discovery inputs are invalid."""


class DiscoveryDescriptor(StrictBaseModel):
    descriptor_id: str
    evidence_family: str
    target_ids: list[str] = Field(default_factory=list)
    field_names: list[str] = Field(default_factory=list)
    requested_attribute_phrases: list[str] = Field(default_factory=list)
    expected_value_shapes: list[str] = Field(default_factory=list)
    unit_patterns: list[str] = Field(default_factory=list)
    positive_labels: list[str] = Field(default_factory=list)
    negative_labels: list[str] = Field(default_factory=list)
    indicators: list[str] = Field(default_factory=list)
    max_candidates: int = Field(default=10, ge=1)
    minimum_score: int = Field(default=1, ge=0)

    @field_validator("descriptor_id", "evidence_family")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            msg = "value must not be blank"
            raise ValueError(msg)
        return value


class DiscoveryRequest(StrictBaseModel):
    schema_version: str = DISCOVERY_REQUEST_SCHEMA_VERSION
    contract_version: str = DISCOVERY_REQUEST_CONTRACT_VERSION
    discovery_run_name: str
    source_inventory_path: str
    cache_root: str
    output_path: str
    descriptors: list[DiscoveryDescriptor] = Field(min_length=1)
    provenance: list[str] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def schema_version_must_match(cls, value: str) -> str:
        if value != DISCOVERY_REQUEST_SCHEMA_VERSION:
            msg = f"unsupported discovery request schema_version: {value}"
            raise ValueError(msg)
        return value

    @field_validator("contract_version")
    @classmethod
    def contract_version_must_be_v1(cls, value: str) -> str:
        if not value.startswith("1."):
            msg = f"unsupported discovery request contract_version: {value}"
            raise ValueError(msg)
        return value


def load_discovery_request(
    request_path: Path,
    *,
    schema_path: Path = DEFAULT_DISCOVERY_REQUEST_SCHEMA_PATH,
) -> DiscoveryRequest:
    data = _read_json_object(request_path)
    schema = _read_json_object(schema_path)
    schema_errors = validate_json_schema_subset(data, schema)
    if schema_errors:
        msg = "discovery request JSON Schema validation failed: " + "; ".join(schema_errors)
        raise CachedEvidenceDiscoveryError(msg)
    try:
        return DiscoveryRequest.model_validate(data)
    except ValidationError as exc:
        msg = f"discovery request semantic validation failed: {exc}"
        raise CachedEvidenceDiscoveryError(msg) from exc


def run_cached_evidence_discovery(
    request: DiscoveryRequest,
    *,
    sources: list[SourceRegistryEntry],
) -> dict[str, Any]:
    source_by_id = {source.source_id: source for source in sources}
    page_inventory = load_cached_page_inventory(Path(request.cache_root), source_by_id)
    ranked: list[dict[str, Any]] = []
    for descriptor in request.descriptors:
        candidates = [
            score_page_for_descriptor(page, descriptor)
            for page in page_inventory
        ]
        candidates = [
            candidate
            for candidate in candidates
            if candidate["candidate_score"] >= descriptor.minimum_score
        ]
        candidates.sort(
            key=lambda item: (
                -int(item["candidate_score"]),
                str(item["source_document_name"]),
                int(item["page_number"]),
            )
        )
        ranked.extend(candidates[: descriptor.max_candidates])
    return {
        "schema_version": "segro_cached_evidence_family_discovery_result_v1",
        "discovery_run_name": request.discovery_run_name,
        "source_count": len(sources),
        "cached_page_inventory_summary": summarize_cached_pages(page_inventory),
        "ranked_candidate_pages": ranked,
        "readiness_decisions_made": False,
        "extraction_decisions_made": False,
    }


def reconcile_discovery_with_manifest(
    *,
    discovery_result: dict[str, Any],
    manifest: EvidenceClusterManifest,
    exclusion_justifications: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest_pages = {
        (assignment.source_id, page_number)
        for target in manifest.targets
        for assignment in target.source_assignments
        for page_range in assignment.candidate_page_ranges
        for page_number in range(page_range.page_start, page_range.page_end + 1)
    }
    justifications = {
        (str(row["descriptor_id"]), str(row["source_id"]), int(row["page_number"])): row
        for row in exclusion_justifications or []
    }
    rows = []
    omitted = []
    for candidate in discovery_result["ranked_candidate_pages"]:
        key = (str(candidate["source_id"]), int(candidate["page_number"]))
        justification_key = (
            str(candidate["descriptor_id"]),
            str(candidate["source_id"]),
            int(candidate["page_number"]),
        )
        present = key in manifest_pages
        row = {
            **candidate,
            "already_present_in_manifest": present,
            "planner_exclusion_justification": justifications.get(justification_key),
        }
        rows.append(row)
        if not present and candidate["candidate_score"] > 0:
            omitted.append(row)
    return {
        "schema_version": "segro_cached_evidence_manifest_reconciliation_v1",
        "manifest_run_name": manifest.run_name,
        "candidate_count": len(rows),
        "omitted_higher_signal_count": len(omitted),
        "reconciled_candidates": rows,
        "omitted_higher_signal_pages": omitted,
        "manifest_was_altered": False,
    }


def load_cached_page_inventory(
    cache_root: Path,
    source_by_id: dict[str, SourceRegistryEntry],
) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(cache_root.rglob("page_*.json")):
        data = _read_json_object(path)
        source_id = str(data.get("source_id") or path.parent.parent.name)
        source = source_by_id.get(source_id)
        if source is None:
            continue
        text = str(data.get("extracted_text") or "")
        rows.append(
            {
                "source_id": source_id,
                "source_document_name": Path(source.logical_path).name,
                "logical_path": source.logical_path,
                "page_number": int(data.get("page_number") or 0),
                "cache_path": str(path),
                "text": text,
                "text_empty": not text.strip(),
                "text_character_count": len(text),
                "page_quality": page_quality(text),
            }
        )
    return rows


def score_page_for_descriptor(
    page: dict[str, Any],
    descriptor: DiscoveryDescriptor,
) -> dict[str, Any]:
    text = str(page["text"])
    lower = text.casefold()
    positive = matched_terms(lower, descriptor.positive_labels)
    requested = matched_terms(lower, descriptor.requested_attribute_phrases)
    negative = matched_terms(lower, descriptor.negative_labels)
    indicator_matches = matched_indicator_terms(lower, descriptor.indicators)
    value_matches = matched_regexes(text, patterns_for_shapes(descriptor.expected_value_shapes))
    unit_matches = matched_regexes(text, descriptor.unit_patterns)
    score = (
        len(positive) * 6
        + len(requested) * 5
        + len(indicator_matches) * 4
        + len(value_matches) * 3
        + len(unit_matches) * 5
        - len(negative) * 8
    )
    reasons = []
    exclusions = []
    if positive:
        reasons.append("positive_label_match")
    if requested:
        reasons.append("requested_attribute_phrase_match")
    if indicator_matches:
        reasons.append("indicator_match")
    if value_matches:
        reasons.append("value_shape_match")
    if unit_matches:
        reasons.append("unit_pattern_match")
    if negative:
        exclusions.append("negative_label_match")
    if page["text_empty"]:
        exclusions.append("text_empty_page_quality_warning")
        score = min(score, 0)
    if (positive or indicator_matches) and not value_matches and not unit_matches:
        exclusions.append("heading_or_label_only_no_value_unit")
    return {
        "descriptor_id": descriptor.descriptor_id,
        "evidence_family": descriptor.evidence_family,
        "target_ids": list(descriptor.target_ids),
        "field_names": list(descriptor.field_names),
        "source_id": page["source_id"],
        "source_document_name": page["source_document_name"],
        "logical_path": page["logical_path"],
        "page_number": page["page_number"],
        "cache_path": page["cache_path"],
        "matched_positive_labels": positive,
        "matched_requested_phrases": requested,
        "matched_indicators": indicator_matches,
        "matched_value_patterns": value_matches,
        "matched_unit_patterns": unit_matches,
        "matched_negative_labels": negative,
        "page_quality": page["page_quality"],
        "text_empty": page["text_empty"],
        "text_character_count": page["text_character_count"],
        "candidate_score": max(score, 0),
        "ranking_reasons": reasons,
        "exclusion_reasons": exclusions,
        "readiness_decision": None,
    }


def page_quality(text: str) -> str:
    if not text.strip():
        return "text_empty"
    if len(text.strip()) < 80:
        return "low_text"
    return "text_available"


def summarize_cached_pages(pages: list[dict[str, Any]]) -> dict[str, Any]:
    quality = Counter(str(page["page_quality"]) for page in pages)
    return {
        "cached_page_count": len(pages),
        "source_count": len({page["source_id"] for page in pages}),
        "text_empty_page_count": quality["text_empty"],
        "page_quality_counts": dict(sorted(quality.items())),
    }


def matched_terms(lower_text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term.casefold() in lower_text]


def matched_indicator_terms(lower_text: str, indicators: list[str]) -> list[str]:
    terms = []
    for indicator in indicators:
        for term in INDICATOR_TERMS.get(indicator, [indicator]):
            if term.casefold() in lower_text:
                terms.append(term)
    return sorted(set(terms))


def patterns_for_shapes(value_shapes: list[str]) -> list[str]:
    patterns = []
    for shape in value_shapes:
        patterns.extend(VALUE_SHAPE_PATTERNS.get(shape, []))
    return patterns


def matched_regexes(text: str, patterns: list[str]) -> list[str]:
    matches = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(pattern)
    return matches


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CachedEvidenceDiscoveryError(f"Expected JSON object: {path}")
    return payload
