"""Generic evidence-cluster manifest contract and planning helpers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.source import SourceRegistryEntry
from segro_evidence_extraction.unit_runner_v1 import validate_json_schema_subset

CLUSTER_MANIFEST_SCHEMA_VERSION = "segro_evidence_cluster_manifest_v1"
CLUSTER_MANIFEST_CONTRACT_VERSION = "1.0.0"
DEFAULT_CLUSTER_MANIFEST_SCHEMA_PATH = Path(
    "schemas/evidence_cluster/segro_evidence_cluster_manifest_v1.schema.json"
)

EvidenceAcquisitionMethod = Literal["cached_text", "native_parser", "ocr", "vlm", "human_review"]
CacheExpectation = Literal["cached", "uncached", "either"]
ReadinessDisposition = Literal[
    "evidence_ready",
    "evidence_present_but_ambiguous",
    "component_only",
    "wrong_unit",
    "wrong_area_concept",
    "wrong_event",
    "dictionary_blocked",
    "semantic_clarification_required",
    "evidence_not_found",
    "requires_targeted_parsing",
    "requires_vlm",
    "human_review_required",
    "permanent_exclusion_current_poc",
]
AcquisitionDisposition = Literal[
    "cached_text_available",
    "uncached_parsing_candidate",
    "already_cached_but_text_empty",
    "method_not_allowed",
    "source_not_registered",
    "invalid_page_range",
    "page_cap_exceeded",
]


class ClusterManifestError(ValueError):
    """Raised when an evidence-cluster manifest is structurally or semantically invalid."""


class PageRange(StrictBaseModel):
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    rationale: str
    expected_cache_state: CacheExpectation = "either"

    @model_validator(mode="after")
    def page_end_must_not_precede_start(self) -> PageRange:
        if self.page_end < self.page_start:
            msg = (
                f"invalid page range {self.page_start}-{self.page_end}: "
                "page_end precedes page_start"
            )
            raise ValueError(msg)
        return self


class ClusterSourceAssignment(StrictBaseModel):
    source_id: str
    source_document_role: str
    logical_source_path: str | None = None
    section_label: str
    candidate_page_ranges: list[PageRange] = Field(min_length=1)
    allowed_methods: list[EvidenceAcquisitionMethod] = Field(min_length=1)
    max_pages_for_assignment: int = Field(ge=1)

    @field_validator("source_id", "source_document_role", "section_label")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            msg = "value must not be blank"
            raise ValueError(msg)
        return value


class ClusterTarget(StrictBaseModel):
    target_id: str
    field_name: str
    domain: str
    value_shape: str
    requested_attribute: str
    expected_evidence_families: list[str] = Field(min_length=1)
    source_assignments: list[ClusterSourceAssignment] = Field(min_length=1)
    max_pages_per_target: int = Field(ge=1)

    @field_validator("target_id", "field_name", "domain", "value_shape", "requested_attribute")
    @classmethod
    def target_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            msg = "value must not be blank"
            raise ValueError(msg)
        return value


class UpstreamProvenance(StrictBaseModel):
    artifact_path: str
    artifact_role: str
    notes: str | None = None


class EvidenceClusterManifest(StrictBaseModel):
    schema_version: str = CLUSTER_MANIFEST_SCHEMA_VERSION
    contract_version: str = CLUSTER_MANIFEST_CONTRACT_VERSION
    run_name: str
    unit_id: str
    project_number: str | None = None
    output_dir: str
    total_page_cap: int = Field(ge=1)
    allowed_readiness_dispositions: list[ReadinessDisposition] = Field(min_length=1)
    targets: list[ClusterTarget] = Field(min_length=1)
    upstream_provenance: list[UpstreamProvenance] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def schema_version_must_match(cls, value: str) -> str:
        if value != CLUSTER_MANIFEST_SCHEMA_VERSION:
            msg = f"unsupported cluster manifest schema_version: {value}"
            raise ValueError(msg)
        return value

    @field_validator("contract_version")
    @classmethod
    def contract_version_must_be_v1(cls, value: str) -> str:
        if not value.startswith("1."):
            msg = f"unsupported cluster manifest contract_version: {value}"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def target_ids_must_be_unique(self) -> EvidenceClusterManifest:
        ids = [target.target_id for target in self.targets]
        duplicates = sorted({target_id for target_id in ids if ids.count(target_id) > 1})
        if duplicates:
            msg = f"duplicate target IDs in cluster manifest: {duplicates}"
            raise ValueError(msg)
        return self


def load_cluster_manifest(
    manifest_path: Path,
    *,
    schema_path: Path = DEFAULT_CLUSTER_MANIFEST_SCHEMA_PATH,
    registered_sources: list[SourceRegistryEntry] | None = None,
) -> EvidenceClusterManifest:
    data = _read_json_object(manifest_path)
    schema = _read_json_object(schema_path)
    schema_errors = validate_json_schema_subset(data, schema)
    if schema_errors:
        msg = "cluster manifest JSON Schema validation failed: " + "; ".join(schema_errors)
        raise ClusterManifestError(msg)
    try:
        manifest = EvidenceClusterManifest.model_validate(data)
    except ValidationError as exc:
        raise ClusterManifestError(f"cluster manifest semantic validation failed: {exc}") from exc
    semantic_errors = validate_cluster_manifest_semantics(
        manifest,
        registered_sources=registered_sources,
    )
    if semantic_errors:
        msg = "cluster manifest semantic validation failed: " + "; ".join(semantic_errors)
        raise ClusterManifestError(msg)
    return manifest


def validate_cluster_manifest_semantics(
    manifest: EvidenceClusterManifest,
    *,
    registered_sources: list[SourceRegistryEntry] | None = None,
) -> list[str]:
    errors: list[str] = []
    registered_ids = {source.source_id for source in registered_sources or []}
    for target in manifest.targets:
        target_pages = 0
        for assignment in target.source_assignments:
            if registered_sources is not None and assignment.source_id not in registered_ids:
                errors.append(f"{target.target_id}: unknown source_id {assignment.source_id!r}")
            assignment_pages = 0
            for page_range in assignment.candidate_page_ranges:
                page_count = page_range.page_end - page_range.page_start + 1
                assignment_pages += page_count
                target_pages += page_count
            if assignment_pages > assignment.max_pages_for_assignment:
                errors.append(
                    f"{target.target_id}/{assignment.source_id}: page range count "
                    f"{assignment_pages} exceeds assignment cap "
                    f"{assignment.max_pages_for_assignment}"
                )
        if target_pages > target.max_pages_per_target:
            errors.append(
                f"{target.target_id}: page range count {target_pages} exceeds target cap "
                f"{target.max_pages_per_target}"
            )
    total_candidate_pages = len(deduplicated_page_candidates(manifest))
    if total_candidate_pages > manifest.total_page_cap:
        errors.append(
            f"deduplicated candidate pages {total_candidate_pages} exceed total page cap "
            f"{manifest.total_page_cap}"
        )
    output_path = Path(manifest.output_dir)
    if not manifest.output_dir.strip() or output_path.is_absolute():
        errors.append("output_dir must be a non-empty relative path and must not influence logic")
    return errors


def manifest_identity(manifest: EvidenceClusterManifest) -> str:
    payload = manifest.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_cluster_preparation_plan(
    manifest: EvidenceClusterManifest,
    *,
    cached_pages: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    cached_lookup = {
        (source_id, int(page["page_number"])): page
        for source_id, pages in (cached_pages or {}).items()
        for page in pages
    }
    frozen_targets = [frozen_target_record(target) for target in manifest.targets]
    page_candidates = page_candidate_records(manifest, cached_lookup)
    deduplicated = deduplicated_page_candidate_records(page_candidates)
    target_assignments = target_assignment_records(manifest, page_candidates)
    return {
        "schema_version": "segro_evidence_cluster_preparation_plan_v1",
        "manifest_identity": manifest_identity(manifest),
        "run_name": manifest.run_name,
        "unit_id": manifest.unit_id,
        "output_dir": manifest.output_dir,
        "frozen_target_manifest": frozen_targets,
        "page_acquisition_manifest": page_candidates,
        "page_deduplication_summary": {
            "candidate_page_references": len(page_candidates),
            "deduplicated_page_count": len(deduplicated),
            "deduplication_savings": len(page_candidates) - len(deduplicated),
            "deduplicated_pages": deduplicated,
        },
        "target_assignment_summary": target_assignments,
        "readiness_disposition_policy": list(manifest.allowed_readiness_dispositions),
        "business_outcomes_generated": False,
    }


def frozen_target_record(target: ClusterTarget) -> dict[str, Any]:
    return {
        "target_id": target.target_id,
        "field_name": target.field_name,
        "domain": target.domain,
        "value_shape": target.value_shape,
        "requested_attribute": target.requested_attribute,
        "expected_evidence_families": list(target.expected_evidence_families),
        "max_pages_per_target": target.max_pages_per_target,
    }


def page_candidate_records(
    manifest: EvidenceClusterManifest,
    cached_lookup: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in manifest.targets:
        for assignment in target.source_assignments:
            for page_range in assignment.candidate_page_ranges:
                for page_number in range(page_range.page_start, page_range.page_end + 1):
                    cached_page = cached_lookup.get((assignment.source_id, page_number))
                    text = str((cached_page or {}).get("extracted_text") or "")
                    rows.append(
                        {
                            "target_id": target.target_id,
                            "field_name": target.field_name,
                            "source_id": assignment.source_id,
                            "source_document_role": assignment.source_document_role,
                            "logical_source_path": assignment.logical_source_path,
                            "section_label": assignment.section_label,
                            "page_number": page_number,
                            "requested_attribute": target.requested_attribute,
                            "expected_evidence_families": list(target.expected_evidence_families),
                            "allowed_methods": list(assignment.allowed_methods),
                            "expected_cache_state": page_range.expected_cache_state,
                            "cache_state": cache_state_for(cached_page, text),
                            "acquisition_disposition": acquisition_disposition_for(
                                cached_page,
                                text,
                                assignment.allowed_methods,
                            ),
                            "range_rationale": page_range.rationale,
                        }
                    )
    return rows


def deduplicated_page_candidates(manifest: EvidenceClusterManifest) -> set[tuple[str, int]]:
    pages: set[tuple[str, int]] = set()
    for target in manifest.targets:
        for assignment in target.source_assignments:
            for page_range in assignment.candidate_page_ranges:
                pages.update(
                    (assignment.source_id, page_number)
                    for page_number in range(page_range.page_start, page_range.page_end + 1)
                )
    return pages


def deduplicated_page_candidate_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[str]] = defaultdict(list)
    source_meta: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["source_id"]), int(row["page_number"]))
        grouped[key].append(str(row["target_id"]))
        source_meta[key] = row
    return [
        {
            "source_id": source_id,
            "page_number": page_number,
            "target_ids": sorted(set(target_ids)),
            "cache_state": source_meta[(source_id, page_number)]["cache_state"],
            "acquisition_disposition": source_meta[(source_id, page_number)][
                "acquisition_disposition"
            ],
        }
        for (source_id, page_number), target_ids in sorted(grouped.items())
    ]


def target_assignment_records(
    manifest: EvidenceClusterManifest,
    page_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in page_candidates:
        by_target[str(row["target_id"])].append(row)
    records = []
    for target in manifest.targets:
        rows = by_target[target.target_id]
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[str(row["acquisition_disposition"])] += 1
        records.append(
            {
                "target_id": target.target_id,
                "field_name": target.field_name,
                "candidate_page_count": len(rows),
                "disposition_counts": dict(sorted(counts.items())),
                "ready_for_cached_review": counts["cached_text_available"] > 0,
                "requires_acquisition": any(
                    row["acquisition_disposition"] == "uncached_parsing_candidate"
                    for row in rows
                ),
            }
        )
    return records


def cache_state_for(cached_page: dict[str, Any] | None, text: str) -> str:
    if cached_page is None:
        return "not_cached"
    if text.strip():
        return "cached_text_available"
    return "cached_text_empty"


def acquisition_disposition_for(
    cached_page: dict[str, Any] | None,
    text: str,
    allowed_methods: list[EvidenceAcquisitionMethod],
) -> AcquisitionDisposition:
    if cached_page is not None and text.strip():
        return "cached_text_available"
    if cached_page is not None:
        return "already_cached_but_text_empty"
    if "native_parser" in allowed_methods:
        return "uncached_parsing_candidate"
    return "method_not_allowed"


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ClusterManifestError(f"Expected JSON object: {path}")
    return payload
