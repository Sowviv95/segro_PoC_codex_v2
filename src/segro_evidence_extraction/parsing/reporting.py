"""Parsing artifact writing."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from segro_evidence_extraction.parsing.models import ParsingResult


class JsonSerializableModel(Protocol):
    def model_dump_json(self) -> str: ...


def write_parsing_artifacts(result: ParsingResult, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    import time

    artifact_start = time.perf_counter()
    paths = {
        "parsing_manifest": output_dir / "parsing_manifest.json",
        "parsed_documents": output_dir / "parsed_documents.jsonl",
        "parsed_pages": output_dir / "parsed_pages.jsonl",
        "parsed_sheets": output_dir / "parsed_sheets.jsonl",
        "page_classifications": output_dir / "page_classifications.csv",
        "table_candidates": output_dir / "table_candidates.jsonl",
        "drawing_candidates": output_dir / "drawing_candidates.jsonl",
        "parsing_issues": output_dir / "parsing_issues.csv",
        "parsing_summary": output_dir / "parsing_summary.json",
        "parsing_summary_md": output_dir / "parsing_summary.md",
        "document_timings": output_dir / "document_timings.csv",
        "slow_pages": output_dir / "slow_pages.csv",
        "stage_timings": output_dir / "stage_timings.json",
    }
    _write_jsonl(paths["parsed_documents"], result.parsed_documents)
    _write_jsonl(paths["parsed_pages"], result.pages)
    _write_jsonl(paths["parsed_sheets"], result.sheets)
    _write_classifications(paths["page_classifications"], result)
    _write_jsonl(paths["table_candidates"], result.table_candidates)
    _write_jsonl(paths["drawing_candidates"], result.drawing_candidates)
    _write_issues(paths["parsing_issues"], result)
    _write_document_timings(paths["document_timings"], result)
    _write_slow_pages(paths["slow_pages"], result)

    result.stage_timings["artifact_writing_ms"] = (time.perf_counter() - artifact_start) * 1000
    result.stage_timings["total_ms"] = (
        result.stage_timings.get("parse_loop_ms", 0.0)
        + result.stage_timings["artifact_writing_ms"]
    )
    result.summary.total_runtime_ms = result.stage_timings["total_ms"]
    result.summary.total_page_sheet_units = result.summary.pages_parsed + result.summary.sheets_parsed
    output_paths = {name: str(path) for name, path in paths.items()}
    result.output_paths = output_paths
    result.summary.artifact_paths = output_paths

    manifest = result.model_dump(
        mode="json",
        exclude={
            "pages",
            "sheets",
            "table_candidates",
            "drawing_candidates",
            "parsed_documents",
        },
    )
    paths["parsing_manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["parsing_summary"].write_text(result.summary.model_dump_json(indent=2), encoding="utf-8")
    paths["parsing_summary_md"].write_text(_summary_md(result), encoding="utf-8")
    paths["stage_timings"].write_text(json.dumps(result.stage_timings, indent=2), encoding="utf-8")
    return output_paths


def _write_jsonl(path: Path, rows: Sequence[JsonSerializableModel]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _write_classifications(path: Path, result: ParsingResult) -> None:
    fields = [
        "classification_id",
        "source_id",
        "page_or_sheet",
        "subject_type",
        "primary_label",
        "secondary_labels",
        "confidence",
        "alternatives",
        "rationale",
        "method_type",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        units = list(result.pages) + list(result.sheets)
        for unit in units:
            classification = unit.classification
            if classification is None:
                continue
            writer.writerow(
                {
                    "classification_id": classification.classification_id,
                    "source_id": classification.evidence_references[0].source_id
                    if classification.evidence_references
                    else "",
                    "page_or_sheet": classification.evidence_references[0].page_or_sheet
                    if classification.evidence_references
                    else "",
                    "subject_type": classification.subject_type,
                    "primary_label": classification.primary_label,
                    "secondary_labels": "|".join(classification.secondary_labels),
                    "confidence": classification.confidence,
                    "alternatives": "|".join(
                        alternative.label for alternative in classification.alternative_labels
                    ),
                    "rationale": classification.rationale,
                    "method_type": classification.method_type,
                }
            )


def _write_issues(path: Path, result: ParsingResult) -> None:
    fields = ["code", "severity", "source_id", "page_or_sheet", "message", "raw_value"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for issue in result.issues:
            writer.writerow(issue.model_dump(mode="json"))


def _write_document_timings(path: Path, result: ParsingResult) -> None:
    fields = [
        "source_id",
        "logical_path",
        "parser_name",
        "open_ms",
        "iteration_ms",
        "text_extraction_ms",
        "normalization_ms",
        "quality_ms",
        "classification_ms",
        "cache_read_ms",
        "cache_write_ms",
        "total_ms",
        "pages_or_sheets",
        "failures",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for timing in result.document_timings:
            writer.writerow(timing.model_dump(mode="json"))


def _write_slow_pages(path: Path, result: ParsingResult) -> None:
    fields = ["source_id", "logical_path", "page_or_sheet", "status", "total_ms", "warnings"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for timing in result.page_timings:
            if any("slow_page" in warning for warning in timing.warnings):
                writer.writerow(
                    {
                        "source_id": timing.source_id,
                        "logical_path": timing.logical_path,
                        "page_or_sheet": timing.page_or_sheet,
                        "status": timing.status,
                        "total_ms": timing.total_ms,
                        "warnings": "|".join(timing.warnings),
                    }
                )


def _summary_md(result: ParsingResult) -> str:
    lines = [
        "# Parsing Summary",
        "",
        f"- Documents seen: {result.summary.documents_seen}",
        f"- Documents parsed: {result.summary.documents_parsed}",
        f"- Documents failed: {result.summary.documents_failed}",
        f"- Pages parsed: {result.summary.pages_parsed}",
        f"- Sheets parsed: {result.summary.sheets_parsed}",
        f"- Total runtime ms: {result.summary.total_runtime_ms:.2f}",
        f"- Cache hits: {result.summary.cache_hits}",
        f"- Cache misses: {result.summary.cache_misses}",
        f"- OCR-required pages: {result.summary.ocr_required_pages}",
        f"- Table candidates: {result.summary.table_candidates}",
        f"- Drawing candidates: {result.summary.drawing_candidates}",
        "",
        "## Page/Sheet Classifications",
    ]
    lines.extend(f"- {label}: {count}" for label, count in result.summary.classifications.items())
    return "\n".join(lines)
