from __future__ import annotations

import json
import multiprocessing
import shutil
from collections.abc import Iterator
from pathlib import Path

from pypdf import PdfWriter

from segro_evidence_extraction.models.common import ProvenanceRef
from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry
from segro_evidence_extraction.parsing.batch_worker import BatchRequest, BatchWorkerConfig
from segro_evidence_extraction.parsing.cache import JsonParseCache
from segro_evidence_extraction.parsing.interfaces import ProgressCallback
from segro_evidence_extraction.parsing.models import (
    DocumentTiming,
    OcrRouting,
    ParsedPage,
    ParseOptions,
    ParserWarning,
    ParsingConfig,
    ParsingStatus,
)
from segro_evidence_extraction.parsing.page_cache import (
    PARSED_PAGE_SCHEMA_VERSION,
    CachedBatchParsingService,
    CanonicalParsedPage,
    CanonicalParsedPageCache,
    cache_key_composition,
)
from segro_evidence_extraction.parsing.quality import assess_text_quality


class CacheFixtureParser:
    parser_name = "cache-fake"
    parser_version = "cache-fake-v1"

    def parse(
        self,
        source: SourceRegistryEntry,
        *,
        source_path: Path,
        config: ParsingConfig,
        options: ParseOptions,
        cache: JsonParseCache | None,
        progress: ProgressCallback | None,
    ) -> Iterator[ParsedPage | ParserWarning | DocumentTiming]:
        _ = (source_path, config, cache)
        _record_invocation(source)
        if progress:
            progress("parse:document_opened", source.logical_path)
        assert options.page_start is not None
        assert options.page_end is not None
        for page_number in range(options.page_start, options.page_end + 1):
            if source.logical_role == "fail-page-2" and page_number == 2:
                yield ParserWarning(
                    code="pdf_page_parse_failed",
                    message="Synthetic failure.",
                    severity="error",
                    source_id=source.source_id,
                    page_or_sheet="page-2",
                )
                continue
            yield _page(
                source,
                page_number,
                with_warning=source.logical_role == "warn-page-1" and page_number == 1,
            )
        yield DocumentTiming(
            source_id=source.source_id,
            logical_path=source.logical_path,
            parser_name=self.parser_name,
            pages_or_sheets=options.page_end - options.page_start + 1,
        )


def cache_fixture_parser_factory() -> CacheFixtureParser:
    return CacheFixtureParser()


def test_first_invocation_writes_artifacts_second_is_full_cache_hit() -> None:
    root = _case_dir("cold_warm")
    request = _request(root, start=1, end=3)
    service = _service(root)
    cold = service.parse(request)
    warm = service.parse(request)
    assert cold.pages_newly_parsed == [1, 2, 3]
    assert cold.page_artifacts_written == 3
    assert warm.cache_hits == 3
    assert warm.worker_invocation_count == 0
    assert _invocations(request.source) == ["1-3"]
    assert [page.page_number for page in warm.pages] == [1, 2, 3]
    assert multiprocessing.active_children() == []


def test_partial_cache_reuse_parses_only_missing_pages_and_preserves_order() -> None:
    root = _case_dir("partial")
    service = _service(root)
    request_1_4 = _request(root, start=1, end=4)
    service.parse(request_1_4)
    request_1_6 = _request(root, start=1, end=6)
    result = service.parse(request_1_6)
    assert result.cache_hits == 4
    assert result.cache_misses == 2
    assert result.missing_ranges_sent_to_workers == ["5-6"]
    assert result.pages_newly_parsed == [5, 6]
    assert [page.page_number for page in result.pages] == [1, 2, 3, 4, 5, 6]
    assert _invocations(request_1_6.source) == ["1-4", "5-6"]
    assert multiprocessing.active_children() == []


def test_corrupt_artifact_is_reported_and_reparsed() -> None:
    root = _case_dir("corrupt")
    service = _service(root)
    request = _request(root, start=1, end=1)
    service.parse(request)
    path = next((root / "cache").rglob("page_000001.json"))
    path.write_text("{not-json", encoding="utf-8")
    result = service.parse(request)
    assert result.invalid_cache_entries == 1
    assert result.cache_warnings[0].code == "invalid_cache_artifact"
    assert result.pages_newly_parsed == [1]
    assert _invocations(request.source) == ["1-1", "1-1"]


def test_interrupted_temp_file_is_not_cache_hit_or_invalid() -> None:
    root = _case_dir("temp_file")
    service = _service(root)
    request = _request(root, start=1, end=1)
    service.parse(request)
    artifact = next((root / "cache").rglob("page_000001.json"))
    artifact.with_name(".page_000001.json.incomplete.tmp").write_text("{bad", encoding="utf-8")
    result = service.parse(request)
    assert result.cache_hits == 1
    assert result.invalid_cache_entries == 0
    assert result.worker_invocation_count == 0


def test_identity_changes_invalidate_cache_entries() -> None:
    root = _case_dir("identity")
    request = _request(root, start=1, end=1)
    _service(root).parse(request)
    assert _service(root).parse(request).cache_hits == 1
    changed_hash = request.model_copy(
        update={"source": request.source.model_copy(update={"file_hash": "c" * 64})}
    )
    assert _service(root).parse(changed_hash).cache_misses == 1
    assert _service(root, parser_name="other").parse(request).cache_misses == 1
    assert _service(root, parser_version="other-v").parse(request).cache_misses == 1
    assert _service(root, config=ParsingConfig(max_extracted_chars_per_page=123)).parse(
        request
    ).cache_misses == 1
    schema_cache = CanonicalParsedPageCache(root / "cache", schema_version="schema-v2")
    schema_service = CachedBatchParsingService(
        cache=schema_cache,
        batch_config=BatchWorkerConfig(stall_threshold_seconds=0.3),
        parser_factory=cache_fixture_parser_factory,
    )
    assert schema_service.parse(request).cache_misses == 1


def test_page_warnings_survive_persistence_and_reload() -> None:
    root = _case_dir("warnings")
    request = _request(root, start=1, end=1, role="warn-page-1")
    service = _service(root)
    service.parse(request)
    warm = service.parse(request)
    assert warm.pages[0].status == ParsingStatus.CACHE_HIT
    assert [warning.code for warning in warm.pages[0].warnings] == ["synthetic_warning"]


def test_failed_page_is_not_reused_as_valid_cache_success() -> None:
    root = _case_dir("failure")
    request = _request(root, start=1, end=2, role="fail-page-2")
    service = _service(root)
    first = service.parse(request)
    second = service.parse(request)
    assert first.pages_newly_parsed == [1]
    assert [page.page_number for page in second.pages] == [1]
    assert second.cache_hits == 1
    assert second.cache_misses == 1
    assert second.worker_invocation_count == 1
    assert _invocations(request.source) == ["1-2", "2-2"]


def test_cache_paths_and_key_composition_are_deterministic() -> None:
    root = _case_dir("deterministic")
    request = _request(root, start=1, end=1)
    service = _service(root)
    first = service.parse(request)
    path = next((root / "cache").rglob("page_000001.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.relative_to(root / "cache").parts[0] == request.source.source_id
    assert payload["schema_version"] == PARSED_PAGE_SCHEMA_VERSION
    assert payload["cache_key"] == payload["artifact_identity"]
    assert cache_key_composition() == [
        "schema_version",
        "source_content_hash",
        "page_number",
        "parser_name",
        "parser_version",
        "parser_config_fingerprint",
    ]
    second = service.parse(request)
    assert first.pages_newly_parsed == [1]
    assert second.worker_invocation_count == 0


def test_failed_artifact_is_not_valid_cache_hit() -> None:
    root = _case_dir("failed_artifact")
    request = _request(root, start=1, end=1)
    service = _service(root)
    service.parse(request)
    path = next((root / "cache").rglob("page_000001.json"))
    artifact = CanonicalParsedPage.model_validate_json(path.read_text(encoding="utf-8"))
    path.write_text(
        artifact.model_copy(update={"page_parse_status": "failed"}).model_dump_json(),
        encoding="utf-8",
    )
    result = service.parse(request)
    assert result.invalid_cache_entries == 1
    assert result.pages_newly_parsed == [1]


def _service(
    root: Path,
    *,
    parser_name: str = "cache-fake",
    parser_version: str = "cache-fake-v1",
    config: ParsingConfig | None = None,
) -> CachedBatchParsingService:
    return CachedBatchParsingService(
        cache=CanonicalParsedPageCache(root / "cache"),
        batch_config=BatchWorkerConfig(stall_threshold_seconds=0.3),
        parser_factory=cache_fixture_parser_factory,
        parser_name=parser_name,
        parser_version=parser_version,
        parser_config=config,
    )


def _case_dir(name: str) -> Path:
    root = Path("output/test_page_cache") / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def _request(root: Path, *, start: int, end: int, role: str = "normal") -> BatchRequest:
    pdf = root / "sample.pdf"
    _write_pdf(pdf, pages=max(end, 1))
    source = SourceRegistryEntry(
        source_id=f"src_{root.name}",
        original_path=str(pdf),
        logical_path=pdf.name,
        logical_role=role,
        file_type=FileType.PDF,
        extension=".pdf",
        mime_type="application/pdf",
        file_hash="d" * 64,
        size_bytes=pdf.stat().st_size,
        content_identity="sha256:test",
        metadata={"invoke_marker": str(root / "invocations.txt")},
    )
    return BatchRequest(
        source=source,
        source_path=str(pdf),
        output_dir=str(root / "workers"),
        page_start=start,
        page_end=end,
    )


def _write_pdf(path: Path, pages: int) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)


def _page(source: SourceRegistryEntry, page_number: int, *, with_warning: bool) -> ParsedPage:
    text = f"Canonical cache page {page_number}"
    quality = assess_text_quality(text)
    warnings = [
        ParserWarning(
            code="synthetic_warning",
            message="Synthetic warning survives cache.",
            source_id=source.source_id,
            page_or_sheet=f"page-{page_number}",
        )
    ] if with_warning else []
    return ParsedPage(
        page_id=f"{source.source_id}:p{page_number}",
        source_id=source.source_id,
        content_hash=source.file_hash,
        page_number=page_number,
        physical_page_index=page_number - 1,
        parser_name="cache-fake",
        parser_version="cache-fake-v1",
        text=text,
        character_count=quality.character_count,
        word_count=quality.word_count,
        quality=quality,
        scan_likelihood=quality.scan_likelihood,
        ocr_routing=OcrRouting.NOT_REQUIRED,
        ocr_rationale="Synthetic cache parser.",
        status=ParsingStatus.PARSED_WITH_WARNINGS if warnings else ParsingStatus.PARSED,
        warnings=warnings,
        parsing_duration_ms=1.0,
        provenance=[
            ProvenanceRef(
                source_id=source.source_id,
                page_or_sheet=f"page-{page_number}",
                notes="Synthetic canonical cache parser.",
            )
        ],
    )


def _record_invocation(source: SourceRegistryEntry) -> None:
    marker_value = source.metadata.get("invoke_marker")
    assert isinstance(marker_value, str)
    marker = Path(marker_value)
    with marker.open("a", encoding="utf-8", newline="\n") as handle:
        # The range is inferred by caller through worker output, but this proves invocation count.
        handle.write("invoked\n")


def _invocations(source: SourceRegistryEntry) -> list[str]:
    marker_value = source.metadata.get("invoke_marker")
    assert isinstance(marker_value, str)
    marker = Path(marker_value)
    if not marker.exists():
        return []
    results = sorted(marker.parent.glob("workers/.w/*/batch_worker_result.json"))
    ranges: list[str] = []
    for path in results:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ranges.append(f"{payload['requested_page_start']}-{payload['requested_page_end']}")
    return ranges
