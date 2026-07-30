from __future__ import annotations

import multiprocessing
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

from pypdf import PdfWriter

from segro_evidence_extraction.models.common import ProvenanceRef
from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry
from segro_evidence_extraction.parsing.batch_worker import (
    BatchRequest,
    BatchWorkerConfig,
    run_bounded_batch_worker,
)
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
from segro_evidence_extraction.parsing.quality import assess_text_quality


class ControlledBatchParser:
    parser_name = "fake-pymupdf"
    parser_version = "fake-pymupdf-v1"

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
        assert options.page_start is not None
        assert options.page_end is not None
        _record_open(source, options.page_start, options.page_end)
        if progress:
            progress("parse:document_opened", source.logical_path)
        mode = source.logical_role
        for page_number in range(options.page_start, options.page_end + 1):
            if mode == "fail-page-2" and page_number == 2:
                yield ParserWarning(
                    code="pdf_page_parse_failed",
                    message="Synthetic page failure.",
                    severity="error",
                    source_id=source.source_id,
                    page_or_sheet="page-2",
                )
                continue
            yield _page(source, page_number)
            if mode == "stall-after-2" and page_number == 2:
                time.sleep(10)
            if mode == "always-stall-after-1" and page_number == 1:
                time.sleep(10)
        yield DocumentTiming(
            source_id=source.source_id,
            logical_path=source.logical_path,
            parser_name=self.parser_name,
            pages_or_sheets=options.page_end - options.page_start + 1,
        )


def controlled_parser_factory() -> ControlledBatchParser:
    return ControlledBatchParser()


def test_stalled_worker_is_terminated_checkpointed_and_resumed() -> None:
    root = _case_dir("stall_resume")
    request = _request(root, role="stall-after-2", start=1, end=3)
    result = run_bounded_batch_worker(
        request,
        config=BatchWorkerConfig(stall_threshold_seconds=0.3, max_restarts=1),
        parser_factory=controlled_parser_factory,
    )
    assert "progress_stall" in result.termination_events
    assert result.termination_reason == "completed"
    assert result.pages_completed == [1, 2, 3]
    assert result.restart_count == 1
    assert result.first_incomplete_page is None
    assert _open_ranges(request.source) == ["1-3", "3-3"]
    assert result.active_child_count_after_cleanup == 0
    assert multiprocessing.active_children() == []


def test_restart_exhaustion_returns_normally_and_preserves_checkpoint() -> None:
    root = _case_dir("restart_exhausted")
    request = _request(root, role="always-stall-after-1", start=1, end=3)
    result = run_bounded_batch_worker(
        request,
        config=BatchWorkerConfig(stall_threshold_seconds=0.3, max_restarts=0),
        parser_factory=controlled_parser_factory,
    )
    assert result.termination_reason == "restart_exhausted"
    assert result.pages_completed == [1]
    assert result.first_incomplete_page == 2
    assert result.restart_count == 0
    assert result.active_child_count_after_cleanup == 0
    assert multiprocessing.active_children() == []


def test_page_failure_is_not_process_stall() -> None:
    root = _case_dir("page_failure")
    request = _request(root, role="fail-page-2", start=1, end=3)
    result = run_bounded_batch_worker(
        request,
        config=BatchWorkerConfig(stall_threshold_seconds=0.3, max_restarts=1),
        parser_factory=controlled_parser_factory,
    )
    assert result.termination_reason == "completed"
    assert result.pages_completed == [1, 3]
    assert [failure.page_number for failure in result.pages_failed] == [2]
    assert "progress_stall" not in result.termination_events
    assert result.active_child_count_after_cleanup == 0
    assert multiprocessing.active_children() == []


def test_worker_opens_pdf_once_per_batch() -> None:
    root = _case_dir("open_once")
    request = _request(root, role="normal", start=1, end=3)
    result = run_bounded_batch_worker(
        request,
        config=BatchWorkerConfig(stall_threshold_seconds=0.3),
        parser_factory=controlled_parser_factory,
    )
    assert result.termination_reason == "completed"
    assert _open_ranges(request.source) == ["1-3"]
    assert multiprocessing.active_children() == []


def test_windows_safe_main_boundary_is_preserved() -> None:
    main_text = Path("src/segro_evidence_extraction/__main__.py").read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in main_text


def _case_dir(name: str) -> Path:
    root = Path("output/test_batch_worker") / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def _request(root: Path, *, role: str, start: int, end: int) -> BatchRequest:
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
        file_hash="b" * 64,
        size_bytes=pdf.stat().st_size,
        content_identity="sha256:test",
        metadata={"open_marker": str(root / "open_count.txt")},
    )
    return BatchRequest(
        source=source,
        source_path=str(pdf),
        output_dir=str(root / "out"),
        page_start=start,
        page_end=end,
    )


def _write_pdf(path: Path, pages: int) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)


def _page(source: SourceRegistryEntry, page_number: int) -> ParsedPage:
    text = f"Synthetic page {page_number}"
    quality = assess_text_quality(text)
    return ParsedPage(
        page_id=f"{source.source_id}:p{page_number}",
        source_id=source.source_id,
        content_hash=source.file_hash,
        page_number=page_number,
        physical_page_index=page_number - 1,
        parser_name="fake-pymupdf",
        parser_version="fake-pymupdf-v1",
        text=text,
        character_count=quality.character_count,
        word_count=quality.word_count,
        quality=quality,
        scan_likelihood=quality.scan_likelihood,
        ocr_routing=OcrRouting.NOT_REQUIRED,
        ocr_rationale="Synthetic text fixture.",
        status=ParsingStatus.PARSED,
        parsing_duration_ms=1.0,
        provenance=[
            ProvenanceRef(
                source_id=source.source_id,
                page_or_sheet=f"page-{page_number}",
                notes="Synthetic batch parser.",
            )
        ],
    )


def _record_open(source: SourceRegistryEntry, page_start: int, page_end: int) -> None:
    marker_value = source.metadata.get("open_marker")
    assert isinstance(marker_value, str)
    marker = Path(marker_value)
    with marker.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{page_start}-{page_end}\n")


def _open_ranges(source: SourceRegistryEntry) -> list[str]:
    marker_value = source.metadata.get("open_marker")
    assert isinstance(marker_value, str)
    marker = Path(marker_value)
    return marker.read_text(encoding="utf-8").splitlines()
