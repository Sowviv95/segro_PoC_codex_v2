from __future__ import annotations

import json
import multiprocessing
import shutil
import time
from pathlib import Path

from pypdf import PdfWriter

from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry
from segro_evidence_extraction.parsing.benchmark import (
    BenchmarkSample,
    _run_isolated_repetition,
    run_parser_benchmark,
)
from segro_evidence_extraction.parsing.models import ParsedPage, ParseOptions, ParsingConfig
from segro_evidence_extraction.parsing.pdf import (
    PdfPageParser,
    PyMuPdfPageParser,
    bounded_page_numbers,
)


def sleeping_benchmark_worker(
    parser_name: str,
    sample_payload: dict[str, object],
    result_path: str,
) -> None:
    _ = (parser_name, sample_payload, result_path)
    time.sleep(10)


def test_both_pdf_parser_adapters_parse_small_fixture() -> None:
    root = _case_dir("adapters")
    pdf = root / "sample.pdf"
    _write_pdf(pdf, 2)
    source = _source(pdf)
    for parser in [PdfPageParser(), PyMuPdfPageParser()]:
        items = list(
            parser.parse(
                source,
                source_path=pdf,
                config=ParsingConfig(),
                options=ParseOptions(page_start=1, page_end=1, use_cache=False),
                cache=None,
                progress=None,
            )
        )
        pages = [item for item in items if isinstance(item, ParsedPage)]
        assert len(pages) == 1
        assert pages[0].parser_name == parser.parser_name
        assert pages[0].parser_version == parser.parser_version


def test_explicit_bounded_range_enforcement() -> None:
    assert bounded_page_numbers(2, 4) == [2, 3, 4]
    try:
        bounded_page_numbers(4, 2)
    except ValueError as exc:
        assert "page_end" in str(exc)
    else:
        raise AssertionError("Expected invalid bounded range to fail")


def test_benchmark_timeout_reporting_and_subprocess_cleanup() -> None:
    root = _case_dir("timeout")
    pdf = root / "sample.pdf"
    _write_pdf(pdf, 1)
    run = _run_isolated_repetition(
        parser_name="pypdf",
        sample=BenchmarkSample(identifier="slow", path=str(pdf), page_start=1, page_end=1),
        repetition=1,
        timeout_seconds=0.2,
        worker_target=sleeping_benchmark_worker,
    )
    assert run.timeout is True
    assert run.success is False
    assert run.exception_type == "TimeoutError"
    assert run.remaining_active_child_count_after_cleanup == 0
    assert multiprocessing.active_children() == []


def test_benchmark_failure_reporting() -> None:
    root = _case_dir("failure")
    bad_pdf = root / "bad.pdf"
    bad_pdf.write_text("not a pdf", encoding="utf-8")
    run = _run_isolated_repetition(
        parser_name="pypdf",
        sample=BenchmarkSample(identifier="bad", path=str(bad_pdf), page_start=1, page_end=1),
        repetition=1,
        timeout_seconds=5.0,
    )
    assert run.success is False
    assert run.timeout is False
    assert run.warnings
    assert "pdf_document_open_failed" in run.warnings[0]
    assert run.remaining_active_child_count_after_cleanup == 0


def test_benchmark_artifacts_are_deterministic_for_shape() -> None:
    root = _case_dir("artifacts")
    pdf = root / "sample.pdf"
    _write_pdf(pdf, 1)
    output_dir = root / "out"
    result = run_parser_benchmark(
        source_manifest=root / "unused.json",
        output_dir=output_dir,
        sample_specs=[f"fixture={pdf}:1-1"],
        repetitions=1,
        timeout_seconds=10.0,
    )
    expected_files = {"parser_benchmark_results.json", "parser_benchmark_comparison.md"}
    assert expected_files == {path.name for path in output_dir.iterdir()}
    payload = json.loads((output_dir / "parser_benchmark_results.json").read_text(encoding="utf-8"))
    assert payload["benchmark_version"] == "parser-foundation-stage-a-v1"
    assert payload["samples"][0]["identifier"] == "fixture"
    assert [run["parser_name"] for run in payload["runs"]] == ["pymupdf", "pypdf"]
    assert "Recommend" in (output_dir / "parser_benchmark_comparison.md").read_text(
        encoding="utf-8"
    )
    assert result.artifact_paths["detailed_json"].endswith("parser_benchmark_results.json")


def _case_dir(name: str) -> Path:
    root = Path("output/test_parser_benchmark") / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def _write_pdf(path: Path, pages: int) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)


def _source(path: Path) -> SourceRegistryEntry:
    return SourceRegistryEntry(
        source_id="src_fixture",
        original_path=str(path),
        logical_path=path.name,
        file_type=FileType.PDF,
        extension=".pdf",
        mime_type="application/pdf",
        file_hash="a" * 64,
        size_bytes=path.stat().st_size,
        content_identity="sha256:fixture",
    )
