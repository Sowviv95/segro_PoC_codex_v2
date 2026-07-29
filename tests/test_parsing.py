from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest
from openpyxl import Workbook
from pypdf import PdfWriter
from typer.testing import CliRunner

from segro_evidence_extraction.cli import app
from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry
from segro_evidence_extraction.parsing import ParseOptions, ParsingConfig, parse_sources
from segro_evidence_extraction.parsing.cache import JsonParseCache, cache_key, config_hash
from segro_evidence_extraction.parsing.classification import (
    CLASSIFIER_VERSION,
    classify_page_evidence,
)
from segro_evidence_extraction.parsing.models import (
    NORMALIZATION_VERSION,
    OcrRouting,
    ParsingStatus,
)
from segro_evidence_extraction.parsing.normalization import normalize_text
from segro_evidence_extraction.parsing.pdf import PDF_PARSER_NAME, PDF_PARSER_VERSION
from segro_evidence_extraction.parsing.quality import assess_text_quality, route_ocr
from segro_evidence_extraction.parsing.registry import default_parser_registry


def test_parser_registry_selects_supported_parsers() -> None:
    registry = default_parser_registry()
    assert registry.get(_source("doc.pdf", FileType.PDF)).parser_name == "pypdf-incremental"  # type: ignore[union-attr]
    assert registry.get(_source("sheet.xlsx", FileType.XLSX)).parser_name == "openpyxl-read-only"  # type: ignore[union-attr]
    assert registry.get(_source("archive.zip", FileType.ZIP)) is None


def test_incremental_pdf_page_range_and_no_read_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _case_dir("pdf_range")
    pdf = root / "sample.pdf"
    _write_pdf(pdf, 5)
    manifest = _write_registry(root, [_source(str(pdf), FileType.PDF, size=pdf.stat().st_size)])

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("Path.read_bytes must not be used by parsing")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    result = parse_sources(
        manifest,
        output_dir=root / "out",
        cache_dir=root / "cache",
        options=ParseOptions(page_start=2, page_end=5, max_pages=2),
        write_artifacts=False,
    )
    assert [page.page_number for page in result.pages] == [2, 3]
    assert result.summary.pages_parsed == 2
    assert result.summary.ocr_required_pages == 2


def test_pdf_source_opened_once(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _case_dir("pdf_open_once")
    pdf = root / "sample.pdf"
    _write_pdf(pdf, 3)
    manifest = _write_registry(root, [_source(str(pdf), FileType.PDF, size=pdf.stat().st_size)])
    original_open = Path.open
    opens = 0

    def counting_open(path: Path, *args: object, **kwargs: object) -> object:
        nonlocal opens
        if path == pdf:
            opens += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    parse_sources(
        manifest,
        output_dir=root / "out",
        cache_dir=root / "cache",
        options=ParseOptions(max_pages=2, use_cache=False),
        write_artifacts=False,
    )
    assert opens == 1


def test_cache_key_and_cache_hit_behavior() -> None:
    root = _case_dir("cache")
    pdf = root / "sample.pdf"
    _write_pdf(pdf, 2)
    source = _source(str(pdf), FileType.PDF, size=pdf.stat().st_size)
    manifest = _write_registry(root, [source])
    first = parse_sources(
        manifest,
        output_dir=root / "out1",
        cache_dir=root / "cache",
        options=ParseOptions(max_pages=1),
        write_artifacts=False,
    )
    second = parse_sources(
        manifest,
        output_dir=root / "out2",
        cache_dir=root / "cache",
        options=ParseOptions(max_pages=1),
        write_artifacts=False,
    )
    assert first.summary.cache_misses == 1
    assert second.summary.cache_hits == 1
    assert second.pages[0].status == ParsingStatus.CACHE_HIT


def test_corrupted_cache_is_treated_as_miss() -> None:
    root = _case_dir("corrupt_cache")
    cache = JsonParseCache(root / "cache")
    key = cache_key(
        source_id="src_1",
        content_hash="abc",
        page_or_sheet="page-1",
        parser_name=PDF_PARSER_NAME,
        parser_version=PDF_PARSER_VERSION,
        normalization_version=NORMALIZATION_VERSION,
        classification_version=CLASSIFIER_VERSION,
        config_digest=config_hash(ParsingConfig()),
    )
    (root / "cache" / f"{key}.json").write_text("{not-json", encoding="utf-8")
    assert cache.read_page(key) is None
    assert cache.corrupt == 1


def test_xlsx_read_only_and_csv_streaming() -> None:
    root = _case_dir("spreadsheets")
    xlsx = root / "book.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Assets"
    sheet.append(["Asset", "Value"])
    sheet.append(["Pump", "=1+1"])
    workbook.save(xlsx)
    csv_path = root / "rows.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    manifest = _write_registry(
        root,
        [
            _source(str(xlsx), FileType.XLSX, size=xlsx.stat().st_size),
            _source(str(csv_path), FileType.CSV, size=csv_path.stat().st_size),
        ],
    )
    result = parse_sources(manifest, output_dir=root / "out", cache_dir=root / "cache")
    assert result.summary.sheets_parsed == 2
    assert any(sheet.preview_text and "=1+1" in sheet.preview_text for sheet in result.sheets)
    assert result.summary.table_candidates == 2


def test_text_normalization_quality_classification_and_ocr_routing() -> None:
    text = normalize_text("Certificate\r\n\r\n  Test   Report\t123")
    quality = assess_text_quality(text)
    route, _rationale, _confidence = route_ocr(quality, "pdf")
    classification = classify_page_evidence(
        source_id="src_1",
        page_or_sheet="page-1",
        subject_type="page",  # type: ignore[arg-type]
        text=text,
        quality=quality,
        file_type="pdf",
        table_indicator=False,
        drawing_indicator=False,
    )
    assert text == "Certificate\n\nTest Report 123"
    assert route in {OcrRouting.NOT_REQUIRED, OcrRouting.UNSUITABLE_UNKNOWN}
    assert classification.primary_label in {"certificate", "test report"}
    sparse = assess_text_quality("")
    sparse_route, _rationale, _confidence = route_ocr(sparse, "pdf")
    assert sparse_route == OcrRouting.REQUIRED


def test_page_failure_recovery_and_unsupported_files() -> None:
    root = _case_dir("failures")
    bad_pdf = root / "bad.pdf"
    bad_pdf.write_text("not a pdf", encoding="utf-8")
    zip_path = root / "archive.zip"
    zip_path.write_text("zip placeholder", encoding="utf-8")
    manifest = _write_registry(
        root,
        [
            _source(str(bad_pdf), FileType.PDF, size=bad_pdf.stat().st_size),
            _source(str(zip_path), FileType.ZIP, size=zip_path.stat().st_size),
        ],
    )
    result = parse_sources(manifest, output_dir=root / "out", cache_dir=root / "cache")
    assert any(issue.code == "pdf_document_open_failed" for issue in result.issues)
    assert any(issue.code == "unsupported_parser" for issue in result.issues)


def test_artifacts_and_progress_callback_are_deterministic() -> None:
    root = _case_dir("artifacts")
    pdf = root / "sample.pdf"
    _write_pdf(pdf, 2)
    manifest = _write_registry(root, [_source(str(pdf), FileType.PDF, size=pdf.stat().st_size)])
    progress: list[tuple[str, str]] = []
    result = parse_sources(
        manifest,
        output_dir=root / "out",
        cache_dir=root / "cache",
        options=ParseOptions(max_pages=1, progress_every=1),
        progress=lambda stage, subject: progress.append((stage, subject)),
    )
    expected = {
        "parsing_manifest.json",
        "parsed_documents.jsonl",
        "parsed_pages.jsonl",
        "parsed_sheets.jsonl",
        "page_classifications.csv",
        "table_candidates.jsonl",
        "drawing_candidates.jsonl",
        "parsing_issues.csv",
        "parsing_summary.json",
        "parsing_summary.md",
        "document_timings.csv",
        "slow_pages.csv",
        "stage_timings.json",
    }
    assert expected.issubset({path.name for path in (root / "out").iterdir()})
    assert result.summary.pages_parsed == 1
    assert any(stage == "parse:page" for stage, _subject in progress)


def test_slow_page_warning_and_no_ocr_or_network_references() -> None:
    root = _case_dir("slow")
    pdf = root / "sample.pdf"
    _write_pdf(pdf, 1)
    manifest = _write_registry(root, [_source(str(pdf), FileType.PDF, size=pdf.stat().st_size)])
    result = parse_sources(
        manifest,
        output_dir=root / "out",
        cache_dir=root / "cache",
        config=ParsingConfig(max_seconds_per_page_warning=0.0001),
        options=ParseOptions(max_pages=1, use_cache=False),
        write_artifacts=False,
    )
    assert any(warning.code == "slow_page" for page in result.pages for warning in page.warnings)
    parsing_root = Path("src/segro_evidence_extraction/parsing")
    text = "\n".join(path.read_text(encoding="utf-8") for path in parsing_root.glob("*.py"))
    assert "openai" not in text.lower()
    assert "requests" not in text.lower()
    assert "ocr" in text.lower()


def test_cli_parse_inspect_and_run_exit_codes() -> None:
    root = _case_dir("cli")
    pdf = root / "sample.pdf"
    _write_pdf(pdf, 1)
    manifest = _write_registry(root, [_source(str(pdf), FileType.PDF, size=pdf.stat().st_size)])
    runner = CliRunner()
    inspect_result = runner.invoke(app, ["parse", "inspect", "--source-manifest", str(manifest)])
    assert inspect_result.exit_code == 0
    assert '"sources": 1' in inspect_result.output
    run_result = runner.invoke(
        app,
        [
            "parse",
            "run",
            "--source-manifest",
            str(manifest),
            "--output-dir",
            str(root / "out"),
            "--cache-dir",
            str(root / "cache"),
            "--max-pages",
            "1",
        ],
    )
    assert run_result.exit_code == 0
    assert '"pages_parsed": 1' in run_result.output


def test_cli_timing_files_are_finalized_before_write() -> None:
    root = _case_dir("cli_timing_finalized")
    pdf = root / "sample.pdf"
    _write_pdf(pdf, 5)
    manifest = _write_registry(root, [_source(str(pdf), FileType.PDF, size=pdf.stat().st_size)])
    output_dir = root / "out"
    runner = CliRunner()
    run_result = runner.invoke(
        app,
        [
            "parse",
            "run",
            "--source-manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(root / "cache"),
            "--max-pages",
            "5",
            "--no-cache",
        ],
    )
    assert run_result.exit_code == 0
    stage_timings = json.loads((output_dir / "stage_timings.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "parsing_summary.json").read_text(encoding="utf-8"))
    required = {
        "total_ms",
        "parse_loop_ms",
        "artifact_writing_ms",
        "cache_read_ms",
        "cache_write_ms",
        "document_open_ms",
        "text_extraction_ms",
        "normalization_ms",
        "quality_assessment_ms",
        "classification_ms",
    }
    assert required.issubset(stage_timings)
    assert stage_timings["artifact_writing_ms"] >= 0
    assert all(stage_timings["total_ms"] >= stage_timings[name] for name in required)
    assert summary["total_runtime_ms"] == stage_timings["total_ms"]
    assert summary["total_page_sheet_units"] == summary["pages_parsed"] + summary["sheets_parsed"]
    assert summary["total_page_sheet_units"] == 5
    assert (output_dir / "stage_timings.json").stat().st_mtime >= (
        output_dir / "document_timings.csv"
    ).stat().st_mtime


def test_synthetic_5_page_pdf_under_two_seconds() -> None:
    root = _case_dir("perf_5")
    pdf = root / "sample.pdf"
    _write_pdf(pdf, 5)
    manifest = _write_registry(root, [_source(str(pdf), FileType.PDF, size=pdf.stat().st_size)])
    start = time.perf_counter()
    parse_sources(
        manifest,
        output_dir=root / "out",
        cache_dir=root / "cache",
        options=ParseOptions(max_pages=5, use_cache=False),
        write_artifacts=False,
    )
    assert time.perf_counter() - start < 2


def _case_dir(name: str) -> Path:
    root = Path("output/test_parsing") / name
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


def _source(path: str, file_type: FileType, *, size: int = 0) -> SourceRegistryEntry:
    file_path = Path(path)
    return SourceRegistryEntry(
        source_id=f"src_{abs(hash((str(file_path), file_type))) % 10_000_000}",
        original_path=str(file_path),
        logical_path=file_path.name,
        file_type=file_type,
        extension=file_path.suffix,
        mime_type=None,
        file_hash=f"{abs(hash(str(file_path))) % 10_000_000:064d}"[:64],
        size_bytes=size,
        content_identity="sha256:test",
    )


def _write_registry(root: Path, sources: list[SourceRegistryEntry]) -> Path:
    registry = root / "source_registry.jsonl"
    with registry.open("w", encoding="utf-8", newline="\n") as handle:
        for source in sources:
            handle.write(source.model_dump_json() + "\n")
    manifest = root / "source_pack_manifest.json"
    manifest.write_text(
        json.dumps({"output_paths": {"source_registry_jsonl": str(registry)}}),
        encoding="utf-8",
    )
    return manifest
