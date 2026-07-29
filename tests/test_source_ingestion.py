import json
import shutil
import zipfile
from pathlib import Path

import pytest
from openpyxl import Workbook
from typer.testing import CliRunner

from segro_evidence_extraction.cli import app
from segro_evidence_extraction.models.classification import ClassificationStage
from segro_evidence_extraction.models.source import FileType
from segro_evidence_extraction.source_ingestion import ArchiveLimits, ingest_source_pack
from segro_evidence_extraction.source_ingestion.discovery import discover_sources
from segro_evidence_extraction.source_ingestion.file_types import detect_file_type
from segro_evidence_extraction.source_ingestion.hashing import hash_file, source_instance_id
from segro_evidence_extraction.source_ingestion.metadata import inspect_metadata
from segro_evidence_extraction.source_ingestion.service import inspect_source_pack

runner = CliRunner()


def test_recursive_discovery_and_deterministic_ordering() -> None:
    root = _fixture_root("discovery")

    discovered, _ = discover_sources(root)

    assert [item.relative_path for item in discovered] == sorted(
        [item.relative_path for item in discovered],
        key=str.casefold,
    )
    assert any(item.relative_path == "nested/equipment.csv" for item in discovered)
    ignored = [item for item in discovered if item.relative_path == "Thumbs.db"]
    assert ignored and ignored[0].disposition == "ignored"


def test_extension_versus_detected_type() -> None:
    root = _fixture_root("types")
    disguised = root / "manual.bin"
    disguised.write_bytes(b"%PDF-1.4\n/Type /Page\n")

    detected_type, mime_type = detect_file_type(disguised)

    assert detected_type == FileType.PDF
    assert mime_type == "application/pdf"
    assert disguised.suffix == ".bin"


def test_streaming_hash_and_stable_source_ids() -> None:
    root = _fixture_root("hashing")
    first = root / "a.txt"
    second = root / "b.txt"

    first_hash = hash_file(first)
    second_hash = hash_file(second)

    assert first_hash.content_hash == second_hash.content_hash
    assert first_hash.bytes_read == len("same content")
    assert source_instance_id(content_hash=first_hash.content_hash, logical_path="a.txt") == (
        source_instance_id(content_hash=first_hash.content_hash, logical_path="a.txt")
    )
    assert source_instance_id(content_hash=first_hash.content_hash, logical_path="a.txt") != (
        source_instance_id(content_hash=first_hash.content_hash, logical_path="b.txt")
    )


def test_metadata_pdf_and_spreadsheet() -> None:
    root = _fixture_root("metadata")

    result = ingest_source_pack(root, output_dir=root / "out")
    by_name = {Path(source.logical_path).name: source for source in result.registered_sources}

    assert by_name["manual.pdf"].page_count == 2
    assert by_name["schedule.xlsx"].sheet_count == 2
    assert "Assets" in str(by_name["schedule.xlsx"].metadata["sheet_names"])


def test_pdf_metadata_does_not_use_unbounded_read_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _empty_case_dir("pdf_no_read_bytes")
    pdf = root / "large.pdf"
    pdf.write_bytes(b"%PDF-1.4\n/Type /Page\n" + (b"x" * 1024))

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("Path.read_bytes should not be used for PDF metadata")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    metadata, warnings = inspect_metadata(pdf, FileType.PDF)

    assert metadata["page_count"] == 1
    assert warnings == []


def test_zip_member_registration_and_nested_archives() -> None:
    root = _fixture_root("zip_nested")

    result = ingest_source_pack(root, output_dir=root / "out", limits=ArchiveLimits())

    member_paths = {source.archive_member_path for source in result.registered_sources}
    assert "inside/manual.pdf" in member_paths
    assert "nested/inner.zip" in member_paths
    assert "inner.txt" in member_paths
    assert len(result.archive_summaries) == 2


def test_archive_path_traversal_rejection() -> None:
    root = _empty_case_dir("zip_traversal")
    archive = root / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")

    result = ingest_source_pack(root, output_dir=root / "out")

    codes = {
        issue.issue_code
        for summary in result.archive_summaries
        for member in summary.members
        for issue in member.warnings
    }
    assert "ARCHIVE_PATH_TRAVERSAL" in codes


def test_archive_decompression_limits() -> None:
    root = _empty_case_dir("zip_limits")
    archive = root / "limited.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("large.txt", "x" * 100)

    result = ingest_source_pack(
        root,
        output_dir=root / "out",
        limits=ArchiveLimits(max_uncompressed_bytes=10, max_member_size=10),
    )

    codes = {
        issue.issue_code
        for summary in result.archive_summaries
        for member in summary.members
        for issue in member.warnings
    }
    assert "ARCHIVE_MEMBER_SIZE_LIMIT" in codes
    assert "ARCHIVE_UNCOMPRESSED_LIMIT" in codes


def test_corrupt_archive_warning() -> None:
    root = _empty_case_dir("corrupt_zip")
    (root / "bad.zip").write_bytes(b"not a zip")

    result = ingest_source_pack(root, output_dir=root / "out")

    assert "CORRUPT_ARCHIVE" in result.summary.warnings_by_code


def test_duplicate_groups_same_content_and_same_name() -> None:
    root = _fixture_root("duplicates")

    result = ingest_source_pack(root, output_dir=root / "out")

    duplicate_types = {group.duplicate_type for group in result.duplicate_groups}
    assert "same_content_different_filenames" in duplicate_types
    assert "same_filename_different_content" in duplicate_types


def test_document_classification_and_serialization() -> None:
    root = _fixture_root("classification")

    result = ingest_source_pack(root, output_dir=root / "out")

    labels = {classification.primary_label for classification in result.classifications}
    assert "building manual" in labels
    assert "spreadsheet" in labels
    assert all(
        classification.classification_stage == ClassificationStage.SOURCE_DOCUMENT
        for classification in result.classifications
    )
    assert result.classifications[0].model_dump(mode="json")["method_type"] == "rule"


def test_unknown_classification() -> None:
    root = _empty_case_dir("unknown")
    (root / "mystery.pdf").write_bytes(b"%PDF-1.4\n/Type /Page\n")

    result = ingest_source_pack(root, output_dir=root / "out")

    assert result.summary.unknown_classifications == 1


def test_cli_sources_inspect_and_ingest() -> None:
    root = _fixture_root("cli")
    output_dir = root / "out"

    inspect_result = runner.invoke(
        app,
        ["sources", "inspect", "--source-path", str(root)],
    )
    ingest_result = runner.invoke(
        app,
        ["sources", "ingest", "--source-path", str(root), "--output-dir", str(output_dir)],
    )

    assert inspect_result.exit_code == 0
    assert "recursive_file_count" in inspect_result.output
    assert ingest_result.exit_code == 0
    assert (output_dir / "source_pack_manifest.json").exists()
    assert (output_dir / "source_registry.jsonl").exists()


def test_deterministic_artifacts() -> None:
    root = _fixture_root("artifacts")
    output_dir = _empty_case_dir("artifacts_out")

    first = ingest_source_pack(root, output_dir=output_dir)
    second = ingest_source_pack(root, output_dir=output_dir)

    assert [source.source_id for source in first.registered_sources] == [
        source.source_id for source in second.registered_sources
    ]
    summary = json.loads((output_dir / "source_summary.json").read_text())
    assert summary["files_registered"] == len(second.registered_sources)


def test_real_source_pack_smoke_skips_when_unavailable() -> None:
    source_path = Path("data/input/source_packs/enfield_unit1")
    if not source_path.exists():
        pytest.skip("Local Enfield source pack unavailable")

    report = inspect_source_pack(source_path)

    assert report["recursive_file_count"] >= 1


def test_no_network_or_model_references_in_source_ingestion() -> None:
    root = Path("src/segro_evidence_extraction/source_ingestion")
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))

    assert "openai" not in text.lower()
    assert "requests" not in text.lower()
    assert "httpx" not in text.lower()


def _fixture_root(name: str) -> Path:
    root = _empty_case_dir(name)
    (root / "nested").mkdir(parents=True, exist_ok=True)
    (root / "a.txt").write_text("same content", encoding="utf-8")
    (root / "b.txt").write_text("same content", encoding="utf-8")
    (root / "same.pdf").write_bytes(b"%PDF-1.4\n/Type /Page\n")
    (root / "nested" / "same.pdf").write_bytes(b"%PDF-1.4\n/Type /Page\nchanged\n")
    (root / "manual.pdf").write_bytes(b"%PDF-1.4\n/Type /Page\n/Type /Page\n")
    (root / "nested" / "equipment.csv").write_text("asset,count\nDoor,2\n", encoding="utf-8")
    (root / "Thumbs.db").write_text("ignored", encoding="utf-8")
    _xlsx(root / "schedule.xlsx")
    _zip(root / "archive.zip")
    return root


def _empty_case_dir(name: str) -> Path:
    root = Path("output/test_source_ingestion") / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _xlsx(path: Path) -> None:
    workbook = Workbook()
    workbook.active.title = "Assets"
    workbook.create_sheet("Maintenance")
    workbook.save(path)


def _zip(path: Path) -> None:
    inner = path.parent / "_inner.zip"
    with zipfile.ZipFile(inner, "w") as handle:
        handle.writestr("inner.txt", "inner")
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("inside/manual.pdf", b"%PDF-1.4\n/Type /Page\n")
        handle.write(inner, "nested/inner.zip")
    inner.unlink()
