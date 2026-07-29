import csv
import json
from pathlib import Path

import pytest
from openpyxl import Workbook
from typer.testing import CliRunner

from segro_evidence_extraction.cli import (
    DICTIONARY_REJECTED_ROWS_EXIT_CODE,
    DICTIONARY_VALIDATION_ERROR_EXIT_CODE,
    app,
)
from segro_evidence_extraction.dictionary.column_mapping import (
    build_column_mapping,
    normalize_header,
)
from segro_evidence_extraction.dictionary.normalizer import stable_target_row_id
from segro_evidence_extraction.dictionary.readers import MissingWorksheetError, read_dictionary
from segro_evidence_extraction.dictionary.service import ingest_dictionary, inspect_dictionary

runner = CliRunner()


def test_xlsx_reader_preserves_physical_rows() -> None:
    workspace = _case_dir("xlsx_reader")
    workbook_path = _xlsx_fixture(workspace)

    raw = read_dictionary(workbook_path, sheet_name="Extraction Template", header_row=1)

    assert raw.metadata.sheet_names == ["Extraction Template"]
    assert raw.rows[0].physical_row_number == 2
    assert raw.rows[0].values["Field Name"] == "door_count"


def test_csv_reader_preserves_physical_rows() -> None:
    workspace = _case_dir("csv_reader")
    csv_path = _csv_fixture(workspace)

    raw = read_dictionary(csv_path, sheet_name=None, header_row=1)

    assert raw.metadata.format == "csv"
    assert raw.rows[0].physical_row_number == 2
    assert raw.rows[1].is_blank


def test_missing_worksheet_error() -> None:
    workspace = _case_dir("missing_worksheet")
    workbook_path = _xlsx_fixture(workspace)

    with pytest.raises(MissingWorksheetError):
        read_dictionary(workbook_path, sheet_name="Missing", header_row=1)


def test_header_normalization_and_alias_precedence() -> None:
    assert normalize_header(" Data_Requirement  ID ") == "data requirement id"
    mapping = build_column_mapping(
        {
            "mapping_version": "test",
            "dictionary_id": "dict",
            "required_source_columns": ["Field Name"],
            "columns": {"expected_field": {"aliases": ["missing", "Field Name"]}},
        },
        ["Field Name", "Unused"],
    )

    assert mapping.field_to_source["expected_field"] == "Field Name"
    assert mapping.unused_columns == ["Unused"]


def test_required_columns_are_validated() -> None:
    with pytest.raises(ValueError, match="Missing required source columns"):
        build_column_mapping(
            {
                "mapping_version": "test",
                "dictionary_id": "dict",
                "required_source_columns": ["Field Name"],
                "columns": {},
            },
            ["Other"],
        )


def test_stable_ids_are_deterministic_and_row_sensitive() -> None:
    first = stable_target_row_id(
        dictionary_id="dict",
        source_name="sheet",
        physical_row=2,
        requirement_id="DR1",
        expected_field="door_count",
    )
    second = stable_target_row_id(
        dictionary_id="dict",
        source_name="sheet",
        physical_row=2,
        requirement_id="DR1",
        expected_field="door_count",
    )
    third = stable_target_row_id(
        dictionary_id="dict",
        source_name="sheet",
        physical_row=3,
        requirement_id="DR1",
        expected_field="door_count",
    )

    assert first == second
    assert first != third


def test_ingestion_accounts_for_blank_warning_and_rejected_rows() -> None:
    workspace = _case_dir("accounting")
    workbook_path = _xlsx_fixture(workspace, include_bad_row=True)
    mapping_path = _mapping_fixture(workspace)
    output_dir = workspace / "out"

    result = ingest_dictionary(
        workbook_path,
        sheet_name="Extraction Template",
        mapping_config=mapping_path,
        output_dir=output_dir,
    )

    assert result.summary.physical_rows == 4
    assert result.summary.blank_rows == 1
    assert result.summary.candidate_rows == 3
    assert result.summary.normalized_rows == 2
    assert result.summary.rows_with_warnings == 1
    assert result.summary.rejected_rows == 1
    assert result.normalized_targets[0].cardinality == "multiple"
    assert result.normalized_targets[0].expected_data_type == "integer"
    assert "Internal" in result.normalized_targets[0].accepted_values
    assert "table" in result.summary.counts_by_likely_evidence_type
    assert (output_dir / "normalized_targets.jsonl").exists()
    assert (output_dir / "dictionary_summary.md").exists()


def test_deterministic_artifact_serialization() -> None:
    workspace = _case_dir("deterministic")
    workbook_path = _xlsx_fixture(workspace)
    mapping_path = _mapping_fixture(workspace)

    first = ingest_dictionary(
        workbook_path,
        sheet_name="Extraction Template",
        mapping_config=mapping_path,
        output_dir=workspace / "first",
    )
    second = ingest_dictionary(
        workbook_path,
        sheet_name="Extraction Template",
        mapping_config=mapping_path,
        output_dir=workspace / "second",
    )

    first_ids = [target.target_row_id for target in first.normalized_targets]
    second_ids = [target.target_row_id for target in second.normalized_targets]
    assert first_ids == second_ids
    assert json.loads((workspace / "first" / "dictionary_summary.json").read_text())[
        "normalized_rows"
    ] == 1


def test_inspect_report() -> None:
    workspace = _case_dir("inspect_report")
    workbook_path = _xlsx_fixture(workspace)

    report = inspect_dictionary(workbook_path, sheet_name="Extraction Template")

    assert report.selected_source == "Extraction Template"
    assert report.physical_rows == 2
    assert report.columns[0] == "Field Name"


def test_cli_dictionary_inspect() -> None:
    workspace = _case_dir("cli_inspect")
    workbook_path = _xlsx_fixture(workspace)

    result = runner.invoke(
        app,
        [
            "dictionary",
            "inspect",
            "--dictionary-path",
            str(workbook_path),
            "--sheet-name",
            "Extraction Template",
        ],
    )

    assert result.exit_code == 0
    assert "physical_rows" in result.output


def test_cli_dictionary_validate_exit_codes() -> None:
    workspace = _case_dir("cli_validate")
    workbook_path = _xlsx_fixture(workspace, include_bad_row=True)
    mapping_path = _mapping_fixture(workspace)

    result = runner.invoke(
        app,
        [
            "dictionary",
            "validate",
            "--dictionary-path",
            str(workbook_path),
            "--sheet-name",
            "Extraction Template",
            "--mapping-config",
            str(mapping_path),
            "--output-dir",
            str(workspace / "out"),
        ],
    )

    assert result.exit_code == DICTIONARY_REJECTED_ROWS_EXIT_CODE
    assert "rejected_rows" in result.output


def test_cli_dictionary_validate_missing_sheet_exit() -> None:
    workspace = _case_dir("cli_missing_sheet")
    workbook_path = _xlsx_fixture(workspace)
    mapping_path = _mapping_fixture(workspace)

    result = runner.invoke(
        app,
        [
            "dictionary",
            "validate",
            "--dictionary-path",
            str(workbook_path),
            "--sheet-name",
            "Missing",
            "--mapping-config",
            str(mapping_path),
        ],
    )

    assert result.exit_code == DICTIONARY_VALIDATION_ERROR_EXIT_CODE


def test_real_workbook_integration_skips_when_unavailable() -> None:
    workbook_path = Path("data/input/data_dictionary/SEGRO_Extraction_Template.xlsx")
    if not workbook_path.exists():
        pytest.skip("Local SEGRO workbook unavailable")

    report = inspect_dictionary(workbook_path, sheet_name="Extraction Template")

    assert report.selected_source == "Extraction Template"
    assert "Field Name" in report.columns


def _xlsx_fixture(tmp_path: Path, *, include_bad_row: bool = False) -> Path:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Extraction Template"
    worksheet.append(_headers())
    worksheet.append(
        [
            "door_count",
            "DR1",
            "Loading Doors",
            "Component",
            "Count",
            "Number of doors",
            "e.g. Internal, External; schedule",
            "Door",
            "Loading Door",
            "Y",
            None,
        ]
    )
    worksheet.append([None] * len(_headers()))
    if include_bad_row:
        worksheet.append(
            [
                "door_manufacturer",
                None,
                "Loading Doors",
                "Component",
                "Manufacturer",
                "Door manufacturer",
                None,
                "Door",
                "Loading Door",
                None,
                None,
            ]
        )
        worksheet.append(
            [
                None,
                None,
                None,
                "Component",
                "Manufacturer",
                None,
                None,
                "Door",
                "Loading Door",
                "maybe",
                None,
            ]
        )
    path = tmp_path / "dictionary.xlsx"
    workbook.save(path)
    return path


def _csv_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "dictionary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_headers())
        writer.writerow(["field", "DR1", "Term", "Domain", "Label", "", "", "", "", "N", ""])
        writer.writerow([None] * len(_headers()))
    return path


def _mapping_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "mapping.yaml"
    path.write_text(Path("configs/dictionaries/segro_extraction_template_v1.yaml").read_text())
    return path


def _case_dir(name: str) -> Path:
    path = Path("output/test_dictionary_ingestion") / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _headers() -> list[str]:
    return [
        "Field Name",
        "Data Requirement id",
        "Term Name",
        "Sub-Domain",
        "Field Label",
        "Business Definition",
        "Notes",
        "Component Type Mapping (comes from component list)",
        "Sub-Type Mapping (comes from component sub-type list)",
        "Multiple Records Possible (Y/N)",
        "Example",
    ]
