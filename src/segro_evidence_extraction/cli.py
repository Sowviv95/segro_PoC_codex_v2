"""CLI skeleton for the evidence-first extraction engine."""

from pathlib import Path
from typing import Annotated

import typer

from segro_evidence_extraction.config import load_settings
from segro_evidence_extraction.dictionary.column_mapping import ColumnMappingError
from segro_evidence_extraction.dictionary.readers import DictionaryReadError
from segro_evidence_extraction.dictionary.service import ingest_dictionary, inspect_dictionary

UNIMPLEMENTED_EXIT_CODE = 3
DICTIONARY_VALIDATION_ERROR_EXIT_CODE = 2
DICTIONARY_REJECTED_ROWS_EXIT_CODE = 4

app = typer.Typer(
    help="SEGRO evidence-first extraction foundation. Extraction is not implemented yet.",
    no_args_is_help=True,
)
dictionary_app = typer.Typer(help="Dictionary inspection and target normalization commands.")
app.add_typer(dictionary_app, name="dictionary")


@app.command()
def config_check(
    config_path: Annotated[
        Path,
        typer.Option("--config-path", help="YAML configuration file to validate."),
    ] = Path("configs/default.yaml"),
) -> None:
    """Validate and display non-secret runtime configuration."""

    settings = load_settings(config_path)
    typer.echo(settings.model_dump_json(exclude={"openai_api_key"}, indent=2))


@app.command()
def extract(
    dictionary_path: Annotated[
        Path,
        typer.Option("--dictionary-path", exists=True, readable=True),
    ],
    asset_config: Annotated[
        Path,
        typer.Option("--asset-config", exists=True, readable=True),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
) -> None:
    """Validate command arguments, then stop because extraction is not implemented."""

    _ = (dictionary_path, asset_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(
        "Clean evidence-first extraction is not implemented yet.",
        err=True,
    )
    raise typer.Exit(UNIMPLEMENTED_EXIT_CODE)


@dictionary_app.command("inspect")
def dictionary_inspect(
    dictionary_path: Annotated[
        Path,
        typer.Option("--dictionary-path", exists=True, readable=True),
    ],
    sheet_name: Annotated[str | None, typer.Option("--sheet-name")] = None,
    header_row: Annotated[int, typer.Option("--header-row", min=1)] = 1,
) -> None:
    """Inspect workbook/CSV structure without normalizing targets."""

    try:
        report = inspect_dictionary(dictionary_path, sheet_name=sheet_name, header_row=header_row)
    except DictionaryReadError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(DICTIONARY_VALIDATION_ERROR_EXIT_CODE) from exc
    typer.echo(report.model_dump_json(indent=2))


@dictionary_app.command("validate")
def dictionary_validate(
    dictionary_path: Annotated[
        Path,
        typer.Option("--dictionary-path", exists=True, readable=True),
    ],
    sheet_name: Annotated[str | None, typer.Option("--sheet-name")] = None,
    mapping_config: Annotated[
        Path,
        typer.Option("--mapping-config", exists=True, readable=True),
    ] = Path("configs/dictionaries/segro_extraction_template_v1.yaml"),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path(
        "output/sprint2_dictionary_validation"
    ),
) -> None:
    """Validate and normalize a dictionary into target specifications."""

    try:
        result = ingest_dictionary(
            dictionary_path,
            sheet_name=sheet_name,
            mapping_config=mapping_config,
            output_dir=output_dir,
        )
    except (DictionaryReadError, ColumnMappingError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(DICTIONARY_VALIDATION_ERROR_EXIT_CODE) from exc
    typer.echo(result.summary.model_dump_json(indent=2))
    if result.summary.rejected_rows > 0 or result.summary.duplicate_ids > 0:
        raise typer.Exit(DICTIONARY_REJECTED_ROWS_EXIT_CODE)
