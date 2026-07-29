"""CLI skeleton for the evidence-first extraction engine."""

from pathlib import Path
from typing import Annotated

import typer

from segro_evidence_extraction.config import load_settings

UNIMPLEMENTED_EXIT_CODE = 3

app = typer.Typer(
    help="SEGRO evidence-first extraction foundation. Extraction is not implemented yet.",
    no_args_is_help=True,
)


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
        "Clean evidence-first extraction is not implemented in Foundation Sprint 1.",
        err=True,
    )
    raise typer.Exit(UNIMPLEMENTED_EXIT_CODE)
