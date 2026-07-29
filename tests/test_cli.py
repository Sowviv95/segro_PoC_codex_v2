from typer.testing import CliRunner

from segro_evidence_extraction.cli import UNIMPLEMENTED_EXIT_CODE, app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "SEGRO evidence-first extraction foundation" in result.output
    assert "extract" in result.output


def test_cli_extract_validates_required_arguments() -> None:
    result = runner.invoke(app, ["extract"])

    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_cli_extract_unimplemented_exit() -> None:
    result = runner.invoke(
        app,
        [
            "extract",
            "--dictionary-path",
            ".env.example",
            "--asset-config",
            "configs/default.yaml",
            "--output-dir",
            "output/cli-unimplemented-test",
        ],
    )

    assert result.exit_code == UNIMPLEMENTED_EXIT_CODE
    assert "not implemented" in result.output
