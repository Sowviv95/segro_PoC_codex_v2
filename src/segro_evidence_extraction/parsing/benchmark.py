"""Bounded Stage A parser benchmark runner."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import psutil
from pydantic import Field

from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry
from segro_evidence_extraction.parsing.interfaces import DocumentParser
from segro_evidence_extraction.parsing.models import (
    DocumentTiming,
    ParsedPage,
    ParseOptions,
    ParserWarning,
    ParsingConfig,
)
from segro_evidence_extraction.parsing.pdf import (
    PdfPageParser,
    PyMuPdfPageParser,
    bounded_page_numbers,
)
from segro_evidence_extraction.parsing.service import ParsingError, load_source_registry

BENCHMARK_VERSION = "parser-foundation-stage-a-v1"
DEFAULT_REPETITIONS = 3
PARSER_NAMES = ("pypdf", "pymupdf")


class BenchmarkSample(StrictBaseModel):
    identifier: str
    path: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    anchors: list[str] = Field(default_factory=list)
    source_id: str | None = None
    logical_path: str | None = None


class BenchmarkPageResult(StrictBaseModel):
    page_number: int
    success: bool
    parse_time_ms: float = Field(ge=0)
    character_count: int = Field(ge=0)
    non_whitespace_character_count: int = Field(ge=0)
    alphabetic_ratio: float = Field(ge=0, le=1)
    control_or_replacement_character_count: int = Field(ge=0)
    anchors_present: dict[str, bool]


class BenchmarkRunResult(StrictBaseModel):
    parser_name: str
    parser_version: str
    sample_id: str
    source_path: str
    requested_page_range: str
    repetition: int
    wall_time_ms: float = Field(ge=0)
    success: bool
    failure: bool
    timeout: bool
    exception_type: str | None = None
    exception_message: str | None = None
    worker_exit_code: int | None = None
    peak_worker_rss_bytes: int | None = Field(default=None, ge=0)
    remaining_active_child_count_after_cleanup: int = Field(ge=0)
    pages: list[BenchmarkPageResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BenchmarkResult(StrictBaseModel):
    benchmark_version: str
    repetitions: int
    timeout_seconds: float = Field(gt=0)
    parsers: list[str]
    samples: list[BenchmarkSample]
    runs: list[BenchmarkRunResult]
    recommendation: str
    reused_spike_concepts: list[str]
    artifact_paths: dict[str, str]


class _WorkerEnvelope(StrictBaseModel):
    parser_name: str
    parser_version: str
    pages: list[BenchmarkPageResult]
    warnings: list[str]
    exception_type: str | None = None
    exception_message: str | None = None


def run_parser_benchmark(
    *,
    source_manifest: Path,
    output_dir: Path,
    sample_specs: list[str] | None = None,
    repetitions: int = DEFAULT_REPETITIONS,
    timeout_seconds: float = 60.0,
) -> BenchmarkResult:
    if repetitions < 1:
        raise ParsingError("Benchmark repetitions must be at least 1.")
    samples = (
        _samples_from_specs(sample_specs) if sample_specs else _default_samples(source_manifest)
    )
    for sample in samples:
        _validate_sample(sample)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[BenchmarkRunResult] = []
    for sample in samples:
        for parser_name in PARSER_NAMES:
            for repetition in range(1, repetitions + 1):
                runs.append(
                    _run_isolated_repetition(
                        parser_name=parser_name,
                        sample=sample,
                        repetition=repetition,
                        timeout_seconds=timeout_seconds,
                    )
                )
    recommendation = _recommend(runs)
    result = BenchmarkResult(
        benchmark_version=BENCHMARK_VERSION,
        repetitions=repetitions,
        timeout_seconds=timeout_seconds,
        parsers=list(PARSER_NAMES),
        samples=samples,
        runs=sorted(
            runs,
            key=lambda run: (run.sample_id, run.parser_name, run.repetition),
        ),
        recommendation=recommendation,
        reused_spike_concepts=[
            "Provider-neutral ParsedPage/PageTiming-style result models were retained.",
            "Explicit page-range concepts and progress callback boundaries were retained.",
            "Windows-safe spawned child entry points use importable top-level functions.",
            "Rejected: one spawned process per page, resume/checkpoint logic, "
            "and worker coordinator.",
        ],
        artifact_paths={},
    )
    artifact_paths = _write_artifacts(result, output_dir)
    result.artifact_paths = artifact_paths
    return result


def _run_isolated_repetition(
    *,
    parser_name: str,
    sample: BenchmarkSample,
    repetition: int,
    timeout_seconds: float,
    worker_target: Any | None = None,
) -> BenchmarkRunResult:
    context = multiprocessing.get_context("spawn")
    temp_dir = Path(_benchmark_temp_root()) / f"run_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True)
    try:
        result_path = Path(temp_dir) / "result.json"
        result_path.write_text("", encoding="utf-8")
        process = context.Process(
            target=worker_target or _benchmark_worker_entry,
            args=(parser_name, sample.model_dump(), str(result_path)),
        )
        started = time.perf_counter()
        process.start()
        peak_rss = _rss(process.pid)
        timed_out = False
        try:
            while process.is_alive():
                process.join(timeout=0.05)
                peak_rss = max(peak_rss or 0, _rss(process.pid) or 0)
                if time.perf_counter() - started > timeout_seconds:
                    timed_out = True
                    _terminate_process(process)
                    break
            process.join(timeout=1)
            wall_time_ms = (time.perf_counter() - started) * 1000
            envelope = _read_worker_envelope(result_path) if result_path.exists() else None
        finally:
            if process.is_alive():
                _terminate_process(process)
            process.join(timeout=1)
            exit_code = process.exitcode
            process.close()
        active_children = len(multiprocessing.active_children())
        exception_type = None
        exception_message = None
        pages: list[BenchmarkPageResult] = []
        warnings: list[str] = []
        parser_version = "unknown"
        if envelope is not None:
            parser_version = envelope.parser_version
            pages = envelope.pages
            warnings = envelope.warnings
            exception_type = envelope.exception_type
            exception_message = envelope.exception_message
        elif timed_out:
            exception_type = "TimeoutError"
            exception_message = f"Benchmark repetition exceeded {timeout_seconds:.2f}s."
        else:
            exception_type = "WorkerResultMissing"
            exception_message = "Worker exited without writing benchmark results."
        expected_pages = len(bounded_page_numbers(sample.page_start, sample.page_end))
        success = (
            not timed_out
            and exception_type is None
            and len(pages) == expected_pages
            and all(page.success for page in pages)
        )
        return BenchmarkRunResult(
            parser_name=parser_name,
            parser_version=parser_version,
            sample_id=sample.identifier,
            source_path=sample.path,
            requested_page_range=f"{sample.page_start}-{sample.page_end}",
            repetition=repetition,
            wall_time_ms=wall_time_ms,
            success=success,
            failure=not success,
            timeout=timed_out,
            exception_type=exception_type,
            exception_message=exception_message,
            worker_exit_code=exit_code,
            peak_worker_rss_bytes=peak_rss,
            remaining_active_child_count_after_cleanup=active_children,
            pages=pages,
            warnings=warnings,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _benchmark_worker_entry(
    parser_name: str,
    sample_payload: dict[str, object],
    result_path: str,
) -> None:
    sample = BenchmarkSample.model_validate(sample_payload)
    parser = _build_parser(parser_name)
    source = _source_for_sample(sample)
    pages: list[BenchmarkPageResult] = []
    warnings: list[str] = []
    exception_type = None
    exception_message = None
    try:
        for item in parser.parse(
            source,
            source_path=Path(sample.path),
            config=ParsingConfig(),
            options=ParseOptions(
                page_start=sample.page_start,
                page_end=sample.page_end,
                use_cache=False,
                resume=False,
            ),
            cache=None,
            progress=None,
        ):
            if isinstance(item, ParsedPage):
                pages.append(_page_result(item, sample.anchors))
            elif isinstance(item, ParserWarning):
                warnings.append(f"{item.code}: {item.message}")
            elif isinstance(item, DocumentTiming):
                continue
    except BaseException as exc:  # noqa: BLE001 - child reports cleanly to parent
        exception_type = type(exc).__name__
        exception_message = str(exc)
    envelope = _WorkerEnvelope(
        parser_name=parser.parser_name,
        parser_version=parser.parser_version,
        pages=pages,
        warnings=warnings,
        exception_type=exception_type,
        exception_message=exception_message,
    )
    Path(result_path).write_text(envelope.model_dump_json(), encoding="utf-8")


def _build_parser(parser_name: str) -> DocumentParser:
    if parser_name == "pypdf":
        return PdfPageParser()
    if parser_name == "pymupdf":
        return PyMuPdfPageParser()
    raise ValueError(f"Unknown benchmark parser: {parser_name}")


def _page_result(page: ParsedPage, anchors: list[str]) -> BenchmarkPageResult:
    text = page.text or ""
    return BenchmarkPageResult(
        page_number=page.page_number,
        success=True,
        parse_time_ms=page.parsing_duration_ms,
        character_count=len(text),
        non_whitespace_character_count=sum(1 for char in text if not char.isspace()),
        alphabetic_ratio=page.quality.alphabetic_ratio,
        control_or_replacement_character_count=sum(
            1 for char in text if char == "\ufffd" or (ord(char) < 32 and char not in "\n\t")
        ),
        anchors_present={anchor: anchor.lower() in text.lower() for anchor in anchors},
    )


def _default_samples(source_manifest: Path) -> list[BenchmarkSample]:
    sources = load_source_registry(source_manifest)
    by_name = {Path(source.logical_path).name.lower(): source for source in sources}
    return [
        _manifest_sample(
            by_name,
            "Rolec EV Charger Pre-Commissioning Information Sheet - Enfield U1.pdf",
            "rolec_pre_commissioning_all_pages",
            1,
            1,
            ["Rolec", "Pre-Commissioning"],
        ),
        _manifest_sample(
            by_name,
            "Building Manual - Part 5 The Health & Safety File.pdf",
            "building_manual_part5_pages_001_010",
            1,
            10,
            ["HEALTH AND SAFETY FILE", "REMAINING IDENTIFIED"],
        ),
        _manifest_sample(
            by_name,
            "Building Manual - Part 2 Building Fabric.pdf",
            "building_manual_part2_pages_001_005",
            1,
            5,
            ["BUILDING FABRIC", "INTRODUCTION"],
        ),
        _manifest_sample(
            by_name,
            "Building Manual - Part 1 General.pdf",
            "building_manual_part1_drawing_page_007",
            7,
            7,
            ["Unit 1", "Landscape"],
        ),
        _manifest_sample(
            by_name,
            "Building Manual - Part 1 General.pdf",
            "building_manual_part1_narrative_page_020",
            20,
            20,
            ["bird-nesting", "qualified ecologist"],
        ),
    ]


def _manifest_sample(
    sources_by_name: dict[str, SourceRegistryEntry],
    logical_name: str,
    identifier: str,
    page_start: int,
    page_end: int,
    anchors: list[str],
) -> BenchmarkSample:
    source = sources_by_name.get(logical_name.lower())
    if source is None:
        raise ParsingError(f"Required benchmark source not found in manifest: {logical_name}")
    return BenchmarkSample(
        identifier=identifier,
        path=source.original_path,
        page_start=page_start,
        page_end=page_end,
        anchors=anchors,
        source_id=source.source_id,
        logical_path=source.logical_path,
    )


def _samples_from_specs(sample_specs: list[str]) -> list[BenchmarkSample]:
    return [_sample_from_spec(spec) for spec in sample_specs]


def _sample_from_spec(spec: str) -> BenchmarkSample:
    try:
        identifier, remainder = spec.split("=", 1)
    except ValueError as exc:
        raise ParsingError(
            "Sample must be formatted as identifier=path:start-end[:anchor|anchor]."
        ) from exc
    match = re.match(r"^(?P<path>.+):(?P<start>\d+)-(?P<end>\d+)(?::(?P<anchors>.*))?$", remainder)
    if match is None:
        raise ParsingError(
            "Sample must be formatted as identifier=path:start-end[:anchor|anchor]."
        )
    path_text = match.group("path")
    start_text = match.group("start")
    end_text = match.group("end")
    anchor_text = match.group("anchors") or ""
    anchors = anchor_text.split("|") if anchor_text else []
    return BenchmarkSample(
        identifier=identifier,
        path=path_text,
        page_start=int(start_text),
        page_end=int(end_text),
        anchors=[anchor for anchor in anchors if anchor],
        logical_path=Path(path_text).name,
    )


def _validate_sample(sample: BenchmarkSample) -> None:
    _ = bounded_page_numbers(sample.page_start, sample.page_end)
    path = Path(sample.path)
    if not path.exists():
        raise ParsingError(f"Benchmark sample PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ParsingError(f"Benchmark sample is not a PDF: {path}")


def _source_for_sample(sample: BenchmarkSample) -> SourceRegistryEntry:
    path = Path(sample.path)
    return SourceRegistryEntry(
        source_id=sample.source_id or f"bench_{_short_hash(str(path))}",
        original_path=str(path),
        logical_path=sample.logical_path or path.name,
        file_type=FileType.PDF,
        extension=path.suffix,
        mime_type="application/pdf",
        file_hash=_short_hash(str(path)),
        size_bytes=path.stat().st_size,
        content_identity=f"sha256:{_short_hash(str(path))}",
    )


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_worker_envelope(path: Path) -> _WorkerEnvelope | None:
    try:
        return _WorkerEnvelope.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _benchmark_temp_root() -> str:
    root = Path(".pytest_tmp/parser_foundation_stage_a_benchmark")
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def _rss(pid: int | None) -> int | None:
    if pid is None:
        return None
    try:
        return int(psutil.Process(pid).memory_info().rss)
    except psutil.Error:
        return None


def _terminate_process(process: Any) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)


def _recommend(runs: list[BenchmarkRunResult]) -> str:
    summaries = {parser: _parser_summary(runs, parser) for parser in PARSER_NAMES}
    stable = sorted(
        summaries.items(),
        key=lambda item: (
            -item[1]["successes"],
            item[1]["timeouts"],
            -item[1]["anchor_hits"],
            item[1]["median_wall_ms"],
        ),
    )
    winner = stable[0][0]
    other = stable[1][0]
    return (
        f"Recommend {winner} for the next bounded parser foundation stage. "
        f"It recorded {summaries[winner]['successes']} successful runs versus "
        f"{summaries[other]['successes']} for {other}, with "
        f"{summaries[winner]['timeouts']} timeouts, "
        f"{summaries[winner]['anchor_hits']} anchor hits, and median wall time "
        f"{summaries[winner]['median_wall_ms']:.2f} ms. This recommendation weights "
        "stability, clean failures, text usefulness, and performance rather than speed alone."
    )


def _parser_summary(runs: list[BenchmarkRunResult], parser_name: str) -> dict[str, float]:
    selected = [run for run in runs if run.parser_name == parser_name]
    wall_times = sorted(run.wall_time_ms for run in selected)
    median_wall_ms = wall_times[len(wall_times) // 2] if wall_times else 0.0
    return {
        "successes": float(sum(1 for run in selected if run.success)),
        "timeouts": float(sum(1 for run in selected if run.timeout)),
        "anchor_hits": float(
            sum(
                1
                for run in selected
                for page in run.pages
                for present in page.anchors_present.values()
                if present
            )
        ),
        "median_wall_ms": median_wall_ms,
    }


def _write_artifacts(result: BenchmarkResult, output_dir: Path) -> dict[str, str]:
    detailed_path = output_dir / "parser_benchmark_results.json"
    markdown_path = output_dir / "parser_benchmark_comparison.md"
    result_for_json = result.model_copy(update={"artifact_paths": {}})
    detailed_path.write_text(
        json.dumps(result_for_json.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown_report(result), encoding="utf-8")
    return {
        "detailed_json": str(detailed_path),
        "comparison_markdown": str(markdown_path),
    }


def _markdown_report(result: BenchmarkResult) -> str:
    lines = [
        "# Parser Foundation Reset Stage A Benchmark",
        "",
        f"Benchmark version: `{result.benchmark_version}`",
        f"Repetitions: {result.repetitions}",
        f"Safety timeout: {result.timeout_seconds:.2f}s",
        "",
        "## Samples",
        "",
        "| Sample | Path | Pages | Anchors |",
        "|---|---|---:|---|",
    ]
    for sample in result.samples:
        lines.append(
            f"| {sample.identifier} | `{sample.path}` | "
            f"{sample.page_start}-{sample.page_end} | {', '.join(sample.anchors)} |"
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Parser | Sample | Successes | Timeouts | Wall ms | Chars | Anchors |",
            "|---|---|---:|---:|---|---:|---:|",
        ]
    )
    for parser_name in PARSER_NAMES:
        for sample in result.samples:
            selected = [
                run
                for run in result.runs
                if run.parser_name == parser_name and run.sample_id == sample.identifier
            ]
            wall = ", ".join(f"{run.wall_time_ms:.1f}" for run in selected)
            chars = sum(page.character_count for run in selected for page in run.pages)
            anchors = sum(
                1
                for run in selected
                for page in run.pages
                for present in page.anchors_present.values()
                if present
            )
            lines.append(
                f"| {parser_name} | {sample.identifier} | "
                f"{sum(1 for run in selected if run.success)}/{len(selected)} | "
                f"{sum(1 for run in selected if run.timeout)} | {wall} | {chars} | {anchors} |"
            )
    lines.extend(["", "## Recommendation", "", result.recommendation, "", "## Reused Concepts", ""])
    lines.extend(f"- {concept}" for concept in result.reused_spike_concepts)
    return "\n".join(lines) + "\n"
