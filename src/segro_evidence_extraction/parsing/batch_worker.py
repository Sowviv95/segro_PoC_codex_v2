"""Safe bounded PDF batch worker with resumable checkpoints."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import re
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import psutil
from pydantic import Field, field_validator

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
from segro_evidence_extraction.parsing.pdf import PyMuPdfPageParser, bounded_page_numbers
from segro_evidence_extraction.parsing.service import ParsingError, load_source_registry

BATCH_WORKER_VERSION = "parser-foundation-stage-b-v1"
DEFAULT_MAX_BATCH_PAGES = 10

BatchTerminationReason = Literal[
    "completed",
    "startup_timeout",
    "progress_stall",
    "restart_exhausted",
    "worker_failure",
]


class BatchWorkerConfig(StrictBaseModel):
    stall_threshold_seconds: float = Field(default=5.0, gt=0)
    startup_timeout_seconds: float = Field(default=20.0, gt=0)
    max_restarts: int = Field(default=1, ge=0)
    max_batch_pages: int = Field(default=DEFAULT_MAX_BATCH_PAGES, ge=1, le=DEFAULT_MAX_BATCH_PAGES)
    queue_poll_seconds: float = Field(default=0.05, gt=0)


class BatchRequest(StrictBaseModel):
    source: SourceRegistryEntry
    source_path: str
    output_dir: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    parser_name: str = "pymupdf"

    @field_validator("page_end")
    @classmethod
    def end_must_not_precede_start(cls, value: int, info: object) -> int:
        data = getattr(info, "data", {})
        start = data.get("page_start")
        if isinstance(start, int) and value < start:
            msg = "page_end must be greater than or equal to page_start"
            raise ValueError(msg)
        return value


class BatchPageFailure(StrictBaseModel):
    page_number: int
    exception_type: str
    message: str
    timestamp: str


class BatchPageObservation(StrictBaseModel):
    page_number: int
    parse_duration_ms: float = Field(ge=0)
    character_count: int = Field(ge=0)
    non_whitespace_character_count: int = Field(ge=0)
    timestamp: str


class BatchCheckpoint(StrictBaseModel):
    checkpoint_version: str = BATCH_WORKER_VERSION
    source_id: str
    source_path: str
    parser_name: str
    parser_version: str
    requested_page_start: int
    requested_page_end: int
    completed_pages: list[int] = Field(default_factory=list)
    failed_pages: list[BatchPageFailure] = Field(default_factory=list)
    page_observations: list[BatchPageObservation] = Field(default_factory=list)
    first_incomplete_page: int | None = None
    checkpoint_timestamps: list[str] = Field(default_factory=list)
    restart_count: int = Field(default=0, ge=0)


class BatchProgressEvent(StrictBaseModel):
    event_type: str
    timestamp: str
    page_number: int | None = None
    message: str | None = None
    parser_name: str | None = None
    parser_version: str | None = None
    parse_duration_ms: float | None = Field(default=None, ge=0)
    parsed_page: ParsedPage | None = None


class BatchRunResult(StrictBaseModel):
    batch_worker_version: str = BATCH_WORKER_VERSION
    parser_name: str
    parser_version: str
    source_id: str
    source_path: str
    requested_page_start: int
    requested_page_end: int
    pages_attempted: list[int]
    pages_completed: list[int]
    pages_failed: list[BatchPageFailure]
    per_page_parse_duration_ms: dict[str, float]
    worker_startup_timestamp: str | None
    document_open_completed_timestamp: str | None
    heartbeat_timestamps: dict[str, str]
    stall_threshold_seconds: float
    termination_reason: BatchTerminationReason
    termination_events: list[BatchTerminationReason] = Field(default_factory=list)
    restart_count: int = Field(ge=0)
    checkpoint_count: int = Field(ge=0)
    first_incomplete_page: int | None
    total_wall_time_ms: float = Field(ge=0)
    worker_exit_codes: list[int | None]
    active_child_count_after_cleanup: int = Field(ge=0)
    peak_worker_rss_bytes: int | None = Field(default=None, ge=0)
    checkpoint_path: str
    result_path: str
    progress_events: list[BatchProgressEvent]
    parsed_pages: list[ParsedPage] = Field(default_factory=list)


ParserFactory = Callable[[], DocumentParser]


def run_bounded_batch_worker(
    request: BatchRequest,
    *,
    config: BatchWorkerConfig | None = None,
    parser_factory: ParserFactory | None = None,
) -> BatchRunResult:
    config = config or BatchWorkerConfig()
    requested_pages = bounded_page_numbers(request.page_start, request.page_end)
    if len(requested_pages) > config.max_batch_pages:
        raise ParsingError(
            f"Stage B batch range is {len(requested_pages)} pages; maximum is "
            f"{config.max_batch_pages}."
        )
    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(output_dir, request)
    parsed_pages_path = _parsed_pages_path(output_dir, request)
    result_path = output_dir / "batch_worker_result.json"
    store = _CheckpointStore(checkpoint_path, request)
    checkpoint = store.load_or_initial()
    restart_count = checkpoint.restart_count
    progress_events: list[BatchProgressEvent] = []
    worker_exit_codes: list[int | None] = []
    peak_rss: int | None = None
    worker_startup_timestamp: str | None = None
    document_open_completed_timestamp: str | None = None
    heartbeat_timestamps: dict[str, str] = {}
    termination_reason: BatchTerminationReason = "completed"
    termination_events: list[BatchTerminationReason] = []
    parsed_pages: list[ParsedPage] = []
    started = time.perf_counter()
    first_page = _first_incomplete(requested_pages, checkpoint)

    while first_page is not None:
        context = multiprocessing.get_context("spawn")
        progress_channel = _ProgressChannel.create(context, output_dir)
        process = context.Process(
            target=_batch_worker_entry,
            args=(
                request.model_dump(mode="json"),
                first_page,
                request.page_end,
                str(checkpoint_path),
                str(parsed_pages_path),
                progress_channel.target,
                parser_factory,
            ),
        )
        process.start()
        peak_rss = max(peak_rss or 0, _rss(process.pid) or 0)
        open_seen = False
        last_progress_at: float | None = None
        launch_at = time.perf_counter()
        while process.is_alive():
            event = progress_channel.poll(config.queue_poll_seconds)
            if event is not None:
                progress_events.append(event)
                if event.event_type == "worker_started":
                    worker_startup_timestamp = event.timestamp
                elif event.event_type == "document_opened":
                    open_seen = True
                    document_open_completed_timestamp = event.timestamp
                    last_progress_at = time.perf_counter()
                elif event.event_type in {"page_completed", "page_failed"}:
                    last_progress_at = time.perf_counter()
                    if event.page_number is not None:
                        heartbeat_timestamps[str(event.page_number)] = event.timestamp
                elif event.event_type == "worker_exception":
                    termination_reason = "worker_failure"
                    termination_events.append("worker_failure")
            peak_rss = max(peak_rss or 0, _rss(process.pid) or 0)
            now = time.perf_counter()
            if not open_seen and now - launch_at > config.startup_timeout_seconds:
                termination_reason = "startup_timeout"
                termination_events.append("startup_timeout")
                _terminate_process(process)
                break
            if open_seen and last_progress_at is not None:
                if now - last_progress_at > config.stall_threshold_seconds:
                    termination_reason = "progress_stall"
                    termination_events.append("progress_stall")
                    _terminate_process(process)
                    break
        progress_channel.drain(progress_events, heartbeat_timestamps)
        process.join(timeout=1)
        worker_exit_codes.append(process.exitcode)
        if process.is_alive():
            _terminate_process(process)
        process.close()
        parsed_pages = _load_parsed_pages(parsed_pages_path)
        checkpoint = store.load_or_initial()
        first_page = _first_incomplete(requested_pages, checkpoint)
        if first_page is None:
            termination_reason = "completed"
            break
        if termination_reason not in {"progress_stall", "startup_timeout", "worker_failure"}:
            termination_reason = "worker_failure"
        if restart_count >= config.max_restarts:
            termination_reason = "restart_exhausted"
            termination_events.append("restart_exhausted")
            break
        restart_count += 1
        checkpoint.restart_count = restart_count
        checkpoint.first_incomplete_page = first_page
        store.write(checkpoint)

    checkpoint = store.load_or_initial()
    checkpoint.restart_count = restart_count
    checkpoint.first_incomplete_page = _first_incomplete(requested_pages, checkpoint)
    store.write(checkpoint)
    result = _build_result(
        request=request,
        checkpoint=checkpoint,
        progress_events=progress_events,
        worker_startup_timestamp=worker_startup_timestamp,
        document_open_completed_timestamp=document_open_completed_timestamp,
        heartbeat_timestamps=heartbeat_timestamps,
        stall_threshold_seconds=config.stall_threshold_seconds,
        termination_reason=termination_reason,
        termination_events=termination_events,
        parsed_pages=parsed_pages,
        total_wall_time_ms=(time.perf_counter() - started) * 1000,
        worker_exit_codes=worker_exit_codes,
        active_child_count_after_cleanup=len(multiprocessing.active_children()),
        peak_worker_rss_bytes=peak_rss,
        checkpoint_path=checkpoint_path,
        result_path=result_path,
    )
    _atomic_write_json(result_path, result.model_dump(mode="json"))
    return result


def run_manifest_bounded_batch(
    *,
    source_manifest: Path,
    source_id: str,
    output_dir: Path,
    page_start: int,
    page_end: int,
    config: BatchWorkerConfig | None = None,
) -> BatchRunResult:
    source = _source_by_id(source_manifest, source_id)
    return run_bounded_batch_worker(
        BatchRequest(
            source=source,
            source_path=source.original_path,
            output_dir=str(output_dir),
            page_start=page_start,
            page_end=page_end,
        ),
        config=config,
    )


def _batch_worker_entry(
    request_payload: dict[str, object],
    page_start: int,
    page_end: int,
    checkpoint_path: str,
    parsed_pages_path: str,
    progress_target: Any,
    parser_factory: ParserFactory | None,
) -> None:
    request = BatchRequest.model_validate(request_payload)
    parser = parser_factory() if parser_factory else PyMuPdfPageParser()
    store = _CheckpointStore(Path(checkpoint_path), request)
    parsed_page_store = _ParsedPageStore(Path(parsed_pages_path))
    completed_parsed_pages = parsed_page_store.load()
    _emit(progress_target, "worker_started", parser_name=parser.parser_name)

    def progress(stage: str, subject: str) -> None:
        if stage == "parse:document_opened":
            _emit(
                progress_target,
                "document_opened",
                parser_name=parser.parser_name,
                parser_version=parser.parser_version,
                message=subject,
            )

    try:
        for item in parser.parse(
            request.source,
            source_path=Path(request.source_path),
            config=ParsingConfig(),
            options=ParseOptions(
                page_start=page_start,
                page_end=page_end,
                max_pages=page_end - page_start + 1,
                use_cache=False,
                resume=False,
                progress_every=1,
            ),
            cache=None,
            progress=progress,
        ):
            checkpoint = store.load_or_initial()
            checkpoint.parser_name = parser.parser_name
            checkpoint.parser_version = parser.parser_version
            if isinstance(item, ParsedPage):
                observation = BatchPageObservation(
                    page_number=item.page_number,
                    parse_duration_ms=item.parsing_duration_ms,
                    character_count=item.character_count,
                    non_whitespace_character_count=sum(
                        1 for char in (item.text or "") if not char.isspace()
                    ),
                    timestamp=_now(),
                )
                checkpoint.completed_pages = sorted(
                    {*checkpoint.completed_pages, item.page_number}
                )
                checkpoint.page_observations = [
                    obs
                    for obs in checkpoint.page_observations
                    if obs.page_number != item.page_number
                ]
                checkpoint.page_observations.append(observation)
                checkpoint.first_incomplete_page = _first_incomplete(
                    bounded_page_numbers(request.page_start, request.page_end),
                    checkpoint,
                )
                store.write(checkpoint)
                completed_parsed_pages = [
                    page
                    for page in completed_parsed_pages
                    if page.page_number != item.page_number
                ]
                completed_parsed_pages.append(item)
                parsed_page_store.write(completed_parsed_pages)
                _emit(
                    progress_target,
                    "page_completed",
                    page_number=item.page_number,
                    parse_duration_ms=item.parsing_duration_ms,
                    parser_name=parser.parser_name,
                    parser_version=parser.parser_version,
                )
            elif isinstance(item, ParserWarning):
                failed_page = _warning_page_number(item)
                if failed_page is not None:
                    failure = BatchPageFailure(
                        page_number=failed_page,
                        exception_type=item.code,
                        message=item.message,
                        timestamp=_now(),
                    )
                    checkpoint.failed_pages = [
                        failure_item
                        for failure_item in checkpoint.failed_pages
                        if failure_item.page_number != failed_page
                    ]
                    checkpoint.failed_pages.append(failure)
                    checkpoint.first_incomplete_page = _first_incomplete(
                        bounded_page_numbers(request.page_start, request.page_end),
                        checkpoint,
                    )
                    store.write(checkpoint)
                    _emit(
                        progress_target,
                        "page_failed",
                        page_number=failed_page,
                        message=item.message,
                        parser_name=parser.parser_name,
                        parser_version=parser.parser_version,
                    )
            elif isinstance(item, DocumentTiming):
                continue
    except BaseException as exc:  # noqa: BLE001 - report and let coordinator finish cleanly
        _emit(
            progress_target,
            "worker_exception",
            message=f"{type(exc).__name__}: {exc}",
            parser_name=parser.parser_name,
            parser_version=parser.parser_version,
        )


class _CheckpointStore:
    def __init__(self, path: Path, request: BatchRequest) -> None:
        self.path = path
        self.request = request

    def load_or_initial(self) -> BatchCheckpoint:
        if self.path.exists():
            try:
                return BatchCheckpoint.model_validate_json(
                    self.path.read_text(encoding="utf-8")
                )
            except ValueError:
                pass
        return BatchCheckpoint(
            source_id=self.request.source.source_id,
            source_path=self.request.source_path,
            parser_name=self.request.parser_name,
            parser_version="unknown",
            requested_page_start=self.request.page_start,
            requested_page_end=self.request.page_end,
            first_incomplete_page=self.request.page_start,
        )

    def write(self, checkpoint: BatchCheckpoint) -> None:
        timestamp = _now()
        checkpoint.checkpoint_timestamps.append(timestamp)
        completed_or_failed = set(checkpoint.completed_pages) | {
            failure.page_number for failure in checkpoint.failed_pages
        }
        requested = bounded_page_numbers(
            checkpoint.requested_page_start,
            checkpoint.requested_page_end,
        )
        checkpoint.completed_pages = sorted(set(checkpoint.completed_pages))
        checkpoint.failed_pages = sorted(checkpoint.failed_pages, key=lambda item: item.page_number)
        checkpoint.page_observations = sorted(
            checkpoint.page_observations,
            key=lambda item: item.page_number,
        )
        checkpoint.first_incomplete_page = next(
            (page for page in requested if page not in completed_or_failed),
            None,
        )
        _atomic_write_json(self.path, checkpoint.model_dump(mode="json"))


class _ParsedPageStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[ParsedPage]:
        return _load_parsed_pages(self.path)

    def write(self, pages: list[ParsedPage]) -> None:
        payload: dict[str, object] = {
            "pages": [
                page.model_dump(mode="json")
                for page in sorted(pages, key=lambda item: item.page_number)
            ],
        }
        _atomic_write_json(self.path, payload)


def _source_by_id(source_manifest: Path, source_id: str) -> SourceRegistryEntry:
    for source in load_source_registry(source_manifest):
        if source.source_id == source_id:
            if source.file_type != FileType.PDF:
                raise ParsingError(f"Stage B batch worker only supports PDF sources: {source_id}")
            return source
    raise ParsingError(f"Source ID not found in manifest: {source_id}")


def _build_result(
    *,
    request: BatchRequest,
    checkpoint: BatchCheckpoint,
    progress_events: list[BatchProgressEvent],
    worker_startup_timestamp: str | None,
    document_open_completed_timestamp: str | None,
    heartbeat_timestamps: dict[str, str],
    stall_threshold_seconds: float,
    termination_reason: BatchTerminationReason,
    termination_events: list[BatchTerminationReason],
    parsed_pages: list[ParsedPage],
    total_wall_time_ms: float,
    worker_exit_codes: list[int | None],
    active_child_count_after_cleanup: int,
    peak_worker_rss_bytes: int | None,
    checkpoint_path: Path,
    result_path: Path,
) -> BatchRunResult:
    completed = sorted(checkpoint.completed_pages)
    failed = sorted(checkpoint.failed_pages, key=lambda item: item.page_number)
    attempted = sorted({*completed, *(failure.page_number for failure in failed)})
    return BatchRunResult(
        parser_name=checkpoint.parser_name,
        parser_version=checkpoint.parser_version,
        source_id=request.source.source_id,
        source_path=request.source_path,
        requested_page_start=request.page_start,
        requested_page_end=request.page_end,
        pages_attempted=attempted,
        pages_completed=completed,
        pages_failed=failed,
        per_page_parse_duration_ms={
            str(observation.page_number): observation.parse_duration_ms
            for observation in checkpoint.page_observations
        },
        worker_startup_timestamp=worker_startup_timestamp,
        document_open_completed_timestamp=document_open_completed_timestamp,
        heartbeat_timestamps=heartbeat_timestamps,
        stall_threshold_seconds=stall_threshold_seconds,
        termination_reason=termination_reason,
        termination_events=termination_events,
        restart_count=checkpoint.restart_count,
        checkpoint_count=len(checkpoint.checkpoint_timestamps),
        first_incomplete_page=checkpoint.first_incomplete_page,
        total_wall_time_ms=total_wall_time_ms,
        worker_exit_codes=worker_exit_codes,
        active_child_count_after_cleanup=active_child_count_after_cleanup,
        peak_worker_rss_bytes=peak_worker_rss_bytes,
        checkpoint_path=str(checkpoint_path),
        result_path=str(result_path),
        progress_events=progress_events,
        parsed_pages=sorted(parsed_pages, key=lambda page: page.page_number),
    )


class _ProgressChannel:
    def __init__(self, target: Any, event_path: Path | None = None) -> None:
        self.target = target
        self.event_path = event_path
        self._offset = 0

    @classmethod
    def create(cls, context: Any, output_dir: Path) -> _ProgressChannel:
        try:
            return cls(context.Queue(maxsize=100))
        except (OSError, PermissionError):
            event_path = output_dir / f"batch_worker_events_{uuid.uuid4().hex}.jsonl"
            event_path.parent.mkdir(parents=True, exist_ok=True)
            event_path.write_text("", encoding="utf-8")
            return cls({"kind": "file", "path": str(event_path)}, event_path)

    def poll(self, timeout_seconds: float) -> BatchProgressEvent | None:
        if self.event_path is not None:
            deadline = time.perf_counter() + timeout_seconds
            while time.perf_counter() < deadline:
                event = self._read_file_event()
                if event is not None:
                    return event
                time.sleep(min(0.01, timeout_seconds))
            return self._read_file_event()
        try:
            payload = self.target.get(timeout=timeout_seconds)
        except Exception:  # noqa: BLE001 - queue.Empty differs across contexts
            return None
        return BatchProgressEvent.model_validate(payload)

    def drain(
        self,
        progress_events: list[BatchProgressEvent],
        heartbeat_timestamps: dict[str, str],
    ) -> None:
        while True:
            event = self.poll(0.001)
            if event is None:
                return
            progress_events.append(event)
            if (
                event.event_type in {"page_completed", "page_failed"}
                and event.page_number is not None
            ):
                heartbeat_timestamps[str(event.page_number)] = event.timestamp

    def _read_file_event(self) -> BatchProgressEvent | None:
        if self.event_path is None or not self.event_path.exists():
            return None
        with self.event_path.open("r", encoding="utf-8") as handle:
            handle.seek(self._offset)
            line = handle.readline()
            self._offset = handle.tell()
        if not line:
            return None
        return BatchProgressEvent.model_validate_json(line)


def _emit(
    progress_target: Any,
    event_type: str,
    *,
    page_number: int | None = None,
    message: str | None = None,
    parser_name: str | None = None,
    parser_version: str | None = None,
    parse_duration_ms: float | None = None,
    parsed_page: ParsedPage | None = None,
) -> None:
    event = BatchProgressEvent(
        event_type=event_type,
        timestamp=_now(),
        page_number=page_number,
        message=message,
        parser_name=parser_name,
        parser_version=parser_version,
        parse_duration_ms=parse_duration_ms,
        parsed_page=parsed_page,
    )
    payload = event.model_dump(mode="json")
    if isinstance(progress_target, dict) and progress_target.get("kind") == "file":
        path_value = progress_target.get("path")
        if isinstance(path_value, str):
            with Path(path_value).open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return
    progress_target.put(payload)


def _first_incomplete(
    requested_pages: list[int],
    checkpoint: BatchCheckpoint,
) -> int | None:
    attempted = set(checkpoint.completed_pages) | {
        failure.page_number for failure in checkpoint.failed_pages
    }
    return next((page for page in requested_pages if page not in attempted), None)


def _warning_page_number(warning: ParserWarning) -> int | None:
    if warning.page_or_sheet is None:
        return None
    match = re.match(r"page-(\d+)$", warning.page_or_sheet)
    return int(match.group(1)) if match else None


def _checkpoint_path(output_dir: Path, request: BatchRequest) -> Path:
    safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.source.source_id)
    filename = (
        f"{safe_source}_pages_{request.page_start:04d}_"
        f"{request.page_end:04d}_checkpoint.json"
    )
    return output_dir / filename


def _parsed_pages_path(output_dir: Path, request: BatchRequest) -> Path:
    safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.source.source_id)
    filename = (
        f"{safe_source}_pages_{request.page_start:04d}_"
        f"{request.page_end:04d}_parsed_pages.json"
    )
    return output_dir / filename


def _load_parsed_pages(path: Path) -> list[ParsedPage]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pages = payload.get("pages")
        if not isinstance(pages, list):
            return []
        return sorted(
            [ParsedPage.model_validate(page) for page in pages],
            key=lambda page: page.page_number,
        )
    except (OSError, ValueError, TypeError):
        return []


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


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


def _now() -> str:
    return datetime.now(UTC).isoformat()


def source_id_for_path(path: Path) -> str:
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    return f"src_{digest}"
