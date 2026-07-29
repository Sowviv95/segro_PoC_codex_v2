"""Streaming source hashing and deterministic source identity."""

from hashlib import sha256
from pathlib import Path
from time import perf_counter

from segro_evidence_extraction.source_ingestion.models import HashResult


def hash_file(path: Path) -> HashResult:
    digest = sha256()
    bytes_read = 0
    start = perf_counter()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                bytes_read += len(block)
                digest.update(block)
    except OSError as exc:
        return HashResult(
            content_hash="",
            bytes_read=bytes_read,
            duration_ms=(perf_counter() - start) * 1000,
            failed=True,
            error=str(exc),
        )
    return HashResult(
        content_hash=digest.hexdigest(),
        bytes_read=bytes_read,
        duration_ms=(perf_counter() - start) * 1000,
    )


def source_instance_id(
    *,
    content_hash: str,
    logical_path: str,
    parent_archive_source_id: str | None = None,
    archive_member_path: str | None = None,
) -> str:
    payload = "|".join(
        [
            content_hash,
            logical_path,
            parent_archive_source_id or "",
            archive_member_path or "",
        ]
    )
    return f"src_{sha256(payload.encode()).hexdigest()[:16]}"


def content_identity(content_hash: str) -> str:
    return f"sha256:{content_hash}"
