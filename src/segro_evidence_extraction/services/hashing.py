"""Hashing utilities for deterministic source identifiers."""

from hashlib import sha256
from pathlib import Path

from segro_evidence_extraction.models.source import SourceHasher


class Sha256SourceHasher(SourceHasher):
    """Streaming SHA-256 source hasher."""

    def hash_path(self, path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


def deterministic_source_id(logical_path: str, file_hash: str) -> str:
    payload = f"{logical_path}\n{file_hash}".encode()
    return f"src_{sha256(payload).hexdigest()[:16]}"
