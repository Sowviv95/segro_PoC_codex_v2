"""Lightweight file type detection."""

from pathlib import Path

from segro_evidence_extraction.models.source import FileType

_IMAGE_SIGNATURES = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
}


def detect_file_type(path: Path) -> tuple[FileType, str | None]:
    extension = path.suffix.lower()
    header = _read_header(path)
    if header.startswith(b"%PDF"):
        return FileType.PDF, "application/pdf"
    if header.startswith(b"PK\x03\x04"):
        if extension == ".xlsx":
            xlsx_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            return FileType.XLSX, xlsx_mime
        return FileType.ZIP, "application/zip"
    if extension == ".zip":
        return FileType.ZIP, "application/zip"
    if header.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return FileType.XLS, "application/vnd.ms-excel"
    for signature, mime_type in _IMAGE_SIGNATURES.items():
        if header.startswith(signature):
            return FileType.IMAGE, mime_type
    if extension == ".csv":
        return FileType.CSV, "text/csv"
    if extension == ".json":
        return FileType.JSON, "application/json"
    if extension in {".txt", ".md", ".log"} or _looks_like_text(header):
        return FileType.TEXT, "text/plain"
    return FileType.UNKNOWN, None


def is_supported_file_type(file_type: FileType) -> bool:
    return file_type in {
        FileType.PDF,
        FileType.XLSX,
        FileType.XLS,
        FileType.CSV,
        FileType.TEXT,
        FileType.JSON,
        FileType.ZIP,
        FileType.IMAGE,
    }


def _read_header(path: Path, size: int = 512) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(size)
    except OSError:
        return b""


def _looks_like_text(header: bytes) -> bool:
    if not header:
        return False
    if b"\x00" in header:
        return False
    try:
        header.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True
