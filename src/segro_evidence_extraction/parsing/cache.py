"""JSON page/sheet cache for resumable parsing."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from segro_evidence_extraction.parsing.models import ParsedPage, ParsedSheet, ParsingConfig


def config_hash(config: ParsingConfig) -> str:
    payload = config.model_dump_json(exclude_none=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_key(
    *,
    source_id: str,
    content_hash: str,
    page_or_sheet: str,
    parser_name: str,
    parser_version: str,
    normalization_version: str,
    classification_version: str,
    config_digest: str,
) -> str:
    payload = "|".join(
        [
            source_id,
            content_hash,
            page_or_sheet,
            parser_name,
            parser_version,
            normalization_version,
            classification_version,
            config_digest,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JsonParseCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.read_ms = 0.0
        self.write_ms = 0.0
        self.hits = 0
        self.misses = 0
        self.corrupt = 0

    def read_page(self, key: str) -> ParsedPage | None:
        path = self._path(key)
        start = time.perf_counter()
        try:
            if not path.exists():
                self.misses += 1
                return None
            value = ParsedPage.model_validate_json(path.read_text(encoding="utf-8"))
            self.hits += 1
            return value
        except (OSError, ValidationError, ValueError, json.JSONDecodeError):
            self.corrupt += 1
            self.misses += 1
            return None
        finally:
            self.read_ms += (time.perf_counter() - start) * 1000

    def write_page(self, key: str, page: ParsedPage) -> None:
        self._write_json(key, page.model_dump(mode="json"))

    def read_sheet(self, key: str) -> ParsedSheet | None:
        path = self._path(key)
        start = time.perf_counter()
        try:
            if not path.exists():
                self.misses += 1
                return None
            value = ParsedSheet.model_validate_json(path.read_text(encoding="utf-8"))
            self.hits += 1
            return value
        except (OSError, ValidationError, ValueError, json.JSONDecodeError):
            self.corrupt += 1
            self.misses += 1
            return None
        finally:
            self.read_ms += (time.perf_counter() - start) * 1000

    def write_sheet(self, key: str, sheet: ParsedSheet) -> None:
        self._write_json(key, sheet.model_dump(mode="json"))

    def _write_json(self, key: str, payload: dict[str, Any]) -> None:
        start = time.perf_counter()
        try:
            self._path(key).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        finally:
            self.write_ms += (time.perf_counter() - start) * 1000

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"
