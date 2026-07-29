"""Parser interfaces and progress callback contracts."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

from segro_evidence_extraction.models.source import SourceRegistryEntry
from segro_evidence_extraction.parsing.cache import JsonParseCache
from segro_evidence_extraction.parsing.models import (
    DocumentTiming,
    ImageRegion,
    ParsedPage,
    ParsedSheet,
    ParseOptions,
    ParserWarning,
    ParsingConfig,
)

ProgressCallback = Callable[[str, str], None]


class DocumentParser(Protocol):
    parser_name: str
    parser_version: str

    def parse(
        self,
        source: SourceRegistryEntry,
        *,
        source_path: Path,
        config: ParsingConfig,
        options: ParseOptions,
        cache: JsonParseCache | None,
        progress: ProgressCallback | None,
    ) -> Iterator[ParsedPage | ParsedSheet | ImageRegion | ParserWarning | DocumentTiming]:
        """Yield parsed units incrementally with final document timing."""
