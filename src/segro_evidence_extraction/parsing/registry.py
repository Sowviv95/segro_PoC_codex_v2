"""Parser selection for registered source records."""

from segro_evidence_extraction.models.source import FileType, SourceRegistryEntry
from segro_evidence_extraction.parsing.images import ImageMetadataParser
from segro_evidence_extraction.parsing.interfaces import DocumentParser
from segro_evidence_extraction.parsing.pdf import PdfPageParser
from segro_evidence_extraction.parsing.spreadsheets import CsvSheetParser, XlsxSheetParser
from segro_evidence_extraction.parsing.text import TextDocumentParser


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers: dict[FileType, DocumentParser] = {
            FileType.PDF: PdfPageParser(),
            FileType.XLSX: XlsxSheetParser(),
            FileType.CSV: CsvSheetParser(),
            FileType.TEXT: TextDocumentParser(),
            FileType.JSON: TextDocumentParser(),
            FileType.IMAGE: ImageMetadataParser(),
        }

    def get(self, source: SourceRegistryEntry) -> DocumentParser | None:
        return self._parsers.get(source.file_type)


def default_parser_registry() -> ParserRegistry:
    return ParserRegistry()
