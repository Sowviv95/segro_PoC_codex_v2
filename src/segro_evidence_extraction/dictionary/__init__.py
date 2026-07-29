"""Dictionary ingestion and target normalization."""

from segro_evidence_extraction.dictionary.models import (
    DictionaryIngestionResult,
    DictionaryInspectReport,
    RawDictionaryRow,
    ValidationIssue,
)
from segro_evidence_extraction.dictionary.service import ingest_dictionary, inspect_dictionary

__all__ = [
    "DictionaryIngestionResult",
    "DictionaryInspectReport",
    "RawDictionaryRow",
    "ValidationIssue",
    "ingest_dictionary",
    "inspect_dictionary",
]
