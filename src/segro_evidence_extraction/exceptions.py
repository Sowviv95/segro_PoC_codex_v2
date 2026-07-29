"""Package-specific exceptions."""


class SegroEvidenceExtractionError(Exception):
    """Base exception for foundation-level errors."""


class ConfigurationError(SegroEvidenceExtractionError):
    """Raised when runtime configuration cannot be validated."""


class PipelineNotImplementedError(SegroEvidenceExtractionError):
    """Raised by sprint-1 command paths that intentionally do not extract."""
