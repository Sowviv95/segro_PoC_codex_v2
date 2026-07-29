"""Conservative text normalization for parsed pages and sheets."""

import re

NORMALIZATION_VERSION = "text-normalization-v1"


def normalize_text(text: str) -> str:
    """Normalize extraction artifacts without semantic rewriting."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{4,}", "\n\n\n", normalized)
    return normalized.strip()


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True
