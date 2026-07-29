"""Deterministic page-level text quality signals."""

import re

from segro_evidence_extraction.parsing.models import OcrRouting, TextQualityMetrics


def assess_text_quality(
    text: str,
    *,
    page_area: float | None = None,
    extraction_error: str | None = None,
) -> TextQualityMetrics:
    character_count = len(text)
    words = re.findall(r"\b\w+\b", text)
    lines = [line for line in text.splitlines() if line.strip()]
    alpha = sum(1 for char in text if char.isalpha())
    numeric = sum(1 for char in text if char.isdigit())
    repeated = _repeated_character_count(text)
    density_denominator = page_area if page_area and page_area > 0 else 1.0
    text_density = character_count / density_denominator
    indicators: list[str] = []
    if extraction_error:
        indicators.append(extraction_error)
    if "\ufffd" in text:
        indicators.append("replacement_character")
    if character_count == 0:
        indicators.append("no_extracted_text")
    scan_likelihood = _scan_likelihood(character_count, len(words), alpha, len(text), indicators)
    return TextQualityMetrics(
        character_count=character_count,
        word_count=len(words),
        line_count=len(lines),
        average_line_length=(sum(len(line) for line in lines) / len(lines)) if lines else 0.0,
        alphabetic_ratio=(alpha / character_count) if character_count else 0.0,
        numeric_ratio=(numeric / character_count) if character_count else 0.0,
        repeated_character_ratio=(repeated / character_count) if character_count else 0.0,
        text_density=text_density,
        extraction_error_indicators=indicators,
        likely_text_order_degradation=_likely_text_order_degradation(text, lines),
        scan_likelihood=scan_likelihood,
    )


def route_ocr(quality: TextQualityMetrics, file_type: str) -> tuple[OcrRouting, str, float]:
    if file_type == "image":
        return OcrRouting.RECOMMENDED, "Image sources require OCR for text recovery.", 0.75
    if quality.character_count == 0 and file_type == "pdf":
        return OcrRouting.REQUIRED, "PDF page has no extractable text.", 0.85
    if quality.scan_likelihood >= 0.7 and file_type == "pdf":
        return (
            OcrRouting.RECOMMENDED,
            "Sparse extracted text suggests a scanned or image page.",
            0.7,
        )
    if quality.character_count >= 80:
        return OcrRouting.NOT_REQUIRED, "Page has usable extracted text.", 0.8
    return OcrRouting.UNSUITABLE_UNKNOWN, "Insufficient deterministic evidence to route OCR.", 0.45


def _repeated_character_count(text: str) -> int:
    return sum(len(match.group(0)) for match in re.finditer(r"(.)\1{5,}", text))


def _scan_likelihood(
    character_count: int,
    word_count: int,
    alpha_count: int,
    total_count: int,
    indicators: list[str],
) -> float:
    if "no_extracted_text" in indicators:
        return 0.95
    if total_count == 0:
        return 0.95
    alpha_ratio = alpha_count / total_count
    if character_count < 20:
        return 0.8
    if character_count < 80 and word_count < 10:
        return 0.65
    if alpha_ratio < 0.2 and character_count < 250:
        return 0.55
    return 0.1


def _likely_text_order_degradation(text: str, lines: list[str]) -> bool:
    if not lines:
        return False
    very_short = sum(1 for line in lines if 0 < len(line.strip()) <= 2)
    return very_short / len(lines) > 0.35 and len(text) > 200
