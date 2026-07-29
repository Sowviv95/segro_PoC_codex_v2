"""Duplicate source detection."""

from collections import defaultdict
from hashlib import sha256

from segro_evidence_extraction.models.source import SourceRegistryEntry
from segro_evidence_extraction.source_ingestion.models import DuplicateGroup, DuplicateType


def detect_duplicates(sources: list[SourceRegistryEntry]) -> list[DuplicateGroup]:
    groups: list[DuplicateGroup] = []
    groups.extend(_groups_by_content(sources))
    groups.extend(_groups_by_relative_path(sources))
    groups.extend(_groups_by_archive_member(sources))
    groups.extend(_same_name_different_content(sources))
    return groups


def _groups_by_content(sources: list[SourceRegistryEntry]) -> list[DuplicateGroup]:
    by_hash: dict[str, list[SourceRegistryEntry]] = defaultdict(list)
    for source in sources:
        by_hash[source.file_hash].append(source)
    groups: list[DuplicateGroup] = []
    for content_hash, items in sorted(by_hash.items()):
        if len(items) < 2:
            continue
        filenames = {item.logical_path.rsplit("/", 1)[-1] for item in items}
        duplicate_type = (
            DuplicateType.SAME_CONTENT_DIFFERENT_FILENAMES
            if len(filenames) > 1
            else DuplicateType.IDENTICAL_CONTENT
        )
        groups.append(
            _group(
                duplicate_type,
                items,
                content_hash,
                "Sources have identical SHA-256 content hashes.",
            )
        )
    return groups


def _groups_by_relative_path(sources: list[SourceRegistryEntry]) -> list[DuplicateGroup]:
    by_path: dict[str, list[SourceRegistryEntry]] = defaultdict(list)
    for source in sources:
        by_path[source.logical_path].append(source)
    return [
        _group(
            DuplicateType.DUPLICATE_RELATIVE_PATH,
            items,
            None,
            "Sources share the same logical path.",
        )
        for _, items in sorted(by_path.items())
        if len(items) > 1
    ]


def _groups_by_archive_member(sources: list[SourceRegistryEntry]) -> list[DuplicateGroup]:
    by_member: dict[tuple[str | None, str | None], list[SourceRegistryEntry]] = defaultdict(list)
    for source in sources:
        if source.archive_member_path:
            by_member[(source.parent_archive_source_id, source.archive_member_path)].append(source)
    return [
        _group(
            DuplicateType.DUPLICATE_ARCHIVE_MEMBER_PATH,
            items,
            None,
            "Archive member path appears more than once under the same archive.",
        )
        for _, items in sorted(by_member.items())
        if len(items) > 1
    ]


def _same_name_different_content(sources: list[SourceRegistryEntry]) -> list[DuplicateGroup]:
    by_name: dict[str, list[SourceRegistryEntry]] = defaultdict(list)
    for source in sources:
        by_name[source.logical_path.rsplit("/", 1)[-1].casefold()].append(source)
    return [
        _group(
            DuplicateType.SAME_FILENAME_DIFFERENT_CONTENT,
            items,
            None,
            "Sources share a filename but have different content hashes.",
        )
        for _, items in sorted(by_name.items())
        if len({item.file_hash for item in items}) > 1
    ]


def _group(
    duplicate_type: DuplicateType,
    items: list[SourceRegistryEntry],
    content_hash: str | None,
    rationale: str,
) -> DuplicateGroup:
    ordered = sorted(items, key=lambda item: item.logical_path.casefold())
    seed = "|".join([duplicate_type, *(item.source_id for item in ordered)])
    return DuplicateGroup(
        group_id=f"dup_{sha256(seed.encode()).hexdigest()[:16]}",
        duplicate_type=duplicate_type,
        source_ids=[item.source_id for item in ordered],
        content_hash=content_hash,
        recommended_canonical_source_id=ordered[0].source_id,
        rationale=rationale,
    )
