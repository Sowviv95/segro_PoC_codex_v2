# Parser Foundation Stage C Cache

Stage B checkpoints and Stage C parsed-page cache artifacts serve different purposes.

Stage B checkpoints are execution-recovery artifacts. They record worker progress, completed
pages, failed pages, restart count, and first incomplete page for one bounded worker execution.
They are allowed to be tied to a specific run directory.

Stage C cache artifacts are canonical reusable parsed-page content. They are independent of
worker restarts, process IDs, timestamps, temporary paths, and output directories. A cached page
can be reused by a later identical bounded request without opening the PDF.

## Cache Key

Each page cache key is the SHA-256 hash of stable JSON containing:

- `schema_version`
- `source_content_hash`
- `page_number`
- `parser_name`
- `parser_version`
- `parser_config_fingerprint`

The key excludes output directory, run timestamp, process ID, restart count, and temporary paths.

## Layout

Parsed pages are written as UTF-8 JSON:

```text
<cache-root>/
  <source-id>/
    <parser-fingerprint>/
      page_000001.json
      page_000002.json
```

Writes are atomic: JSON is written to a sibling temporary file and then moved into place.
Incomplete temporary files are ignored because reads only target deterministic page paths.

## Invalidation

A cache miss occurs when any identity input changes: source content hash, parser name, parser
version, parser configuration fingerprint, or parsed-page schema version. Corrupt or malformed
artifacts are reported as invalid cache entries and treated as misses.

## Partial Ranges

The cache service reads the explicit requested range, returns valid cached pages immediately, and
sends only missing contiguous ranges to Stage B workers. Returned pages are ordered by the
requested page range.

## Scope

The cache stores provider-neutral parsed-page content only: extracted text, counts, parse status,
warnings, duration, parser identity, source identity, schema version, and cache key. It does not
store parser-library objects, embeddings, hierarchy nodes, extraction results, model outputs, or
database/vector-store state.
