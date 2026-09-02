/**
 * Shared display formatters for byte counts and custom-metadata values —
 * the two helpers that had been copied verbatim into a component whenever
 * one was needed.
 *
 * Both are pure and synchronous, so a template may bind them directly (as
 * the folder browser and the center panel do) without any change-detection
 * consequence: they read only their arguments, never component state, so
 * whatever already notified the view of a new argument is still the only
 * notification path.
 */

/**
 * Human-readable size for a byte count: `512 B`, `2.0 KB`, `5.0 MB`.
 *
 * A missing size renders as the empty string rather than `0 B` or `-`, so a
 * table cell for a row that legitimately has no size (a directory) stays
 * blank instead of claiming the size is zero.
 */
export function formatBytes(bytes?: number | null): string {
  if (bytes == null) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Format one custom-metadata entry for display, keyed off the category
 * *label* rather than a declared type: `custom_metadata` is an untyped
 * `Record<string, unknown>` from the importer, so the label is the only
 * signal available about what a value means.
 *
 * Unrecognised labels — and values whose runtime type doesn't match the
 * label's expectation — fall through to `String(value)` unchanged.
 *
 * Shared by the center panel's metadata list and the Browse bin popup's
 * copy of it, so the same categories read identically in both.
 */
export function formatMetadataValue(label: string, value: unknown): string {
  if (label === 'File Size' && typeof value === 'number') {
    return (value / 1024).toFixed(1) + ' KB';
  }
  if (label === 'Duration' && typeof value === 'number') {
    return value.toFixed(1) + 's';
  }
  if (label === 'Frequency' && typeof value === 'number') {
    return value + ' Hz';
  }
  return String(value);
}
