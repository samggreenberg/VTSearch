/**
 * Shared date formatting helpers. Always renders absolute dates - no
 * relative ("3d ago") output. Relative time drifts and rounds away
 * precision that matters in a model-management dashboard; columns like
 * `last_trained` and version identifiers need a stable string.
 *
 * - `formatTimestamp(seconds, { withTime: true })` - for `last_trained`,
 *   `created`, ingest start/finish, etc. Renders `YYYY-MM-DD HH:MM` in
 *   the user's local timezone.
 * - `formatTimestamp(seconds, { withTime: false })` - date only.
 * - `formatVersion(iso)` - for the footer version string. Renders
 *   `YYYY-MM-DD` from an ISO 8601 UTC timestamp. Falls through unchanged
 *   if the input doesn't parse (e.g. the `0.0.0-unknown` fallback).
 */

function pad2(n: number): string {
  return n < 10 ? `0${n}` : String(n);
}

function formatLocal(d: Date, withTime: boolean): string {
  const date = `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
  if (!withTime) return date;
  return `${date} ${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}

export function formatTimestamp(
  secondsSinceEpoch: number | null | undefined,
  options: { withTime?: boolean } = {},
): string {
  if (!secondsSinceEpoch) return '-';
  const withTime = options.withTime ?? true;
  return formatLocal(new Date(secondsSinceEpoch * 1000), withTime);
}

export function formatVersion(version: string | null | undefined): string {
  if (!version) return '';
  const d = new Date(version);
  if (Number.isNaN(d.getTime())) return version;
  return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`;
}
