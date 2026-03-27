/**
 * Format a current/total progress fraction for display.
 *
 * When both values are large enough to be byte counts (>= 1 MB),
 * renders them in human-readable units (e.g. "497 MB / 1.10 GB").
 * Otherwise returns a plain "current/total" string.
 */
export function formatProgressFraction(current: number, total: number): string {
  const MB = 1_048_576;
  if (total >= MB) {
    return `${formatBytes(current)} / ${formatBytes(total)}`;
  }
  return `${current}/${total}`;
}

function formatBytes(bytes: number): string {
  const GB = 1_073_741_824;
  const MB = 1_048_576;
  if (bytes >= GB) {
    return `${(bytes / GB).toFixed(2)} GB`;
  }
  return `${Math.round(bytes / MB)} MB`;
}
