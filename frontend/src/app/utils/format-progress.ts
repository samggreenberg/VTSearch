/**
 * Format a current/total progress fraction for display.
 *
 * When both values are large enough to be byte counts (>= 1 MB),
 * renders them in compact human-readable units (e.g. "497/1.10GB").
 * Otherwise returns a plain "current/total" string with comma
 * separators for readability.
 */
export function formatProgressFraction(current: number, total: number): string {
  const GB = 1_073_741_824;
  const MB = 1_048_576;
  if (total >= GB) {
    return `${(current / GB).toFixed(2)}/${(total / GB).toFixed(2)}GB`;
  }
  if (total >= MB) {
    return `${Math.round(current / MB)}/${Math.round(total / MB)}MB`;
  }
  return `${current.toLocaleString()}/${total.toLocaleString()}`;
}
