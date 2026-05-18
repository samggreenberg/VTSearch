import { ProgressEvent } from '../models/api.models';

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

/**
 * Format a `ProgressEvent` into the canonical
 * ``[Step S/T] (C/T) message`` string used by every progress consumer.
 *
 * Each piece is optional:
 *   - The ``[Step S/T]`` prefix appears only when ``total_steps > 1``.
 *   - The ``(C/T)`` fraction appears only when ``total > 0``.
 *   - When neither is present, returns the bare ``message`` (or
 *     ``defaultMessage`` if ``message`` is empty).
 */
export function formatProgressMessage(
  progress: ProgressEvent | null | undefined,
  defaultMessage = '',
): string {
  const prog = progress ?? {};
  let msg = prog.message || defaultMessage;
  const step = prog.step;
  const totalSteps = prog.total_steps;
  if (step != null && totalSteps != null && totalSteps > 1) {
    msg = `[Step ${step}/${totalSteps}] ${msg}`;
  }
  const current = prog.current;
  const total = prog.total;
  if (current != null && total != null && total > 0) {
    const fraction = `(${formatProgressFraction(current, total)})`;
    const stepEnd = msg.indexOf('] ');
    if (stepEnd !== -1) {
      msg = msg.slice(0, stepEnd + 2) + fraction + ' ' + msg.slice(stepEnd + 2);
    } else {
      msg = msg ? `${fraction} ${msg}` : fraction;
    }
  }
  return msg;
}

/**
 * Return ``true`` when a progress event has no known total — i.e. callers
 * should show an indeterminate spinner rather than a percent bar.
 */
export function isProgressIndeterminate(
  progress: ProgressEvent | null | undefined,
): boolean {
  const prog = progress ?? {};
  return !(prog.current != null && prog.total != null && prog.total > 0);
}
