import type { AutoFindExportStatus } from '../models/api.models';

/** Where a failed AutoFind auto-export is put right (#5463): the tab that
 *  picks the exporter, holds its fields, and can pause it. */
export const AUTO_EXPORT_FIX_HINT = 'Check the exporter in Settings › AutoFind.';

/**
 * One sentence for a failed AutoFind auto-export, shared by the run's toast
 * and the Find Results dialog: the exporter by the name the user picked it
 * under (never its registry name, such as `webhook`), the reason, and where
 * to fix it.
 */
export function autoExportFailure(status: AutoFindExportStatus): string {
  const reason = status.error?.trim() || 'unknown error';
  const stop = /[.!?]$/.test(reason) ? '' : '.';
  return `Sending the results via ${status.exporter_display_name} failed: ${reason}${stop} ${AUTO_EXPORT_FIX_HINT}`;
}
