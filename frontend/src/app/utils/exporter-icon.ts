import type { ExporterEntry } from '../generated/api-client/models/exporter-entry';

/** What kind of destination an exporter is, for the icon on its tab. */
export type ExporterIconKind = 'email' | 'server' | 'webhook' | 'external-link' | 'upload';

/**
 * The icon kind for *exp*'s tab: from the emoji its author gave it, else from
 * what it does (`opens_url` reads as a link, whichever emoji was picked), else
 * from its name. Shared by every dialog that lists exporters as tabs (Export,
 * Find Results), so one exporter wears one icon everywhere.
 */
export function exporterIconKind(exp: ExporterEntry): ExporterIconKind {
  const icon = exp.icon || '';
  if (icon.includes('📧')) return 'email';
  if (icon === '🖥️' || icon === '🖥' || icon === '🖥️') return 'server';
  if (icon === '🌐' || icon === '🌐') return 'webhook';
  if (exp.opens_url) return 'external-link';
  const name = (exp.name || '').toLowerCase();
  if (name.includes('email') || name.includes('smtp')) return 'email';
  if (name.includes('webhook')) return 'webhook';
  if (name.includes('server') || name.includes('file')) return 'server';
  return 'upload';
}
