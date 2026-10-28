/** One importer field a Stats window lists: its raw key and value, and the
 *  label the row shows. */
export interface OriginParamRow {
  key: string;
  label: string;
  value: string;
}

/** The rows a Stats window shows for an importer's recorded fields: a
 *  dataset's origin `params`, or the fields a detector was imported with
 *  (#5334). Empty values are dropped. */
export function originParamRows(params: Record<string, unknown> | null | undefined): OriginParamRow[] {
  if (!params) return [];
  return Object.entries(params)
    .filter(([, v]) => v !== '' && v != null)
    .map(([key, value]) => ({ key, label: formatParamKey(key), value: String(value) }));
}

/** Render a raw origin-param key (e.g. ``media_type``, ``name``) as a label
 *  that matches the hardcoded stat labels: underscores become spaces and the
 *  first letter is capitalized, so a demo dataset's ``name`` param reads
 *  "Name" alongside "Importer" / "Clipper" / "Embedder" instead of lowercase. */
function formatParamKey(key: string): string {
  const spaced = key.replace(/_/g, ' ');
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}
