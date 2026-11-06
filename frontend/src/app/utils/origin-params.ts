/** One importer field the detector Stats window lists: its raw key and
 *  value, and the label the row shows. */
export interface OriginParamRow {
  key: string;
  label: string;
  value: string;
}

/** The rows the detector Stats window shows for the fields a label importer
 *  ran with (#5334). Empty values are dropped. (The dataset Stats window's
 *  rows come worded from the server instead: `source_rows`, #5617.) */
export function originParamRows(params: Record<string, unknown> | null | undefined): OriginParamRow[] {
  if (!params) return [];
  return Object.entries(params)
    .filter(([, v]) => v !== '' && v != null)
    .map(([key, value]) => ({ key, label: formatParamKey(key), value: String(value) }));
}

/** Render a raw origin-param key (e.g. ``media_type``, ``name``) as a label
 *  that matches the hardcoded stat labels: underscores become spaces and the
 *  first letter is capitalized, so a ``filepath`` param reads "Filepath"
 *  alongside "Importer" / "Imported" instead of lowercase. */
function formatParamKey(key: string): string {
  const spaced = key.replace(/_/g, ' ');
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}
