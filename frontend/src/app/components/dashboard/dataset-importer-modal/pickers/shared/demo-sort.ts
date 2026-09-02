import type { DemoDatasetEntry } from '../../../../../generated/api-client/models/demo-dataset-entry';

/**
 * Sort rank for the demo `status` column. Alphabetical order would read as
 * `needs_download < needs_embedding < ready`, burying the demos a user can
 * actually open beneath the ones they'd have to fetch first; this puts the
 * usable ones on top.
 */
const STATUS_ORDER: Record<string, number> = {
  ready: 0,
  needs_embedding: 1,
  needs_download: 2,
};

/**
 * Sort-key extractor for the demo-dataset table, shared by the Add-Dataset
 * demo picker and the New-detector modal's copy of the same table so the two
 * stay in the same order.
 *
 * Pass to {@link sortRowsByColumn} as its `valueAt` argument. Numbers pass
 * through for the numeric comparison; everything else is lowercased so the
 * name/label columns sort case-insensitively.
 */
export function demoSortValue(row: DemoDatasetEntry, column: string): unknown {
  const raw = row[column as keyof DemoDatasetEntry];
  if (column === 'status') return STATUS_ORDER[raw as string] ?? 3;
  return typeof raw === 'number' ? raw : String(raw || '').toLowerCase();
}
