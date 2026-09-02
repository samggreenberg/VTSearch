/**
 * The sort half of the managed-table story, split out from
 * `managed-columns.ts` so it can be imported without dragging the
 * `ManagedColumns` class (resize tracking, drag-reorder, localStorage
 * persistence) along with it.
 *
 * The split exists for bundle reasons: the eager top-bar context pulldowns
 * mirror the Dashboard's sort, but they never resize or reorder a column,
 * so the class belongs on the lazy `dashboard-component` chunk while these
 * two tiny declarations stay reachable from the initial bundle.
 */

export interface SortState<TCol extends string = string> {
  column: TCol;
  asc: boolean;
}

/**
 * Sort *rows* by the value of the column named `column`.
 *
 * A table column id is not necessarily a field on the row objects: purely
 * presentational columns (`actions`, `select`) have no value, and rows typed
 * against a generated OpenAPI model expose only the fields the backend
 * declares.  So the lookup is deliberately widened to a `Record` here rather
 * than each caller's row type carrying a catch-all index signature — the
 * index signature is what used to hide backend/frontend schema drift.
 *
 * Numbers compare numerically; everything else compares as a locale string,
 * with a missing value sorting as `''`.
 *
 * `valueAt` overrides that plain field lookup for tables whose sort order
 * isn't the raw cell value: a status column that sorts by a severity rank
 * rather than alphabetically, say, or a name column that sorts
 * case-insensitively. Return a number to opt that column into the numeric
 * comparison and anything else to opt it into the string one — the two
 * branches below are unchanged, so an extractor only decides *what* is
 * compared, never *how*.
 */
export function sortRowsByColumn<T>(
  rows: readonly T[],
  column: string,
  asc: boolean,
  valueAt: (row: T, column: string) => unknown = (row, col) =>
    (row as Record<string, unknown>)[col],
): T[] {
  const dir = asc ? 1 : -1;
  return [...rows].sort((a, b) => {
    const va = valueAt(a, column) ?? '';
    const vb = valueAt(b, column) ?? '';
    if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * dir;
    return String(va).localeCompare(String(vb)) * dir;
  });
}
