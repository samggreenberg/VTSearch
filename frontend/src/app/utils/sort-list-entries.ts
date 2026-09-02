/**
 * The shared sort ladder for the vote-grid-backed item lists: the Find view's
 * Good/Bad label lists, the detector labelset list, and the VTSBrowse
 * selection panel. All three offered the same modes off the same
 * `<vt-view-controls>` dropdown and had each re-spelled the switch locally.
 *
 * The function is pure and returns a new array, so a caller may hold the
 * result in a `computed()` (label-list, labelset-list) or push it into a
 * signal (browse-selection-panel) without either arrangement changing where
 * the sort runs — it stays inside whatever already notified the view.
 */

/**
 * Every ordering the item lists offer. A given list may accept only a subset:
 * the Browse selection panel has no detector score behind it, so its own mode
 * union omits the two `confidence-*` values.
 */
export type ListSortMode =
  | 'time-desc'
  | 'time-asc'
  | 'name-asc'
  | 'name-desc'
  | 'confidence-desc'
  | 'confidence-asc'
  | 'id-asc';

/**
 * What {@link sortListEntries} needs of an entry. Callers extend this with
 * whatever else their vote grid renders.
 */
export interface SortableListEntry {
  /** Numeric for media ids, string for detector label ids; compared in kind. */
  id: number | string;
  name: string;
  /** Recency key — a click timestamp, or a position in insertion order. */
  time: number;
  /** Detector confidence, `-1` when unscored. Absent on lists with no scores. */
  confidence?: number;
}

export function sortListEntries<T extends SortableListEntry>(
  entries: readonly T[],
  mode: ListSortMode,
): T[] {
  const sorted = [...entries];
  switch (mode) {
    case 'time-desc':
      sorted.sort((a, b) => b.time - a.time);
      break;
    case 'time-asc':
      sorted.sort((a, b) => a.time - b.time);
      break;
    case 'name-asc':
      sorted.sort((a, b) => a.name.localeCompare(b.name));
      break;
    case 'name-desc':
      sorted.sort((a, b) => b.name.localeCompare(a.name));
      break;
    case 'confidence-desc':
      sorted.sort((a, b) => (b.confidence ?? -1) - (a.confidence ?? -1));
      break;
    case 'confidence-asc':
      sorted.sort((a, b) => (a.confidence ?? -1) - (b.confidence ?? -1));
      break;
    case 'id-asc':
    default:
      // Media ids are numeric and must compare numerically (10 after 9);
      // detector label ids are opaque strings and compare as such.
      sorted.sort((a, b) =>
        typeof a.id === 'number' && typeof b.id === 'number'
          ? a.id - b.id
          : String(a.id).localeCompare(String(b.id)),
      );
      break;
  }
  return sorted;
}
