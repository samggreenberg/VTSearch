/**
 * Datasets one multi-dataset import produced (#4747).
 *
 * A multi-dataset import fans one importer run into several datasets ("Photos
 * – Image", "Photos – Face"), and the registry stamps every one of them with
 * the run's `import_group` id plus the `output_category` it stands for. These
 * helpers let a list of registry rows treat those siblings as a unit: kept
 * together whatever the sort, and able to name each other.
 *
 * Kept free of Angular and of the Dashboard's column machinery because the
 * eager top-bar context pulldowns mirror the Dashboard's dataset order too.
 */

/** The slice of a registry row these helpers read. */
export interface ImportGroupedRow {
  name: string;
  import_group?: string | null;
}

/**
 * Pull each import group's rows together, keeping the order otherwise.
 *
 * A group lands where its first row already stood in *rows*, and its members
 * keep their relative order, so a sorted list stays sorted by its first
 * member: sorting by name puts "Photos – Face" right after "Photos – Image"
 * even if "Pets" would otherwise fall between them. Rows with no group are
 * untouched.
 */
export function clusterImportSiblings<T extends ImportGroupedRow>(rows: readonly T[]): T[] {
  const members = new Map<string, T[]>();
  for (const row of rows) {
    const group = row.import_group;
    if (!group) continue;
    const list = members.get(group);
    if (list) list.push(row);
    else members.set(group, [row]);
  }
  if (members.size === 0) return [...rows];

  const out: T[] = [];
  for (const row of rows) {
    const group = row.import_group;
    if (!group) {
      out.push(row);
      continue;
    }
    // The first member places the whole group; later members were already
    // emitted with it.
    const list = members.get(group);
    if (list) {
      out.push(...list);
      members.delete(group);
    }
  }
  return out;
}

/**
 * The names of each row's import siblings, keyed by row id.
 *
 * A row with no group, or whose siblings have all been deleted, has no entry:
 * there is nobody left to name.
 */
export function importSiblingNames<T extends ImportGroupedRow & { id: string }>(
  rows: readonly T[],
): Map<string, string[]> {
  const byGroup = new Map<string, T[]>();
  for (const row of rows) {
    const group = row.import_group;
    if (!group) continue;
    const list = byGroup.get(group);
    if (list) list.push(row);
    else byGroup.set(group, [row]);
  }
  const out = new Map<string, string[]>();
  for (const list of byGroup.values()) {
    if (list.length < 2) continue;
    for (const row of list) {
      out.set(
        row.id,
        list.filter((other) => other !== row).map((other) => other.name),
      );
    }
  }
  return out;
}

/**
 * The rows *row*'s import produced for *mediaType*, other than *row* itself.
 *
 * Show in photo (#4750) offers itself on a Face dataset only when the same
 * import also produced an Image dataset, since that is where a face's photo
 * lives as an item. Empty for a row with no group.
 */
export function importSiblingsOfType<T extends ImportGroupedRow & { id: string; media_type: string }>(
  row: T | null | undefined,
  rows: readonly T[],
  mediaType: string,
): T[] {
  const group = row?.import_group;
  if (!row || !group) return [];
  return rows.filter((other) => other.import_group === group && other.id !== row.id && other.media_type === mediaType);
}
