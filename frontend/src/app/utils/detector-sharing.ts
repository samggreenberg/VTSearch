import { formatTimestamp } from './format-date';

/**
 * What a registry entry says about detector sharing, read the same way by the
 * Detectors card, its menu and the Train view (#5578).
 *
 * The listing (`GET /api/detectors/registry`) carries the state on every
 * entry: `sync_mode` (`local`, `linked` for a received read-only copy,
 * `forked`), `sharing` (the container the entry mirrors or owns) and
 * `sharing_note` (why that container is not in use here, when it is not).
 * Nothing here decides anything the server has not already decided; these are
 * the readings, and the words the rows show for them.
 */

/** The slice of a registry entry these helpers read. */
export interface SharingEntry {
  sync_mode?: string | null;
  sharing?: {
    remote_ref?: string;
    source_name?: string;
    remote_owner?: string | null;
    last_synced?: number | null;
    integrity?: { status?: string; count?: number } | null;
    sync_error?: string | null;
  } | null;
  sharing_note?: string | null;
  labelset_source_status?: { source_name?: string; display_name?: string } | null;
}

/** The words of the read-only hint, wherever a received detector refuses an edit. */
export const READ_ONLY_HINT = 'Read-only: fork to edit';

/** Integrity `status` of a container nobody but its owner has written to. */
const INTEGRITY_CLEAN = 'clean';

/** True for a received, read-only copy of someone else's detector. */
export function isLinked(entry: SharingEntry | null | undefined): boolean {
  return entry?.sync_mode === 'linked';
}

/** True for a detector the user shares: not a received copy, and bound to a container. */
export function isSharedOwned(entry: SharingEntry | null | undefined): boolean {
  return !isLinked(entry) && !!entry?.sharing?.remote_ref;
}

/** The note saying why the entry's container is not in use here, or `''`. */
export function sharingNote(entry: SharingEntry | null | undefined): string {
  return entry?.sharing_note ?? '';
}

/** The last owner push's failure on a shared detector, or `''`. */
export function sharingSyncError(entry: SharingEntry | null | undefined): string {
  return isSharedOwned(entry) ? (entry?.sharing?.sync_error ?? '') : '';
}

/** The sharing source's display name: the binding's status names it, else its id. */
function sourceLabel(entry: SharingEntry): string {
  const name = entry.sharing?.source_name ?? '';
  const status = entry.labelset_source_status;
  if (status && status.source_name === name && status.display_name) return status.display_name;
  return name || 'a sharing source';
}

/** True when the last refresh found votes in the container that its owner did not write. */
export function hasForeignModifications(entry: SharingEntry | null | undefined): boolean {
  const status = entry?.sharing?.integrity?.status;
  return !!status && status !== INTEGRITY_CLEAN;
}

/** One line on what the last refresh found in the container, or `''` before any. */
export function integrityLine(entry: SharingEntry | null | undefined): string {
  const integrity = entry?.sharing?.integrity;
  if (!integrity?.status) return '';
  if (!hasForeignModifications(entry)) return 'Every vote is the owner’s.';
  const n = integrity.count ?? 0;
  return `⚠ ${n} vote${n === 1 ? ' was' : 's were'} modified by someone other than the owner.`;
}

/** When the container was last read or written, as a sentence, or `''`. */
function syncedLine(entry: SharingEntry, verb: string): string {
  const at = entry.sharing?.last_synced;
  return at ? `${verb} ${formatTimestamp(at)}.` : '';
}

/**
 * The Received badge's title: who it came from, that it is read-only, when it
 * was last checked for updates and what that check found.
 */
export function receivedTitle(entry: SharingEntry): string {
  const owner = entry.sharing?.remote_owner;
  return [
    owner ? `Received from ${owner}.` : 'Received from another user.',
    `Read-only: fork it to make a copy you can edit.`,
    syncedLine(entry, 'Last checked for updates'),
    integrityLine(entry),
    sharingNote(entry),
  ]
    .filter(Boolean)
    .join('\n');
}

/**
 * The Shared badge's title: where the labels go and when they last got there.
 * `recipients` is the count from the container when the caller has it
 * (`null` for "not known"), and `isPublic` says everyone can see it.
 */
export function sharedTitle(entry: SharingEntry, recipients: number | null = null, isPublic = false): string {
  let audience = '';
  if (isPublic) audience = 'Everyone can receive it.';
  else if (recipients !== null) audience = `Shared with ${recipients} recipient${recipients === 1 ? '' : 's'}.`;
  return [
    `Shared through ${sourceLabel(entry)}: every save reaches the people it is shared with.`,
    audience,
    syncedLine(entry, 'Last saved there'),
    sharingSyncError(entry) ? `The last save failed: ${sharingSyncError(entry)}` : '',
    sharingNote(entry),
  ]
    .filter(Boolean)
    .join('\n');
}
