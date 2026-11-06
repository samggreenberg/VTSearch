import { buildDetectorCardMenuItems, type DetectorMenuEntry } from './card-context-menu-items';

/** The ids a menu lists, in order. */
const ids = (entry: DetectorMenuEntry, enabled = false, isOwner = true): string[] =>
  buildDetectorCardMenuItems(entry, { isDefaultLogin: true, isOwner }, { enabled }).map((i) => i.id);

const item = (entry: DetectorMenuEntry, id: string, enabled = false, isOwner = true) =>
  buildDetectorCardMenuItems(entry, { isDefaultLogin: true, isOwner }, { enabled }).find((i) => i.id === id);

const NOTE = 'Shared through Team Drive, but detector sharing is off on this server; this is a local copy until that changes.';

const plain: DetectorMenuEntry = { detector_loaded: true, sync_mode: 'local', sharing: null };
const linked: DetectorMenuEntry = {
  detector_loaded: true,
  sync_mode: 'linked',
  sharing: { remote_ref: 'c-1', source_name: 'team', remote_owner: 'ana' },
  labelset_source_status: { source_name: 'team', display_name: 'Team Drive' },
};
const sharedOwned: DetectorMenuEntry = {
  detector_loaded: true,
  sync_mode: 'local',
  sharing: { remote_ref: 'c-2', source_name: 'team' },
  labelset_source_status: { source_name: 'team', display_name: 'Team Drive' },
};
const plainBound: DetectorMenuEntry = {
  detector_loaded: true,
  sync_mode: 'local',
  sharing: null,
  labelset_source_status: { source_name: 'server_json_file', display_name: 'Server JSON file' },
};

describe('buildDetectorCardMenuItems: detector sharing', () => {
  it('leaves a plain detector menu as it was when sharing is off', () => {
    expect(ids(plain)).toEqual(['browse', 'rename', 'add-labels', 'export', 'stats', 'move-to-autofind', 'delete']);
  });

  it('offers Share… and Fork on a plain detector when sharing is on', () => {
    expect(ids(plain, true)).toEqual([
      'browse',
      'rename',
      'add-labels',
      'export',
      'share',
      'fork',
      'stats',
      'move-to-autofind',
      'delete',
    ]);
    expect(item(plain, 'share', true)?.disabled).toBe(false);
  });

  it('keeps Auto-export… beside Share… while the binding is a plain one', () => {
    expect(ids(plainBound, true)).toEqual(expect.arrayContaining(['auto-export', 'share']));
  });

  it('gives a received detector the frozen treatment, Check for updates and Fork', () => {
    const got = ids(linked);
    expect(got).toEqual(['browse', 'export', 'check-updates', 'fork', 'stats', 'move-to-autofind', 'delete']);
    for (const absent of ['rename', 'add-labels', 'auto-export', 'share']) expect(got).not.toContain(absent);
    // Sharing on changes nothing for it: it never shares.
    expect(ids(linked, true)).toEqual(got);
    expect(item(linked, 'delete')?.title).toContain('local copy');
  });

  it("leaves Check for updates to the detector's own creator", () => {
    expect(item(linked, 'check-updates')?.disabled).toBe(false);
    const other = item(linked, 'check-updates', false, false);
    expect(other?.disabled).toBe(true);
    expect(other?.title).toContain('received');
  });

  it('keeps Move to Drafts on a received AutoFind detector, without Delete', () => {
    const got = ids({ ...linked, autofind: true });
    expect(got).toContain('move-to-drafts');
    expect(got).not.toContain('delete');
    expect(got).toEqual(expect.arrayContaining(['check-updates', 'fork']));
  });

  it('puts Share… in place of Auto-export… on a shared detector', () => {
    const got = ids(sharedOwned, true);
    expect(got).toContain('share');
    expect(got).not.toContain('auto-export');
    expect(got).toContain('fork');
    expect(item(sharedOwned, 'share', true, false)?.disabled).toBe(true);
  });

  it('disables the sharing verbs on a row with a note, with the note as the reason, and keeps Fork', () => {
    const notedShared = { ...sharedOwned, sharing_note: NOTE };
    const share = item(notedShared, 'share');
    expect(share?.disabled).toBe(true);
    expect(share?.title).toBe(NOTE);
    expect(ids(notedShared)).not.toContain('auto-export');

    const notedLinked = { ...linked, sharing_note: NOTE };
    const check = item(notedLinked, 'check-updates');
    expect(check?.disabled).toBe(true);
    expect(check?.title).toBe(NOTE);
    expect(item(notedLinked, 'fork')?.disabled).toBeFalsy();
  });
});
