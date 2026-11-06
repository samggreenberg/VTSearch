import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { provideHttpTesting } from '../testing/test-providers';
import { DatasetStateService } from './dataset-state.service';
import { DetectorSharingFlowsService, type ForkLanded } from './detector-sharing-flows.service';
import { VtDialogService } from './dialog.service';
import { ToastService } from './toast.service';

/** Let the flow's awaits run: each `await` on a flushed request is a microtask hop or two. */
const drain = async (): Promise<void> => {
  for (let i = 0; i < 10; i++) await Promise.resolve();
};

const REFRESH_URL = '/api/detectors/registry/d1/refresh';
const FORK_URL = '/api/detectors/registry/d1/fork';

const linkedEntry = {
  id: 'd1',
  name: 'Cats',
  sync_mode: 'linked',
  detector_loaded: true,
  autofind: false,
  sharing: { remote_ref: 'c-1', source_name: 'team', last_synced: 1600000000, integrity: null },
};

function refreshBody(applied: boolean, count = 0, message = '') {
  return {
    ok: true,
    applied,
    message: message || (applied ? "Updated 'Cats': 2 label(s) added, 1 changed, 0 removed." : 'foreign'),
    integrity: { status: count ? 'foreign_modifications' : 'clean', count },
    added: applied ? 2 : 0,
    changed: applied ? 1 : 0,
    removed: 0,
    // As the route answers: the entry view, without the listing's own flags.
    detector: {
      id: 'd1',
      name: 'Cats',
      sync_mode: 'linked',
      sharing: {
        remote_ref: 'c-1',
        source_name: 'team',
        last_synced: 1700000000,
        integrity: { status: count ? 'foreign_modifications' : 'clean', count },
      },
    },
  };
}

describe('DetectorSharingFlowsService', () => {
  let flows: DetectorSharingFlowsService;
  let http: HttpTestingController;
  let dialog: VtDialogService;
  let toast: ToastService;
  let store: DatasetStateService;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [...provideHttpTesting()] });
    flows = TestBed.inject(DetectorSharingFlowsService);
    http = TestBed.inject(HttpTestingController);
    dialog = TestBed.inject(VtDialogService);
    toast = TestBed.inject(ToastService);
    store = TestBed.inject(DatasetStateService);
    // Seed the registry the dashboard shows.
    store.refresh();
    http.expectOne('/api/datasets/registry').flush({ datasets: [] });
    http.expectOne('/api/detectors/registry').flush({ detectors: [linkedEntry] });
  });

  afterEach(() => http.verify());

  describe('Check for updates', () => {
    it('asks with abort first, and on a clean container toasts the counts and updates the row in place', async () => {
      const done = flows.checkForUpdates({ id: 'd1', name: 'Cats' });
      const req = http.expectOne(REFRESH_URL);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual({ foreign: 'abort' });
      req.flush(refreshBody(true));
      expect(await done).toBe('applied');

      expect(dialog.dialogOpen()).toBe(false);
      expect(toast.toasts.at(-1)?.level).toBe('success');
      expect(toast.toasts.at(-1)?.message).toContain('2 label(s) added');
      const row = store.detectorById().get('d1');
      expect(row?.sharing?.last_synced).toBe(1700000000);
      expect(row?.sharing?.integrity?.status).toBe('clean');
      // The listing's own flags survive the merge: no reload was needed.
      expect(row?.detector_loaded).toBe(true);
      http.expectNone('/api/detectors/registry');
    });

    it('opens the integrity dialog when the abort found foreign records, and records the integrity on the row', async () => {
      const done = flows.checkForUpdates({ id: 'd1', name: 'Cats' });
      http.expectOne(REFRESH_URL).flush(refreshBody(false, 3));
      await drain();

      expect(dialog.dialogOpen()).toBe(true);
      expect(dialog.dialogMessage()).toContain('3 votes were modified by someone other than the owner');
      expect(dialog.dialogButtons().map((b) => b.label)).toEqual(['Cancel', 'Import anyway', 'Skip foreign records']);
      expect(store.detectorById().get('d1')?.sharing?.integrity?.count).toBe(3);

      dialog.cancel();
      expect(await done).toBe('cancelled');
      http.expectNone(REFRESH_URL);
    });

    for (const [label, policy] of [
      ['Import anyway', 'import'],
      ['Skip foreign records', 'skip'],
    ] as const) {
      it(`re-posts with foreign: ${policy} on "${label}"`, async () => {
        const done = flows.checkForUpdates({ id: 'd1', name: 'Cats' });
        http.expectOne(REFRESH_URL).flush(refreshBody(false, 2));
        await drain();
        dialog.resolve(dialog.dialogButtons().find((b) => b.label === label)!.value);
        await drain();

        const again = http.expectOne(REFRESH_URL);
        expect(again.request.body).toEqual({ foreign: policy });
        again.flush(refreshBody(true, 2));
        expect(await done).toBe('applied');
        expect(toast.toasts.at(-1)?.level).toBe('success');
      });
    }

    it('resolves failed, without a dialog, when the request fails', async () => {
      const done = flows.checkForUpdates({ id: 'd1', name: 'Cats' });
      http.expectOne(REFRESH_URL).flush({ message: 'nope' }, { status: 502, statusText: 'Bad Gateway' });
      expect(await done).toBe('failed');
      expect(dialog.dialogOpen()).toBe(false);
    });
  });

  describe('Fork', () => {
    const forkBody = (remote_sync: string | null) => ({
      ok: true,
      message: "Forked 'Cats' as 'Cats (fork)'.",
      detector: { id: 'd2', name: 'Cats (fork)', sync_mode: 'forked' },
      remote_sync,
    });

    it('sends no name when the suggested one is kept, refreshes the registry and announces the copy', async () => {
      const landed: ForkLanded[] = [];
      flows.forked$.subscribe((f) => landed.push(f));
      const done = flows.fork({ id: 'd1', name: 'Cats' }, { askName: true });
      await drain();
      expect(dialog.dialogShowInput()).toBe(true);
      expect(dialog.dialogInputValue()).toBe('Cats (fork)');
      dialog.resolve('__input__');
      await drain();

      const req = http.expectOne(FORK_URL);
      expect(req.request.body).toEqual({});
      req.flush(forkBody(null));
      const entry = await done;
      expect(entry?.id).toBe('d2');
      expect(landed).toEqual([{ sourceId: 'd1', entry: expect.objectContaining({ id: 'd2' }) }]);
      http.expectOne('/api/datasets/registry').flush({ datasets: [] });
      http.expectOne('/api/detectors/registry').flush({ detectors: [linkedEntry, { id: 'd2', name: 'Cats (fork)' }] });
      expect(toast.toasts.at(-1)?.level).toBe('success');
    });

    it('sends a name the user typed', async () => {
      const done = flows.fork({ id: 'd1', name: 'Cats' }, { askName: true });
      await drain();
      dialog.dialogInputValue.set('  Mine  ');
      dialog.resolve('__input__');
      await drain();
      const req = http.expectOne(FORK_URL);
      expect(req.request.body).toEqual({ name: 'Mine' });
      req.flush(forkBody(null));
      await done;
      http.expectOne('/api/datasets/registry').flush({ datasets: [] });
      http.expectOne('/api/detectors/registry').flush({ detectors: [] });
    });

    it('turns a remote_sync note into a toast', async () => {
      const done = flows.fork({ id: 'd1', name: 'Cats' });
      http.expectOne(FORK_URL).flush(forkBody('Detector sharing is off on this server, so this fork is local only.'));
      await done;
      http.expectOne('/api/datasets/registry').flush({ datasets: [] });
      http.expectOne('/api/detectors/registry').flush({ detectors: [] });
      expect(toast.toasts.at(-1)?.message).toContain('this fork is local only');
    });

    it('does nothing when the name prompt is cancelled', async () => {
      const done = flows.fork({ id: 'd1', name: 'Cats' }, { askName: true });
      await drain();
      dialog.cancel();
      expect(await done).toBeNull();
      http.expectNone(FORK_URL);
    });
  });
});
