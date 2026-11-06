import { signal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { RECIPIENT_SEARCH_DEBOUNCE_MS, ShareDetectorModalComponent } from './share-detector-modal.component';
import { DatasetStateService } from '../../../services/dataset-state.service';
import { SharingCapabilityService } from '../../../services/sharing-capability.service';
import { VtDialogService } from '../../../services/dialog.service';
import { ToastService } from '../../../services/toast.service';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

const SHARE_URL = '/api/detectors/registry/det1/share';
const VIEW_URL = '/api/detectors/registry/det1/sharing';
const SEARCH_URL = '/api/detectors/sharing/search-recipients';

/** A detector of the user's own, as the registry lists it. */
const plain = { id: 'det1', name: 'Barks', media_type: 'audio', sync_mode: 'local', sharing: null, sharing_note: null };
/** The same detector, auto-exported to a JSON file. */
const autoExported = {
  ...plain,
  labelset_source_status: { source_name: 'server_json_file', display_name: 'Server JSON File', inherited: true, missing_fields: [] },
};
/** The same detector, shared. */
const shared = { ...plain, sharing: { source_name: 'shared_folder', remote_ref: 'c1', sync_error: null } };

/** The container as `GET .../sharing` and `POST .../share` read it. */
function container(recipients: string[], isPublic = false, extra: object = {}): object {
  return {
    detector_id: 'det1',
    sync_mode: 'local',
    is_owner: true,
    sharing: { source_name: 'shared_folder', remote_ref: 'c1' },
    sharing_note: null,
    owner: 'me',
    recipients,
    public: isPublic,
    ...extra,
  };
}

function shareResponse(recipients: string[], isPublic = false, created = false, message = "'Barks' is shared."): object {
  return container(recipients, isPublic, { ok: true, created, message });
}

describe('ShareDetectorModalComponent', () => {
  let fixture: ComponentFixture<ShareDetectorModalComponent>;
  let httpMock: HttpTestingController;
  let detectors: ReturnType<typeof signal<Map<string, object>>>;
  let state: { detectorById: typeof detectors; patchDetector: ReturnType<typeof vi.fn>; refresh: ReturnType<typeof vi.fn> };
  let canSearch: ReturnType<typeof signal<boolean>>;
  let closed: number;

  const el = (): HTMLElement => fixture.nativeElement as HTMLElement;
  const text = (): string => el().textContent ?? '';
  const entryBox = (): HTMLInputElement => el().querySelector('#share-recipient') as HTMLInputElement;
  const publicBox = (): HTMLInputElement => el().querySelector('.share-public input') as HTMLInputElement;
  const recipientNames = (): string[] =>
    Array.from(el().querySelectorAll('.share-recipient-name')).map((n) => n.textContent!.trim());

  async function open(entry: object, opts: { search?: boolean } = {}): Promise<void> {
    canSearch.set(opts.search ?? false);
    detectors.set(new Map([['det1', entry]]));
    fixture = TestBed.createComponent(ShareDetectorModalComponent);
    fixture.componentRef.setInput('detectorId', 'det1');
    fixture.componentRef.setInput('detectorName', 'Barks');
    closed = 0;
    fixture.componentInstance.closed.subscribe(() => closed++);
    await settleZoneless(fixture);
  }

  async function typeIn(value: string): Promise<void> {
    entryBox().value = value;
    entryBox().dispatchEvent(new Event('input'));
    await settleZoneless(fixture);
  }

  async function pressEnter(): Promise<void> {
    entryBox().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    await settleZoneless(fixture);
  }

  beforeEach(async () => {
    detectors = signal(new Map());
    state = { detectorById: detectors, patchDetector: vi.fn(), refresh: vi.fn() };
    canSearch = signal(false);
    await configureZoneless({
      imports: [ShareDetectorModalComponent],
      providers: [
        ...provideHttpTesting(),
        { provide: DatasetStateService, useValue: state },
        {
          provide: SharingCapabilityService,
          useValue: { enabled: signal(true), displayName: signal('Shared folder'), supportsRecipientSearch: canSearch },
        },
      ],
    }).compileComponents();
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    vi.useRealTimers();
    httpMock.verify();
  });

  describe('a detector without a container', () => {
    it('says what sharing will do, reads nothing, and offers no Stop Sharing', async () => {
      await open(plain);
      httpMock.expectNone(VIEW_URL);
      expect(el().querySelector('.share-first')!.textContent).toContain('Shared folder');
      expect(el().querySelector('.share-replaces')).toBeNull();
      expect(text()).toContain('Nobody yet.');
      expect(el().querySelector('.share-stop-btn')).toBeNull();
    });

    it('says the plain auto-export binding is replaced', async () => {
      await open(autoExported);
      expect(el().querySelector('.share-replaces')!.textContent).toContain('Server JSON File');
    });

    it('creates the container with the first change, and says what that replaced', async () => {
      const success = vi.spyOn(TestBed.inject(ToastService), 'success');
      await open(autoExported);
      await typeIn('bob');
      await pressEnter();

      const req = httpMock.expectOne(SHARE_URL);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual({ add: ['bob'] });
      const message = "Shared 'Barks' through Shared folder. Its labels are no longer auto-exported to Server JSON File. Added 1 recipient(s).";
      req.flush(shareResponse(['bob'], false, true, message));
      await settleZoneless(fixture);

      expect(recipientNames()).toEqual(['bob']);
      expect(success).toHaveBeenCalledWith({ message });
      expect(state.refresh).toHaveBeenCalled();
      // Shared now: the first-share copy goes and Stop Sharing comes.
      expect(el().querySelector('.share-first')).toBeNull();
      expect(el().querySelector('.share-stop-btn')).not.toBeNull();
      expect(entryBox().value).toBe('');
    });
  });

  describe('a shared detector', () => {
    async function openShared(recipients: string[], isPublic = false, opts: { search?: boolean } = {}): Promise<void> {
      await open(shared, opts);
      httpMock.expectOne(VIEW_URL).flush(container(recipients, isPublic));
      await settleZoneless(fixture);
    }

    it('lists who it is shared with, and whether publicly', async () => {
      await openShared(['carol', 'Biologists'], true);
      expect(recipientNames()).toEqual(['carol', 'Biologists']);
      expect(publicBox().checked).toBe(true);
      expect(el().querySelector('.share-first')).toBeNull();
    });

    it('adds a typed identity with Enter when the source cannot search', async () => {
      await openShared([]);
      expect(el().querySelector('vt-plugin-combobox')).toBeNull();
      await typeIn('  dave  ');
      await pressEnter();
      const req = httpMock.expectOne(SHARE_URL);
      expect(req.request.body).toEqual({ add: ['dave'] });
      req.flush(shareResponse(['dave']));
      await settleZoneless(fixture);
      expect(recipientNames()).toEqual(['dave']);
      expect(entryBox().value).toBe('');
    });

    it('adds a typed identity with the Add button', async () => {
      await openShared([]);
      const add = el().querySelector('.share-add-btn') as HTMLButtonElement;
      expect(add.disabled).toBe(true);
      await typeIn('erin');
      expect(add.disabled).toBe(false);
      add.click();
      httpMock.expectOne(SHARE_URL).flush(shareResponse(['erin']));
      await settleZoneless(fixture);
      expect(recipientNames()).toEqual(['erin']);
    });

    it('searches as the user types and adds the suggestion picked', async () => {
      await openShared(['carol'], false, { search: true });
      expect(el().querySelector('vt-plugin-combobox')).not.toBeNull();

      vi.useFakeTimers();
      entryBox().value = 'ali';
      entryBox().dispatchEvent(new Event('input'));
      // Nothing is asked until the user pauses.
      httpMock.expectNone(SEARCH_URL);
      vi.advanceTimersByTime(RECIPIENT_SEARCH_DEBOUNCE_MS);
      vi.useRealTimers();

      const search = httpMock.expectOne(SEARCH_URL);
      expect(search.request.body).toEqual({ query: 'ali' });
      search.flush({
        suggestions: [
          { id: 'alice@example.org', display_name: 'Alice Smith', kind: 'user' },
          { id: 'g-ali', display_name: 'Alien hunters', kind: 'group' },
          // Already a recipient: not offered again.
          { id: 'carol', display_name: 'Carol', kind: 'user' },
        ],
      });
      await settleZoneless(fixture);

      const rows = Array.from(el().querySelectorAll('[role="option"]')) as HTMLElement[];
      // Each suggestion by name, with its kind beside it.
      expect(rows.map((r) => r.querySelector('.plugin-combobox-option-label')!.firstChild!.textContent!.trim())).toEqual([
        'Alice Smith',
        'Alien hunters',
      ]);
      expect(rows.map((r) => r.querySelector('.plugin-combobox-option-tag')!.textContent!.trim())).toEqual([
        'person',
        'group',
      ]);
      rows[1].click();
      await settleZoneless(fixture);

      const req = httpMock.expectOne(SHARE_URL);
      expect(req.request.body).toEqual({ add: ['g-ali'] });
      req.flush(shareResponse(['carol', 'g-ali']));
      await settleZoneless(fixture);
      expect(recipientNames()).toEqual(['carol', 'g-ali']);
      expect(entryBox().value).toBe('');
      expect(el().querySelectorAll('[role="option"]')).toHaveLength(0);
    });

    it('adds the suggestion whose name is typed exactly, with Enter', async () => {
      await openShared([], false, { search: true });
      vi.useFakeTimers();
      entryBox().value = 'Alice Smith';
      entryBox().dispatchEvent(new Event('input'));
      vi.advanceTimersByTime(RECIPIENT_SEARCH_DEBOUNCE_MS);
      vi.useRealTimers();
      httpMock
        .expectOne(SEARCH_URL)
        .flush({ suggestions: [{ id: 'alice@example.org', display_name: 'Alice Smith', kind: 'user' }] });
      await settleZoneless(fixture);

      await pressEnter();
      const req = httpMock.expectOne(SHARE_URL);
      expect(req.request.body).toEqual({ add: ['alice@example.org'] });
      req.flush(shareResponse(['alice@example.org']));
      await settleZoneless(fixture);
    });

    it('removes a recipient', async () => {
      await openShared(['carol', 'dave']);
      (el().querySelector('[aria-label="Remove carol"]') as HTMLButtonElement).click();
      const req = httpMock.expectOne(SHARE_URL);
      expect(req.request.body).toEqual({ remove: ['carol'] });
      req.flush(shareResponse(['dave']));
      await settleZoneless(fixture);
      expect(recipientNames()).toEqual(['dave']);
    });

    it('turns Public on and off', async () => {
      await openShared([]);
      publicBox().click();
      const on = httpMock.expectOne(SHARE_URL);
      expect(on.request.body).toEqual({ public: true });
      on.flush(shareResponse([], true));
      await settleZoneless(fixture);
      expect(publicBox().checked).toBe(true);

      publicBox().click();
      const off = httpMock.expectOne(SHARE_URL);
      expect(off.request.body).toEqual({ public: false });
      off.flush(shareResponse([], false));
      await settleZoneless(fixture);
      expect(publicBox().checked).toBe(false);
    });

    it('puts the Public box back when the change is refused', async () => {
      await openShared([]);
      publicBox().click();
      httpMock
        .expectOne(SHARE_URL)
        .flush({ message: 'The sharing service failed', error_code: 'sharing_failed' }, { status: 502, statusText: 'Bad Gateway' });
      await settleZoneless(fixture);
      expect(publicBox().checked).toBe(false);
    });

    it('holds every change while one is in flight', async () => {
      await openShared(['carol']);
      publicBox().click();
      await settleZoneless(fixture);
      expect(publicBox().disabled).toBe(true);
      expect((el().querySelector('[aria-label="Remove carol"]') as HTMLButtonElement).disabled).toBe(true);
      httpMock.expectOne(SHARE_URL).flush(shareResponse(['carol'], true));
      await settleZoneless(fixture);
      expect(publicBox().disabled).toBe(false);
    });

    it('stops sharing after a confirm, and closes', async () => {
      const confirm = vi.spyOn(TestBed.inject(VtDialogService), 'confirmDestructive').mockResolvedValue(true);
      await openShared(['carol']);
      (el().querySelector('.share-stop-btn') as HTMLButtonElement).click();
      await settleZoneless(fixture);
      expect(confirm).toHaveBeenCalledWith('Stop sharing "Barks"?', expect.any(String), 'Stop Sharing');

      const req = httpMock.expectOne(SHARE_URL);
      expect(req.request.method).toBe('DELETE');
      const detector = { ...plain, sharing: null };
      req.flush({ ok: true, message: "Stopped sharing 'Barks' through Shared folder.", detector });
      await settleZoneless(fixture);
      expect(state.patchDetector).toHaveBeenCalledWith('det1', detector);
      expect(closed).toBe(1);
    });

    it('stops nothing when the confirm is declined', async () => {
      vi.spyOn(TestBed.inject(VtDialogService), 'confirmDestructive').mockResolvedValue(false);
      await openShared(['carol']);
      (el().querySelector('.share-stop-btn') as HTMLButtonElement).click();
      await settleZoneless(fixture);
      httpMock.expectNone(SHARE_URL);
      expect(closed).toBe(0);
    });

    it('says so when the container cannot be read, and still offers Stop Sharing', async () => {
      await open(shared);
      httpMock
        .expectOne(VIEW_URL)
        .flush({ message: 'No such container', error_code: 'remote_not_found' }, { status: 404, statusText: 'Not Found' });
      await settleZoneless(fixture);
      expect(el().querySelector('.error-text')!.textContent).toContain('No such container');
      expect(publicBox().disabled).toBe(true);
      expect(el().querySelector('.share-stop-btn')).not.toBeNull();
    });
  });

  it('opens read-only with the note, and only Close, when the container is not in use here', async () => {
    const note = 'Shared through old_service, which is not this server’s sharing source; it is local only here.';
    await open({ ...shared, sharing_note: note });
    httpMock.expectNone(VIEW_URL);
    expect(el().querySelector('.share-note')!.textContent).toContain(note);
    expect(el().querySelector('#share-recipient')).toBeNull();
    expect(el().querySelector('.share-public')).toBeNull();
    expect(el().querySelector('.share-stop-btn')).toBeNull();
    const buttons = Array.from(el().querySelectorAll('.modal-footer button')).map((b) => b.textContent!.trim());
    expect(buttons).toEqual(['Close']);
  });
});
