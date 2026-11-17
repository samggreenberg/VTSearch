import { TestBed } from '@angular/core/testing';
import { computed } from '@angular/core';
import { HttpTestingController, TestRequest } from '@angular/common/http/testing';

import { BlurStateService } from './blur-state.service';
import { ActiveContextService } from './active-context.service';
import { SettingsStateService } from './settings-state.service';
import { activeContextInterceptor } from '../interceptors/active-context.interceptor';
import { provideHttpTesting } from '../testing/test-providers';
import { makeSettingsStateStub } from '../testing/mocks';
import type { AppSettings } from '../generated/api-client/models/app-settings';

const BLURRED = '/api/medias/blurred';
const WITH_RULES = { blur_rules: [{ field: 'nsfw', op: 'truthy' }] } as AppSettings;
const NO_RULES = { blur_rules: [] } as AppSettings;

describe('BlurStateService', () => {
  let service: BlurStateService;
  let httpMock: HttpTestingController;
  let settings: ReturnType<typeof makeSettingsStateStub>['settings'];

  beforeEach(() => {
    const settingsStub = makeSettingsStateStub();
    settings = settingsStub.settings;
    TestBed.configureTestingModule({
      providers: [
        ...provideHttpTesting(activeContextInterceptor),
        { provide: SettingsStateService, useValue: settingsStub.stub },
      ],
    });
    service = TestBed.inject(BlurStateService);
    httpMock = TestBed.inject(HttpTestingController);
    // An active pair, so a request that forgot to name its dataset would show.
    TestBed.inject(ActiveContextService).setActivePair('active-ds', 'active-det');
  });

  afterEach(() => {
    vi.useRealTimers();
    httpMock.verify();
  });

  /** The pending read of *datasetId*'s blurred set. */
  function expectRead(datasetId: string): TestRequest {
    return httpMock.expectOne((r) => r.url === BLURRED && r.headers.get('X-Dataset-Id') === datasetId);
  }

  function expectWrite(datasetId: string, id: number): TestRequest {
    return httpMock.expectOne(
      (r) => r.url === `/api/medias/${id}/blur` && r.headers.get('X-Dataset-Id') === datasetId,
    );
  }

  describe('before a dataset’s set arrives (fail closed)', () => {
    it('reads every image blurred while the instance has blur rules', () => {
      settings.set(WITH_RULES);
      service.retain('ds');
      expect(service.isBlurred('ds', 1)).toBe(true);
      expect(service.isBlurred('ds', 99)).toBe(true);
      expectRead('ds').flush({ ids: [] });
      expect(service.isBlurred('ds', 1)).toBe(false);
    });

    it('reads every image blurred while the settings that would name the rules have not loaded', () => {
      service.retain('ds');
      expect(service.isBlurred('ds', 1)).toBe(true);
      expectRead('ds').flush({ ids: [] });
    });

    it('reads images sharp on an instance with no blur rules, where only an override blurs', () => {
      settings.set(NO_RULES);
      service.retain('ds');
      expect(service.isBlurred('ds', 1)).toBe(false);
      expectRead('ds').flush({ ids: [1] });
      expect(service.isBlurred('ds', 1)).toBe(true);
    });

    it('stays blurred when the first read fails, until a later one lands', async () => {
      vi.useFakeTimers();
      settings.set(WITH_RULES);
      service.retain('ds');
      expectRead('ds').flush(null, { status: 409, statusText: 'Conflict' });
      expect(service.isBlurred('ds', 1)).toBe(true);

      await vi.advanceTimersByTimeAsync(5_000);
      expectRead('ds').flush({ ids: [2] });
      expect(service.isBlurred('ds', 1)).toBe(false);
      expect(service.isBlurred('ds', 2)).toBe(true);
    });

    it('reads a dataset nobody retained as wholly blurred, since nothing loads it', () => {
      settings.set(WITH_RULES);
      expect(service.isBlurred('never-retained', 1)).toBe(true);
      httpMock.expectNone(BLURRED);
    });
  });

  describe('loading', () => {
    it('reads the named dataset’s set with no detector, whichever pair is active', () => {
      service.retain('sibling-ds');
      const req = expectRead('sibling-ds');
      expect(req.request.method).toBe('GET');
      expect(req.request.headers.has('X-Detector-Id')).toBe(false);
      req.flush({ ids: [] });
    });

    it('keys the sets by dataset', () => {
      settings.set(WITH_RULES);
      service.retain('a');
      service.retain('b');
      expectRead('a').flush({ ids: [1, 2] });
      expect(service.isBlurred('a', 1)).toBe(true);
      expect(service.isBlurred('a', 3)).toBe(false);
      // `b` has not answered, so it is still closed: `a`'s answer says nothing about it.
      expect(service.isBlurred('b', 3)).toBe(true);
      expectRead('b').flush({ ids: [3] });
      expect(service.isBlurred('b', 1)).toBe(false);
      expect(service.isBlurred('b', 3)).toBe(true);
    });

    it('is a signal read: a computed over it follows the load and the click', () => {
      settings.set(WITH_RULES);
      service.retain('ds');
      const shown = computed(() => service.isBlurred('ds', 5));
      expect(shown()).toBe(true);
      expectRead('ds').flush({ ids: [] });
      expect(shown()).toBe(false);
      service.setBlurred('ds', 5, true);
      expect(shown()).toBe(true);
      expectWrite('ds', 5).flush({ ok: true, id: 5, blurred: true });
    });

    it('polls, so another client’s click arrives', async () => {
      vi.useFakeTimers();
      settings.set(NO_RULES);
      service.retain('ds');
      expectRead('ds').flush({ ids: [] });
      expect(service.isBlurred('ds', 4)).toBe(false);

      await vi.advanceTimersByTimeAsync(5_000);
      expectRead('ds').flush({ ids: [4] });
      expect(service.isBlurred('ds', 4)).toBe(true);
    });

    it('shares one poll between the sites that retain a dataset', () => {
      service.retain('ds');
      service.retain('ds');
      expectRead('ds').flush({ ids: [] });
      httpMock.expectNone(BLURRED);
    });

    it('keeps the poll through a short release, and stops it once the grace is up', async () => {
      vi.useFakeTimers();
      const release = service.retain('ds');
      expectRead('ds').flush({ ids: [] });

      release();
      release(); // a second call is a no-op, not a second release
      await vi.advanceTimersByTimeAsync(2_000);
      const again = service.retain('ds');
      httpMock.expectNone(BLURRED); // still polling: no fresh start

      again(); // t=2s: the grace runs to t=12s, and the poll ticks on until then
      await vi.advanceTimersByTimeAsync(3_000);
      expectRead('ds').flush({ ids: [] });
      await vi.advanceTimersByTimeAsync(5_000);
      expectRead('ds').flush({ ids: [] });
      await vi.advanceTimersByTimeAsync(120_000);
      httpMock.expectNone(BLURRED);
    });

    it('ignores a dataset id that names nothing', () => {
      service.retain('')();
      httpMock.expectNone(BLURRED);
      settings.set(WITH_RULES);
      expect(service.isBlurred('', 1)).toBe(true);
      settings.set(NO_RULES);
      expect(service.isBlurred('', 1)).toBe(false);
    });
  });

  describe('setBlurred', () => {
    beforeEach(() => {
      settings.set(WITH_RULES);
      service.retain('ds');
      expectRead('ds').flush({ ids: [1] });
    });

    it('shows the click at once, and keeps it once the server confirms', () => {
      service.setBlurred('ds', 1, false);
      service.setBlurred('ds', 2, true);
      expect(service.isBlurred('ds', 1)).toBe(false);
      expect(service.isBlurred('ds', 2)).toBe(true);

      const unblur = expectWrite('ds', 1);
      expect(unblur.request.method).toBe('PUT');
      expect(unblur.request.body).toEqual({ blurred: false });
      expect(unblur.request.headers.has('X-Detector-Id')).toBe(false);
      unblur.flush({ ok: true, id: 1, blurred: false });
      expectWrite('ds', 2).flush({ ok: true, id: 2, blurred: true });

      expect(service.isBlurred('ds', 1)).toBe(false);
      expect(service.isBlurred('ds', 2)).toBe(true);
    });

    it('rolls a failed write back and reads the set again', () => {
      service.setBlurred('ds', 1, false);
      expect(service.isBlurred('ds', 1)).toBe(false);
      expectWrite('ds', 1).flush(null, { status: 500, statusText: 'Server Error' });
      expect(service.isBlurred('ds', 1)).toBe(true);
      expectRead('ds').flush({ ids: [1] });
      expect(service.isBlurred('ds', 1)).toBe(true);
    });

    it('lets the later of two clicks on one item win, whichever answer lands first', () => {
      service.setBlurred('ds', 7, true);
      service.setBlurred('ds', 7, false);
      const [first, second] = httpMock.match((r) => r.url === '/api/medias/7/blur');
      second.flush({ ok: true, id: 7, blurred: false });
      first.flush({ ok: true, id: 7, blurred: true });
      expect(service.isBlurred('ds', 7)).toBe(false);
    });

    it('keeps a later click shown when an earlier one on the item fails', () => {
      service.setBlurred('ds', 7, true);
      service.setBlurred('ds', 7, false);
      const [first, second] = httpMock.match((r) => r.url === '/api/medias/7/blur');
      first.flush(null, { status: 500, statusText: 'Server Error' });
      expect(service.isBlurred('ds', 7)).toBe(false);
      expectRead('ds').flush({ ids: [1, 7] });
      expect(service.isBlurred('ds', 7)).toBe(false);
      second.flush({ ok: true, id: 7, blurred: false });
      expect(service.isBlurred('ds', 7)).toBe(false);
    });
  });

  it('does not let a read issued before a confirmation undo it; a read issued after retires it', async () => {
    vi.useFakeTimers(); // before the poll starts, so its ticks are on the fake clock
    settings.set(WITH_RULES);
    service.retain('ds');
    expectRead('ds').flush({ ids: [1] });
    await vi.advanceTimersByTimeAsync(5_000);
    const stale = expectRead('ds'); // in flight across the click

    service.setBlurred('ds', 1, false);
    expectWrite('ds', 1).flush({ ok: true, id: 1, blurred: false });
    stale.flush({ ids: [1] }); // read before the write landed
    expect(service.isBlurred('ds', 1)).toBe(false);

    // The next read was issued after the confirmation, so it is the truth,
    // even when another client has blurred the item again since.
    await vi.advanceTimersByTimeAsync(5_000);
    expectRead('ds').flush({ ids: [1] });
    expect(service.isBlurred('ds', 1)).toBe(true);
  });

  it('shows a click before the set arrives, and keeps the rest of the dataset closed', () => {
    settings.set(WITH_RULES);
    service.retain('ds');
    service.setBlurred('ds', 3, false);
    expect(service.isBlurred('ds', 3)).toBe(false);
    expect(service.isBlurred('ds', 4)).toBe(true);
    expectWrite('ds', 3).flush({ ok: true, id: 3, blurred: false });
    expect(service.isBlurred('ds', 3)).toBe(false);
    expect(service.isBlurred('ds', 4)).toBe(true);
    expectRead('ds').flush({ ids: [] });
    expect(service.isBlurred('ds', 4)).toBe(false);
  });

  describe('blurredIn (a test over one read, for the Browse canvas)', () => {
    it('answers as isBlurred does: closed before the set arrives, then the set with clicks over it', () => {
      settings.set(WITH_RULES);
      service.retain('ds');
      const before = service.blurredIn('ds');
      expect([before(1), before(2)]).toEqual([true, true]);

      expectRead('ds').flush({ ids: [2] });
      service.setBlurred('ds', 3, true);
      const after = service.blurredIn('ds');
      expect([after(1), after(2), after(3)]).toEqual([false, true, true]);
      // The test answers for the state it was made from, not the live one.
      expect(before(1)).toBe(true);
      expectWrite('ds', 3).flush({ ok: true, id: 3, blurred: true });
    });

    it('answers for no dataset as the fail-closed rule says', () => {
      settings.set(WITH_RULES);
      expect(service.blurredIn('')(1)).toBe(true);
      settings.set(NO_RULES);
      expect(service.blurredIn('')(1)).toBe(false);
    });

    it('is a signal read: a computed over it follows the set arriving and each click', () => {
      settings.set(WITH_RULES);
      service.retain('ds');
      const test = computed(() => service.blurredIn('ds'));
      const first = test();
      expect(first(1)).toBe(true);

      expectRead('ds').flush({ ids: [] });
      expect(test()).not.toBe(first);
      expect(test()(1)).toBe(false);

      service.setBlurred('ds', 1, true);
      expect(test()(1)).toBe(true);
      expectWrite('ds', 1).flush({ ok: true, id: 1, blurred: true });
    });
  });
});
