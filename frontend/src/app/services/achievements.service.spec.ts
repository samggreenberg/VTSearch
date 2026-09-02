import { signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';

import { AchievementsService } from './achievements.service';
import { SettingsStateService } from './settings-state.service';
import { configureZoneless } from '../testing/zoneless-testbed';
import type { AchievementState } from '../generated/api-client/models/achievement-state';
import type { PendingAnnouncement } from '../generated/api-client/models/pending-announcement';
import type { AppSettings } from '../generated/api-client/models/app-settings';

/**
 * AchievementsService is the whole feature's data spine: it GETs
 * /api/achievements, exposes the snapshot as an observable, queues one-time
 * unlock toasts (deduped in-session, marked server-side), and drives
 * acknowledge / phrase-check side effects. These tests pin all of that against
 * a mocked HTTP backend, plus the settings-toggle kill switch that freezes the
 * feature to an empty state.
 */
describe('AchievementsService', () => {
  let service: AchievementsService;
  let httpMock: HttpTestingController;
  // Stand-in for SettingsStateService: the service reads only settingsSignal().
  let enableSetting: ReturnType<typeof signal<AppSettings | null>>;

  function makeAnnouncement(over: Partial<PendingAnnouncement> = {}): PendingAnnouncement {
    return {
      id: 'votes_cast',
      name: 'Prolific Voter',
      icon: 'thumb',
      threshold: 10,
      tier_idx: 0,
      tier_name: 'Bronze',
      ...over,
    };
  }

  function makeState(over: Partial<AchievementState> = {}): AchievementState {
    return {
      tier_names: ['Bronze', 'Silver', 'Gold', 'Platinum'],
      achievements: [],
      pending_announcements: [],
      pending_toasts: [],
      docs: [],
      media_types: [],
      hours: [],
      ...over,
    };
  }

  /** Answer the GET /api/achievements that refresh() issues. */
  function flushGet(state: AchievementState): void {
    httpMock.expectOne('/api/achievements').flush(state);
  }

  beforeEach(() => {
    enableSetting = signal<AppSettings | null>(null);
    configureZoneless({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: SettingsStateService, useValue: { settingsSignal: enableSetting } },
      ],
    });
    service = TestBed.inject(AchievementsService);
    // Flush the disabled-tracking effect so `disabled` is initialized.
    TestBed.tick();
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('starts with an empty snapshot before any refresh', () => {
    expect(service.snapshot.achievements).toEqual([]);
    expect(service.snapshot.pending_announcements).toEqual([]);
  });

  it('refresh() GETs the state and publishes it on state$', () => {
    const seen: AchievementState[] = [];
    service.state.subscribe((s) => seen.push(s));

    service.refresh();
    const state = makeState({ tier_names: ['Bronze'] });
    flushGet(state);

    expect(service.snapshot.tier_names).toEqual(['Bronze']);
    // BehaviorSubject replays the initial EMPTY_STATE, then the fetched one.
    expect(seen.at(-1)).toEqual(state);
  });

  it('coalesces concurrent refreshes into a single request', () => {
    service.refresh();
    service.refresh(); // second call is a no-op while the first is in flight
    // Exactly one request outstanding; verify() in afterEach would flag a leak.
    flushGet(makeState());
    // A later refresh, after the first settled, issues a fresh request.
    service.refresh();
    flushGet(makeState());
  });

  it('falls back to the empty state when the fetch errors', () => {
    service.refresh();
    service.state.subscribe();
    httpMock
      .expectOne('/api/achievements')
      .flush({ message: 'boom' }, { status: 500, statusText: 'Server Error' });
    expect(service.snapshot.achievements).toEqual([]);
  });

  it('emits pending toasts once and marks each toasted', () => {
    const toasts: PendingAnnouncement[] = [];
    service.unlocks.subscribe((p) => toasts.push(p));

    const toast = makeAnnouncement({ id: 'votes_cast', tier_idx: 1 });
    service.refresh();
    flushGet(makeState({ pending_toasts: [toast] }));

    expect(toasts).toEqual([toast]);
    // The service fires a mark-toasted POST for the emitted toast.
    const mark = httpMock.expectOne('/api/achievements/votes_cast/mark-toasted');
    expect(mark.request.method).toBe('POST');
    expect(mark.request.body).toEqual({ tier_idx: 1 });
    mark.flush({});
  });

  it('does not re-emit a toast already emitted this session', () => {
    const toasts: PendingAnnouncement[] = [];
    service.unlocks.subscribe((p) => toasts.push(p));
    const toast = makeAnnouncement({ id: 'votes_cast', tier_idx: 0 });

    service.refresh();
    flushGet(makeState({ pending_toasts: [toast] }));
    httpMock.expectOne('/api/achievements/votes_cast/mark-toasted').flush({});

    // Server hasn't dropped it from pending_toasts yet, but the in-session
    // guard must prevent a duplicate pop.
    service.refresh();
    flushGet(makeState({ pending_toasts: [toast] }));

    expect(toasts).toEqual([toast]);
    httpMock.expectNone('/api/achievements/votes_cast/mark-toasted');
  });

  it('hasPending$ tracks the pending-announcement count', () => {
    const flags: boolean[] = [];
    service.hasPending$.subscribe((v) => flags.push(v));

    service.refresh();
    flushGet(makeState({ pending_announcements: [makeAnnouncement()] }));

    expect(flags.at(-1)).toBe(true);
  });

  it('acknowledge() POSTs then refreshes state', () => {
    service.acknowledge('votes_cast', 2);
    const ack = httpMock.expectOne('/api/achievements/votes_cast/acknowledge');
    expect(ack.request.method).toBe('POST');
    expect(ack.request.body).toEqual({ tier_idx: 2 });
    ack.flush({});
    // The success handler chains a refresh.
    flushGet(makeState());
  });

  it('acknowledgeAll() acknowledges every pending announcement', () => {
    service.refresh();
    flushGet(
      makeState({
        pending_announcements: [
          makeAnnouncement({ id: 'votes_cast', tier_idx: 0 }),
          makeAnnouncement({ id: 'find_media', tier_idx: 1 }),
        ],
      }),
    );

    service.acknowledgeAll();
    httpMock.expectOne('/api/achievements/votes_cast/acknowledge').flush({});
    httpMock.expectOne('/api/achievements/find_media/acknowledge').flush({});
    // Each acknowledge chains its own refresh; the first wins, the rest coalesce.
    flushGet(makeState());
  });

  it('checkPhrase() POSTs the phrase and refreshes on a fresh correct match', () => {
    let result: unknown;
    service.checkPhrase('all systems nominal').subscribe((r) => (result = r));

    const req = httpMock.expectOne('/api/achievements/check-phrase');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ phrase: 'all systems nominal' });
    req.flush({ matched: true, doc_id: 'readme', doc_name: 'README', already_read: false });

    // A newly credited doc triggers a state refresh.
    flushGet(makeState());
    expect(result).toEqual({
      matched: true,
      doc_id: 'readme',
      doc_name: 'README',
      already_read: false,
    });
  });

  it('checkPhrase() does not refresh when the doc was already read', () => {
    service.checkPhrase('old phrase').subscribe();
    httpMock
      .expectOne('/api/achievements/check-phrase')
      .flush({ matched: true, doc_id: 'readme', doc_name: 'README', already_read: true });
    // No follow-up GET: already-credited phrases don't change counters.
    httpMock.expectNone('/api/achievements');
  });

  it('checkPhrase() resolves to a not-matched result when the request errors', () => {
    let result: { matched: boolean } | undefined;
    service.checkPhrase('???').subscribe((r) => (result = r));
    httpMock
      .expectOne('/api/achievements/check-phrase')
      .flush({ message: 'boom' }, { status: 500, statusText: 'Server Error' });
    expect(result).toEqual({ matched: false, doc_id: null, doc_name: null, already_read: false });
  });

  it('requestOpenPanel() emits on the open-panel request stream', () => {
    let opened = 0;
    service.openPanelRequest$.subscribe(() => (opened += 1));
    service.requestOpenPanel();
    expect(opened).toBe(1);
  });

  it('docRawUrl() builds an encoded raw-doc URL', () => {
    expect(service.docRawUrl('user/getting started')).toBe(
      '/api/achievements/docs/user%2Fgetting%20started/raw',
    );
  });

  describe('when achievements are disabled in settings', () => {
    it('refresh() short-circuits to the empty state without hitting the network', () => {
      enableSetting.set({ enable_achievements: false } as AppSettings);
      TestBed.tick();

      service.refresh();
      httpMock.expectNone('/api/achievements');
      expect(service.snapshot.achievements).toEqual([]);
    });

    it('clears cached state and the emitted-toast guard when the toggle flips off', () => {
      // Populate state and emit a toast while enabled.
      const toast = makeAnnouncement({ id: 'votes_cast', tier_idx: 0 });
      service.refresh();
      flushGet(makeState({ achievements: [], pending_toasts: [toast] }));
      httpMock.expectOne('/api/achievements/votes_cast/mark-toasted').flush({});

      // Flip disabled: cached state resets to empty.
      enableSetting.set({ enable_achievements: false } as AppSettings);
      TestBed.tick();
      expect(service.snapshot).toEqual(service.snapshot); // sanity: no throw
      expect(service.snapshot.pending_toasts).toEqual([]);

      // Re-enable and refresh: the same toast pops again because the guard was
      // cleared (server counters are wiped on a re-enable).
      const toasts: PendingAnnouncement[] = [];
      service.unlocks.subscribe((p) => toasts.push(p));
      enableSetting.set({ enable_achievements: true } as AppSettings);
      TestBed.tick();
      service.refresh();
      flushGet(makeState({ pending_toasts: [toast] }));
      httpMock.expectOne('/api/achievements/votes_cast/mark-toasted').flush({});
      expect(toasts).toEqual([toast]);
    });
  });
});
