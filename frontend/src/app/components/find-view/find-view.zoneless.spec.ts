import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';

import { FindViewComponent } from './find-view.component';
import { ActiveContextService } from '../../services/active-context.service';
import { SortStateService } from '../../services/sort-state.service';
import { BrowseSubsetService } from '../../services/browse-subset.service';
import { MediaPrefetchService } from '../../services/media-prefetch.service';
import { MediaStateService } from '../../services/media-state.service';
import { VoteStateService } from '../../services/vote-state.service';
import { VoteHistoryService } from '../../services/vote-history.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleResource, settleZoneless } from '../../testing/settle-resource';
import { provideHttpTesting } from '../../testing/test-providers';
import { FLOOR_STATES, lineFloor, wireFloor } from '../../testing/line-floor';

/**
 * Zoneless staleness canary for the Find view.
 * Phase 2.5 signalized find-view's own subscribe/effect
 * written template-bound state (`datasetName`, `gridGoalWidthLeft`, …, and the
 * `unverifiedSortOrder` computed) AND signalized the shared SortStateService /
 * VoteStateService it binds, so its `sortState.sortBusy`-style getter bindings
 * repaint under zoneless with no per-consumer bridge.
 *
 * Both tests run under a zoneless `TestBed`, drive state through the *production
 * channel* with NO manual `detectChanges()`, then assert on the rendered DOM:
 *  - the dataset name written from the un-bound `/api/dataset/status` subscribe
 *    (a local signal), and
 *  - the scoring overlay gated on `@if (sortState.sortBusy)`, driven by a
 *    `SortStateService` setter — proving the signal-backed service repaints a
 *    getter-bound view (the Phase 2.5 win).
 */
describe('FindViewComponent (zoneless canary)', () => {
  let fixture: ComponentFixture<FindViewComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await configureZoneless({
      imports: [FindViewComponent],
      providers: [...provideHttpTesting(), provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(FindViewComponent);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    // `fixture.destroy()` runs the component's own `ngOnDestroy` *and*
    // destroys the component-provided `PairScopeService`, which is what
    // cancels the in-flight pair-scoped requests (a bare `ngOnDestroy()` call
    // no longer does — the view stopped firing the scope by hand). Destroy
    // first, then drain; a cancelled request cannot be flushed.
    fixture.destroy();
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush([]);
    });
  });

  // Drain find-view's init loads, holding back only the dataset-status response
  // that the first test asserts. The medias/settings reads ride promise-based
  // `rxResource` loaders that issue their GET on a microtask, so drain with
  // `settleResource()` (NOT `whenStable()`, which deadlocks on a loading
  // resource) across a few cycles.
  async function flushInit(): Promise<void> {
    TestBed.tick();
    for (let i = 0; i < 3; i++) {
      await settleResource();
      httpMock.match('/api/medias/ids').forEach((req) =>
        req.flush([{ id: 1, media_type: 'audio' }]),
      );
      httpMock.match('/api/votes').forEach((req) =>
        req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} }),
      );
      httpMock.match('/api/settings').forEach((req) => req.flush({ volume: 0.8 }));
      httpMock.match('/api/min-precision').forEach((req) => req.flush({ min_precision: 0.5 }));
      httpMock.match('/api/media-types').forEach((req) => req.flush({ media_types: [] }));
      httpMock.match('/api/embedders').forEach((req) => req.flush([]));
    }
  }

  it('renders the dataset name pushed from the /api/dataset/status subscribe, no manual detectChanges', async () => {
    await flushInit();
    await settleZoneless(fixture);

    expect(fixture.nativeElement.querySelector('.dataset-name')).toBeNull();

    // Production channel: the un-bound dataset-status subscribe writes the
    // `datasetName` signal.
    httpMock.expectOne('/api/dataset/status').flush({ display_name: 'Find Canary' });
    await settleZoneless(fixture);

    const nameEl = fixture.nativeElement.querySelector('.dataset-name');
    expect(nameEl).not.toBeNull();
    expect(nameEl!.textContent).toContain('Find Canary');
  });

  it('toggles the scoring overlay from a SortStateService setter, no manual detectChanges', async () => {
    await flushInit();
    httpMock.match('/api/dataset/status').forEach((req) => req.flush({ display_name: 'x' }));
    await settleZoneless(fixture);

    const sortState = TestBed.inject(SortStateService);
    // Not busy after init → the `@if (sortState.sortBusy)` overlay is absent.
    expect(fixture.nativeElement.querySelector('.find-wait-overlay')).toBeNull();

    // A signal-backed service setter, called outside any bound handler, must
    // schedule CD for the getter-bound `@if` — that is the Phase 2.5 guarantee.
    sortState.setSortBusy(true);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.find-wait-overlay')).not.toBeNull();

    sortState.setSortBusy(false);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.find-wait-overlay')).toBeNull();
  });

  // O12 (issue #2373): at 1000+ items the scoring overlay must surface real
  // progress — a determinate bar plus a thousands-separated "N / total" count
  // and an ETA chip — through the same SortStateService setters the find$ SSE
  // subscription writes. Verified live against a 1,200-item dataset; a warm
  // run finishes sub-second, so this pins the template contract the live run
  // is too fast to observe.
  it('renders a determinate bar, formatted count, and ETA for a large-dataset scoring run', async () => {
    await flushInit();
    httpMock.match('/api/dataset/status').forEach((req) => req.flush({ display_name: 'x' }));
    await settleZoneless(fixture);

    const sortState = TestBed.inject(SortStateService);
    sortState.setSortBusy(true);
    sortState.setSortStatus('Scoring 1200 items…');
    sortState.setSortProgress(476, 1200, 0.79, 42);
    await settleZoneless(fixture);

    const overlay = fixture.nativeElement.querySelector('.find-wait-overlay');
    expect(overlay).not.toBeNull();
    // The whole-job `overall` fraction drives a determinate bar.
    const track = overlay!.querySelector('[role="progressbar"]') as HTMLElement;
    expect(track.getAttribute('aria-valuenow')).toBe('0.79');
    expect(track.getAttribute('aria-valuemax')).toBe('1');
    const fill = overlay!.querySelector('.progress-fill') as HTMLElement;
    expect(fill.className).not.toContain('indeterminate');
    // The count renders with a thousands separator so 1200 reads as 1,200.
    expect(overlay!.querySelector('.find-wait-count')!.textContent).toContain('476 / 1,200');
    expect(overlay!.querySelector('.find-wait-eta')!.textContent).toContain('sec');
  });

  // Find/Train share the singleton SortStateService, so a fresh entry still
  // holds the previous session's ranking. Against a smaller dataset those stale
  // ids fire a storm of image 404s; ngOnInit resets the ranking before loading.
  it('clears the previous session ranking on a fresh entry', async () => {
    const sortState = TestBed.inject(SortStateService);
    // Seed a ranking from a "previous" (larger) dataset before ngOnInit runs.
    sortState.setSortResults([{ id: 999, score: 0.9 }], 0.5);
    expect(sortState.sortOrder?.length).toBe(1);

    await flushInit();
    httpMock.match('/api/dataset/status').forEach((req) => req.flush({ display_name: 'x' }));
    await settleZoneless(fixture);

    // Reset happened before loadMedias(), so no stale id survives to fire a 404.
    expect(sortState.sortOrder ?? []).toEqual([]);
  });

  // Returning from the Browser is the exception: the preserved ranking and the
  // just-recorded verifications are exactly what we keep, so the reset is
  // skipped.
  it('preserves the ranking when returning from the Browser', async () => {
    const sortState = TestBed.inject(SortStateService);
    const browseSubset = TestBed.inject(BrowseSubsetService);
    sortState.setSortResults([{ id: 999, score: 0.9 }], 0.5);
    browseSubset.markReturningToFind();

    await flushInit();
    httpMock.match('/api/dataset/status').forEach((req) => req.flush({ display_name: 'x' }));
    await settleZoneless(fixture);

    expect(sortState.sortOrder?.map((s) => s.id)).toEqual([999]);
  });
});

/**
 * Issue #2921: a find-label scoring run outlives the pair it was started for.
 * Scoring takes minutes on a large dataset, so switching the active
 * (dataset, detector) pair mid-run used to leave two live subscriptions racing:
 * whichever landed last installed its ranking + threshold into the *active*
 * context, and `advanceToBoundary()` then selected a media id that need not
 * exist in the new dataset. Even the old response landing first was harmful —
 * its `finalize()` dropped the wait overlay while the new run was still going.
 * The run is now scoped to the pair (`PairScopeService`), so a switch tears it down.
 */
describe('FindViewComponent (pair-switch supersession)', () => {
  let fixture: ComponentFixture<FindViewComponent>;
  let httpMock: HttpTestingController;
  let activeContext: ActiveContextService;

  beforeEach(async () => {
    await configureZoneless({
      imports: [FindViewComponent],
      providers: [...provideHttpTesting(), provideRouter([])],
    }).compileComponents();

    activeContext = TestBed.inject(ActiveContextService);
    // A detector must be active before ngOnInit or `runFindLabel` no-ops. Set
    // it *before* creating the component so the pair$ replay this seeds is the
    // subscription's skipped first emission, not a spurious reload.
    activeContext.setActivePair('ds1', 'det1');

    fixture = TestBed.createComponent(FindViewComponent);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    // `fixture.destroy()` runs the component's own `ngOnDestroy` *and*
    // destroys the component-provided `PairScopeService`, which is what
    // cancels the in-flight pair-scoped requests (a bare `ngOnDestroy()` call
    // no longer does — the view stopped firing the scope by hand). Destroy
    // first, then drain; a cancelled request cannot be flushed.
    fixture.destroy();
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush([]);
    });
  });

  // Same drain as the canary above, plus the dataset-status read (no assertion
  // rides it here). The dataset is *images*: this spec is the only find-view
  // one that installs a ranking, which auto-selects the top item into the
  // centre viewer — and the audio player's waveform-decode path needs Web Audio,
  // which jsdom does not implement.
  async function flushInit(): Promise<void> {
    TestBed.tick();
    for (let i = 0; i < 3; i++) {
      await settleResource();
      httpMock.match('/api/medias/ids').forEach((req) =>
        req.flush([{ id: 1, media_type: 'image' }]),
      );
      httpMock.match('/api/votes').forEach((req) =>
        req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} }),
      );
      httpMock.match('/api/settings').forEach((req) => req.flush({ volume: 0.8 }));
      httpMock.match('/api/min-precision').forEach((req) => req.flush({ min_precision: 0.5 }));
      httpMock.match('/api/media-types').forEach((req) => req.flush({ media_types: [] }));
      httpMock.match('/api/embedders').forEach((req) => req.flush([]));
      httpMock.match('/api/dataset/status').forEach((req) => req.flush({ display_name: 'ds' }));
    }
  }

  it('cancels the previous pair\'s scoring run and keeps the overlay on the new one', async () => {
    const sortState = TestBed.inject(SortStateService);
    await flushInit();
    await settleZoneless(fixture);

    // Pair one's scoring run is in flight behind the wait overlay.
    const staleScore = httpMock.expectOne('/api/find-label');
    expect(staleScore.cancelled).toBe(false);
    expect(sortState.sortBusy).toBe(true);

    // The user switches pair while that run is still going.
    activeContext.setActivePair('ds2', 'det2');

    // The stale run is aborted client-side, so its ranking can never land in
    // the new context — and its finalize() did not drop the overlay, because
    // the fresh run re-armed it.
    expect(staleScore.cancelled).toBe(true);
    expect(sortState.sortBusy).toBe(true);

    // Only the new pair's ranking is installed.
    const freshScore = httpMock.expectOne('/api/find-label');
    freshScore.flush({ results: [{ id: 1, score: 0.9 }], threshold: 0.5 });
    await flushInit();
    await settleZoneless(fixture);

    expect(sortState.sortOrder?.map((s) => s.id)).toEqual([1]);
    expect(sortState.threshold).toBe(0.5);
    expect(sortState.sortBusy).toBe(false);
  });

  // The floor POST is deferred until the picker settles (issue #2973), and
  // that settle window is inside the pair scope too: a pick the user abandons
  // by switching pair must never be written into the pair they switched *to*,
  // whose own floor the reload has just re-seeded.
  it('drops a pending floor POST when the pair switches first', async () => {
    await flushInit();
    // Land the first pair's ranking so the picker isn't disabled by sortBusy.
    httpMock
      .expectOne('/api/find-label')
      .flush({ results: [{ id: 1, score: 0.9 }], threshold: 0.5 });
    await flushInit();
    await settleZoneless(fixture);

    vi.useFakeTimers();
    try {
      fixture.componentInstance.onMinPrecisionChange(0.9);
      // Still inside the settle window when the user switches pair.
      vi.advanceTimersByTime(50);
      activeContext.setActivePair('ds2', 'det2');
      vi.advanceTimersByTime(1000);

      httpMock.expectNone((req) => req.url === '/api/min-precision' && req.method === 'POST');
    } finally {
      vi.useRealTimers();
    }
  });
});

/**
 * Issue #2973, carried over from the Inclusion slider to the precision floor
 * (#4246): the picker emits a `change` per arrow key, so walking the floors
 * would leave several `POST /api/min-precision` requests in flight at once,
 * each installing its own threshold on arrival. A slow response for a floor the
 * user had already moved past could land *last* and overwrite the newer
 * threshold, snapping the green/red line (and the left/right split) back to a
 * floor that was no longer selected — with nothing to re-reconcile it until the
 * next pick. Picks funnel through one debounced `switchMap` pipeline, so only
 * the settled floor is sent and only its response is applied.
 */
describe('FindViewComponent (floor supersession)', () => {
  let fixture: ComponentFixture<FindViewComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await configureZoneless({
      imports: [FindViewComponent],
      providers: [...provideHttpTesting(), provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(FindViewComponent);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    vi.useRealTimers();
    // `fixture.destroy()` runs the component's own `ngOnDestroy` *and*
    // destroys the component-provided `PairScopeService`, which is what
    // cancels the in-flight pair-scoped requests (a bare `ngOnDestroy()` call
    // no longer does — the view stopped firing the scope by hand). Destroy
    // first, then drain; a cancelled request cannot be flushed.
    fixture.destroy();
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush([]);
    });
  });

  // Same drain as the canary above; no pair is active, so `runFindLabel` no-ops
  // and the only /api/min-precision traffic afterwards is the picker's.
  async function flushInit(): Promise<void> {
    TestBed.tick();
    for (let i = 0; i < 3; i++) {
      await settleResource();
      httpMock.match('/api/medias/ids').forEach((req) =>
        req.flush([{ id: 1, media_type: 'audio' }]),
      );
      httpMock.match('/api/votes').forEach((req) =>
        req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} }),
      );
      httpMock.match('/api/settings').forEach((req) => req.flush({ volume: 0.8 }));
      httpMock.match('/api/min-precision').forEach((req) => req.flush({ min_precision: 0.5 }));
      httpMock.match('/api/media-types').forEach((req) => req.flush({ media_types: [] }));
      httpMock.match('/api/embedders').forEach((req) => req.flush([]));
      httpMock.match('/api/dataset/status').forEach((req) => req.flush({ display_name: 'ds' }));
    }
  }

  /** Seed a ranking so a returned threshold has an order to be installed against. */
  function seedRanking(): SortStateService {
    const sortState = TestBed.inject(SortStateService);
    sortState.setSortResults([{ id: 1, score: 0.9 }], 0.5);
    return sortState;
  }

  it('coalesces a rapid walk through the floors into one POST for the settled one', async () => {
    await flushInit();
    await settleZoneless(fixture);
    const sortState = seedRanking();
    const component = fixture.componentInstance;

    vi.useFakeTimers();
    // Three floors in quick succession: the picker tracks every one of them...
    component.onMinPrecisionChange(0.1);
    vi.advanceTimersByTime(40);
    component.onMinPrecisionChange(0.5);
    vi.advanceTimersByTime(40);
    component.onMinPrecisionChange(0.9);
    expect(sortState.minPrecision).toBe(0.9);
    // ...but nothing is sent until the picker settles.
    httpMock.expectNone('/api/min-precision');

    vi.advanceTimersByTime(200);
    const req = httpMock.expectOne('/api/min-precision');
    expect(req.request.body).toEqual({ min_precision: 0.9 });
    req.flush({ ...wireFloor('confirmed', { minPrecision: 0.9 }), threshold: 0.7, n_returned: 1 });
    expect(sortState.threshold).toBe(0.7);
    expect(sortState.floor?.status).toBe('confirmed');
    expect(sortState.floor?.minPrecision).toBe(0.9);
  });

  it('cancels a superseded POST so its stale threshold can never land', async () => {
    await flushInit();
    await settleZoneless(fixture);
    const sortState = seedRanking();
    const component = fixture.componentInstance;

    vi.useFakeTimers();
    component.onMinPrecisionChange(0.1);
    vi.advanceTimersByTime(200);
    // The first POST is still in flight (a slow re-cut server-side) when the
    // user picks another floor.
    const stale = httpMock.expectOne('/api/min-precision');
    expect(stale.cancelled).toBe(false);

    component.onMinPrecisionChange(0.9);
    vi.advanceTimersByTime(200);

    // switchMap aborted the superseded request, so its threshold has no
    // subscriber left to install it however late it resolves.
    expect(stale.cancelled).toBe(true);
    const fresh = httpMock.expectOne('/api/min-precision');
    expect(fresh.request.body).toEqual({ min_precision: 0.9 });
    fresh.flush({ ...wireFloor('confirmed', { minPrecision: 0.9 }), threshold: 0.9, n_returned: 1 });
    expect(sortState.threshold).toBe(0.9);
  });

  it('keeps posting after a failed pick', async () => {
    await flushInit();
    await settleZoneless(fixture);
    const sortState = seedRanking();
    const component = fixture.componentInstance;

    vi.useFakeTimers();
    component.onMinPrecisionChange(0.5);
    vi.advanceTimersByTime(200);
    httpMock
      .expectOne('/api/min-precision')
      .flush({ message: 'boom' }, { status: 500, statusText: 'Server Error' });

    // The error is swallowed per-request, so the shared pipeline survives it.
    component.onMinPrecisionChange(0.1);
    vi.advanceTimersByTime(200);
    const retry = httpMock.expectOne('/api/min-precision');
    expect(retry.request.body).toEqual({ min_precision: 0.1 });
    retry.flush({ ...wireFloor('confirmed', { minPrecision: 0.1 }), threshold: 0.6, n_returned: 1 });
    expect(sortState.threshold).toBe(0.6);
  });

  it('sends nothing while the detector is still scoring', async () => {
    await flushInit();
    await settleZoneless(fixture);
    const sortState = seedRanking();
    sortState.setSortBusy(true);

    vi.useFakeTimers();
    fixture.componentInstance.onMinPrecisionChange(0.9);
    vi.advanceTimersByTime(1000);
    httpMock.expectNone('/api/min-precision');
    expect(sortState.minPrecision).toBe(0.5);
  });
});

/**
 * #3896 in Find: the boundary walk advances only once the vote POST is back,
 * the same shape as the Train view, so it warms the next images the same way.
 */
describe('FindViewComponent prefetching the next review images (#3896)', () => {
  let fixture: ComponentFixture<FindViewComponent>;
  let httpMock: HttpTestingController;
  let prefetch: ReturnType<typeof vi.spyOn>;

  // Descending by score; the cutoff at 0.5 sits between ids 2 and 3.
  const ranking = [
    { id: 1, score: 0.9 },
    { id: 2, score: 0.6 },
    { id: 3, score: 0.4 },
    { id: 4, score: 0.2 },
  ];
  const url = (id: number) =>
    TestBed.inject(ActiveContextService).mediaUrl(`/api/medias/${id}/image`);
  const lastCall = () => prefetch.mock.calls[prefetch.mock.calls.length - 1];

  beforeEach(async () => {
    await configureZoneless({
      imports: [FindViewComponent],
      providers: [...provideHttpTesting(), provideRouter([])],
    }).compileComponents();
    fixture = TestBed.createComponent(FindViewComponent);
    httpMock = TestBed.inject(HttpTestingController);
    prefetch = vi.spyOn(TestBed.inject(MediaPrefetchService), 'prefetch');

    TestBed.tick();
    for (let i = 0; i < 3; i++) {
      await settleResource();
      httpMock
        .match('/api/medias/ids')
        .forEach((req) => req.flush(ranking.map(({ id }) => ({ id, media_type: 'image' }))));
      httpMock.match('/api/votes').forEach((req) =>
        req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} }),
      );
      httpMock.match('/api/settings').forEach((req) => req.flush({ volume: 0.8 }));
      httpMock.match('/api/min-precision').forEach((req) => req.flush({ min_precision: 0.5 }));
      httpMock.match('/api/media-types').forEach((req) => req.flush({ media_types: [] }));
      httpMock.match('/api/embedders').forEach((req) => req.flush([]));
    }
    httpMock.match('/api/dataset/status').forEach((req) => req.flush({ display_name: 'x' }));
    await settleZoneless(fixture);
  });

  afterEach(() => {
    fixture.destroy();
    TestBed.inject(VoteStateService).stopPolling();
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush([]);
    });
  });

  it('warms the next two items of the boundary walk, alternating sides', () => {
    TestBed.inject(SortStateService).setSortResults(ranking, 0.5);
    // The walk's seed is the marginal positive, so `below` is served next.
    TestBed.inject(MediaStateService).selectMedia(2);
    // Mirror what advanceToBoundary leaves behind after taking `above`.
    (fixture.componentInstance as unknown as { nextFindSide: string }).nextFindSide = 'below';
    TestBed.tick();

    expect(lastCall()).toEqual([[url(3), url(1)], [url(2)]]);
  });

  it('retargets when an item is verified with the same item on screen', () => {
    TestBed.inject(SortStateService).setSortResults(ranking, 0.5);
    TestBed.inject(MediaStateService).selectMedia(2);
    (fixture.componentInstance as unknown as { nextFindSide: string }).nextFindSide = 'below';
    TestBed.tick();

    TestBed.inject(VoteStateService).setOptimisticVerified(3, true);
    TestBed.tick();

    expect(lastCall()).toEqual([[url(4), url(1)], [url(2)]]);
  });

  it('retargets when the cutoff moves', () => {
    const sortState = TestBed.inject(SortStateService);
    sortState.setSortResults(ranking, 0.5);
    TestBed.inject(MediaStateService).selectMedia(2);
    (fixture.componentInstance as unknown as { nextFindSide: string }).nextFindSide = 'below';
    TestBed.tick();

    // A floor change re-thresholds with 2 still on screen. Cut at 0.3:
    // above = {1, 2, 3}, below = {4}. `below` first → 4, then the nearest item
    // above the line that is not on screen → 3.
    sortState.setSortResults(ranking, 0.3);
    TestBed.tick();

    expect(lastCall()).toEqual([[url(4), url(3)], [url(2)]]);
  });
});

/**
 * #4306: every boundary-walk advance flips which side of the cutoff it serves,
 * so re-running it on `↑` could never land back on the item `↓` left.
 */
describe('FindViewComponent ↓ then ↑ (#4306)', () => {
  let fixture: ComponentFixture<FindViewComponent>;
  let httpMock: HttpTestingController;

  // Descending by score; the cutoff at 0.5 sits between ids 2 and 3.
  const ranking = [
    { id: 1, score: 0.9 },
    { id: 2, score: 0.6 },
    { id: 3, score: 0.4 },
    { id: 4, score: 0.2 },
  ];

  beforeEach(async () => {
    await configureZoneless({
      imports: [FindViewComponent],
      providers: [...provideHttpTesting(), provideRouter([])],
    }).compileComponents();
    fixture = TestBed.createComponent(FindViewComponent);
    httpMock = TestBed.inject(HttpTestingController);

    TestBed.tick();
    for (let i = 0; i < 3; i++) {
      await settleResource();
      httpMock
        .match('/api/medias/ids')
        .forEach((req) => req.flush(ranking.map(({ id }) => ({ id, media_type: 'image' }))));
      httpMock.match('/api/votes').forEach((req) =>
        req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} }),
      );
      httpMock.match('/api/settings').forEach((req) => req.flush({ volume: 0.8 }));
      httpMock.match('/api/min-precision').forEach((req) => req.flush({ min_precision: 0.5 }));
      httpMock.match('/api/media-types').forEach((req) => req.flush({ media_types: [] }));
      httpMock.match('/api/embedders').forEach((req) => req.flush([]));
    }
    httpMock.match('/api/dataset/status').forEach((req) => req.flush({ display_name: 'x' }));
    await settleZoneless(fixture);
  });

  afterEach(() => {
    fixture.destroy();
    TestBed.inject(VoteStateService).stopPolling();
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush([]);
    });
  });

  it('returns to the item the walk started from, not the other side of the line', () => {
    const component = fixture.componentInstance;
    const selected = () => TestBed.inject(MediaStateService).selectedId();
    TestBed.inject(SortStateService).setSortResults(ranking, 0.5);
    // Verifying 2 (the marginal positive) advanced below the line, to 3; the
    // next advance would serve the positive side again.
    TestBed.inject(VoteStateService).setOptimisticVerified(2, true);
    TestBed.inject(VoteHistoryService).record(2);
    TestBed.inject(MediaStateService).selectMedia(3);
    (component as unknown as { nextFindSide: string }).nextFindSide = 'above';
    TestBed.tick();

    component.onNavigate('back');
    expect(selected()).toBe(2);
    component.onNavigate('forward');
    expect(selected()).toBe(3);

    // With no walk left to end, `↑` is the boundary walk again.
    component.onNavigate('forward');
    expect(selected()).toBe(1);
  });
});

/**
 * #4247: when the precision floor promises nothing, find-label still returns a
 * cut - the default one (Inclusion 0) - with the floor's verdict beside it. Every
 * consumer of the cut keeps working on it: the boundary walk, the queue-empty
 * state, and the positive sets behind Browse / To Dataset / Export. Only the
 * line's label changes.
 */
describe('FindViewComponent with no precision promise (#4247)', () => {
  let fixture: ComponentFixture<FindViewComponent>;
  let httpMock: HttpTestingController;
  let sortState: SortStateService;

  // Descending by score; the fallback cut at 0.5 sits between ids 2 and 3.
  const ranking = [
    { id: 1, score: 0.9 },
    { id: 2, score: 0.6 },
    { id: 3, score: 0.4 },
    { id: 4, score: 0.2 },
  ];

  /** The private consumers under test, reached the way the prefetch spec reaches `nextFindSide`. */
  type Consumers = {
    unverifiedGoodIds(): number[];
    goodIds(): number[];
    advanceToBoundary(): void;
    nextFindSide: string;
  };
  const view = () => fixture.componentInstance as unknown as Consumers;

  /** Drain the init loads for *ids*, all images (the viewer's audio path needs Web Audio). */
  async function flushInit(ids: number[]): Promise<void> {
    TestBed.tick();
    for (let i = 0; i < 3; i++) {
      await settleResource();
      httpMock.match('/api/medias/ids').forEach((req) => req.flush(ids.map((id) => ({ id, media_type: 'image' }))));
      httpMock.match('/api/votes').forEach((req) =>
        req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} }),
      );
      httpMock.match('/api/settings').forEach((req) => req.flush({ volume: 0.8 }));
      httpMock.match('/api/min-precision').forEach((req) => req.flush({ min_precision: 0.5 }));
      httpMock.match('/api/media-types').forEach((req) => req.flush({ media_types: [] }));
      httpMock.match('/api/embedders').forEach((req) => req.flush([]));
      httpMock.match('/api/dataset/status').forEach((req) => req.flush({ display_name: 'ds' }));
    }
  }

  async function setUp(withPair: boolean): Promise<void> {
    await configureZoneless({
      imports: [FindViewComponent],
      providers: [...provideHttpTesting(), provideRouter([])],
    }).compileComponents();
    // With a pair active `runFindLabel` scores on init; without one it no-ops
    // and the ranking is installed straight into the sort state, as the
    // prefetch spec above does.
    if (withPair) TestBed.inject(ActiveContextService).setActivePair('ds1', 'det1');
    fixture = TestBed.createComponent(FindViewComponent);
    httpMock = TestBed.inject(HttpTestingController);
    sortState = TestBed.inject(SortStateService);
  }

  afterEach(() => {
    vi.useRealTimers();
    fixture.destroy();
    TestBed.inject(VoteStateService).stopPolling();
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush([]);
    });
  });

  describe('a scoring pass', () => {
    beforeEach(() => setUp(true));

    it.each(FLOOR_STATES)('installs the line of the kept set with its verdict when %s', async (status) => {
      await flushInit([1]);
      httpMock.expectOne('/api/find-label').flush({
        ok: true,
        results: [{ id: 1, score: 0.9 }],
        threshold: 0.5,
        floor: wireFloor(status),
        good_count: 1,
        bad_count: 0,
        detector_name: 'det',
      });
      await flushInit([1]);
      await settleZoneless(fixture);

      expect(sortState.threshold).toBe(0.5);
      expect(sortState.floor?.status).toBe(status);
      // The walk still seeds on the marginal positive. (One result draws no
      // line - nothing falls below it - so the label is pinned further down.)
      expect(TestBed.inject(MediaStateService).selectedId()).toBe(1);
    });
  });

  describe('the consumers of the cut', () => {
    beforeEach(async () => {
      await setUp(false);
      await flushInit(ranking.map(({ id }) => id));
      await settleZoneless(fixture);
    });

    it.each(FLOOR_STATES)('walk the boundary of the line when %s', (status) => {
      sortState.setSortResults(ranking, 0.5, lineFloor(status));
      const mediaState = TestBed.inject(MediaStateService);
      view().nextFindSide = 'above';
      view().advanceToBoundary();
      expect(mediaState.selectedId()).toBe(2);
      view().advanceToBoundary();
      expect(mediaState.selectedId()).toBe(3);
    });

    it.each(FLOOR_STATES)('empty the queue only once every item is verified, when %s', (status) => {
      sortState.setSortResults(ranking, 0.5, lineFloor(status));
      expect(fixture.componentInstance.queueEmpty()).toBe(false);
      const voteState = TestBed.inject(VoteStateService);
      ranking.forEach(({ id }) => voteState.setOptimisticVerified(id, true));
      expect(fixture.componentInstance.queueEmpty()).toBe(true);
    });

    it.each(FLOOR_STATES)('hand Browse / To Dataset / Export the positives above the line when %s', (status) => {
      sortState.setSortResults(ranking, 0.5, lineFloor(status));
      expect(view().unverifiedGoodIds()).toEqual([1, 2]);
      expect(view().goodIds()).toEqual([1, 2]);
    });

    it.each(FLOOR_STATES)('draw the line the same in the work queue when %s, naming the state only in its tooltip', async (status) => {
      sortState.setSortResults(ranking, 0.5, lineFloor(status));
      await settleZoneless(fixture);
      const el = fixture.nativeElement as HTMLElement;
      const line = el.querySelector('.media-threshold-line') as HTMLElement;
      expect(line).not.toBeNull();
      expect(line.className).toBe('media-threshold-line');
      expect(el.querySelector('.stripe-threshold')!.className).toBe('stripe-threshold');
      expect(line.textContent!.trim().toLowerCase()).toBe('threshold');
      expect(line.title).toContain(status === 'unchecked' ? 'Unchecked' : 'random picks');
    });

    it.each(FLOOR_STATES)('install the line and verdict a floor change returns when %s', (status) => {
      sortState.setSortResults(ranking, 0.5, lineFloor(status));
      vi.useFakeTimers();
      fixture.componentInstance.onMinPrecisionChange(0.9);
      vi.advanceTimersByTime(200);
      httpMock
        .expectOne((req) => req.url === '/api/min-precision' && req.method === 'POST')
        .flush({ ...wireFloor(status, { minPrecision: 0.9 }), threshold: 0.5, n_returned: 2 });
      expect(sortState.threshold).toBe(0.5);
      expect(sortState.floor?.status).toBe(status);
      expect(sortState.floor?.minPrecision).toBe(0.9);
    });

    it('move the line when a floor change keeps a larger set', () => {
      sortState.setSortResults(ranking, 0.5, lineFloor('unchecked'));
      vi.useFakeTimers();
      fixture.componentInstance.onMinPrecisionChange(0.25);
      vi.advanceTimersByTime(200);
      httpMock
        .expectOne((req) => req.url === '/api/min-precision' && req.method === 'POST')
        .flush({ ...wireFloor('confirmed', { minPrecision: 0.25, count: 64 }), threshold: 0.3, n_returned: 3 });
      expect(sortState.threshold).toBe(0.3);
      expect(sortState.floor?.status).toBe('confirmed');
      expect(view().unverifiedGoodIds()).toEqual([1, 2, 3]);
    });

    it.each(FLOOR_STATES)('show the floor, its state and the check affordance in the Find row when %s', async (status) => {
      sortState.setSortResults(ranking, 0.5, lineFloor(status));
      await settleZoneless(fixture);
      const row = (fixture.nativeElement as HTMLElement).querySelector('.find-floor-row')!;
      const text = row.querySelector('.floor-state')!.textContent!;
      expect(text).toContain(
        status === 'unchecked' ? 'unchecked' : status === 'confirmed' ? 'Confirmed · likely 55–100% right (checked 5)' : 'Aimed at Centered: likely 11–73% right',
      );
      expect(row.querySelector('.floor-check-btn')!.textContent!.trim()).toBe('Check 5 picks');
    });
  });

  describe('the spot check (#4273)', () => {
    beforeEach(async () => {
      await setUp(false);
      await flushInit(ranking.map(({ id }) => id));
      await settleZoneless(fixture);
      sortState.setSortResults(ranking, 0.5, lineFloor('unchecked'));
      await settleZoneless(fixture);
    });

    const running = (picks: number[]) => ({
      status: 'running',
      min_precision: 0.5,
      round: 1,
      rounds: 1,
      picks_per_round: picks.length,
      candidate: 32,
      start_candidate: 32,
      picks,
      labelled: 0,
      right: 0,
      range: null,
    });

    it('opens from the floor control, takes the vote keys, and installs the line the check ends on', async () => {
      const el = fixture.nativeElement as HTMLElement;
      (el.querySelector('.find-floor-row .floor-check-btn') as HTMLButtonElement).click();
      await settleZoneless(fixture);
      httpMock
        .expectOne((req) => req.url === '/api/precision-check/start')
        .flush({ floor: wireFloor('unchecked'), check: running([3]) });
      await settleZoneless(fixture);
      expect(el.querySelector('vt-floor-check-modal')).not.toBeNull();

      // → votes the pick in the step; the list behind it gets nothing.
      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
      await settleZoneless(fixture);
      expect(httpMock.match((req) => req.url.startsWith('/api/medias/') && req.url.endsWith('/vote'))).toEqual([]);
      const votes = httpMock.expectOne((req) => req.url === '/api/precision-check/votes');
      expect(votes.request.body).toEqual({ votes: [{ id: 3, label: 'good' }] });
      votes.flush({
        floor: wireFloor('confirmed'),
        check: { ...running([]), status: 'confirmed', labelled: 1, right: 1, range: { lo: 0.55, hi: 1, labelled: 1, right: 1 } },
      });
      await settleZoneless(fixture);

      // The finished check moved the line server-side; the view installs it.
      httpMock
        .expectOne((req) => req.url === '/api/min-precision' && req.method === 'GET')
        .flush({ ...wireFloor('confirmed'), threshold: 0.3, n_returned: 3 });
      await settleZoneless(fixture);
      expect(sortState.threshold).toBe(0.3);
      expect(sortState.floor?.status).toBe('confirmed');
      expect(el.querySelector('.find-floor-row .floor-state')!.textContent).toContain('Confirmed');
    });

    it('closing a running check cancels it and leaves the line as it was', async () => {
      const el = fixture.nativeElement as HTMLElement;
      (el.querySelector('.find-floor-row .floor-check-btn') as HTMLButtonElement).click();
      await settleZoneless(fixture);
      httpMock
        .expectOne((req) => req.url === '/api/precision-check/start')
        .flush({ floor: wireFloor('unchecked'), check: running([3, 1]) });
      await settleZoneless(fixture);
      (el.querySelector('vt-floor-check-modal .modal-footer .btn') as HTMLButtonElement).click();
      await settleZoneless(fixture);
      httpMock.expectOne((req) => req.url === '/api/precision-check/cancel').flush({ floor: wireFloor('unchecked'), check: null });
      expect(el.querySelector('vt-floor-check-modal')).toBeNull();
      // The view re-reads the line on close, and it is where it was.
      httpMock
        .expectOne((req) => req.url === '/api/min-precision' && req.method === 'GET')
        .flush({ ...wireFloor('unchecked'), threshold: 0.5, n_returned: 2 });
      await settleZoneless(fixture);
      expect(sortState.threshold).toBe(0.5);
      expect(sortState.floor?.status).toBe('unchecked');
    });
  });
});
