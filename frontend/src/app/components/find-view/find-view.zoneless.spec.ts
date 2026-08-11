import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';

import { FindViewComponent } from './find-view.component';
import { ActiveContextService } from '../../services/active-context.service';
import { SortStateService } from '../../services/sort-state.service';
import { BrowseSubsetService } from '../../services/browse-subset.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleResource, settleZoneless } from '../../testing/settle-resource';
import { provideHttpTesting } from '../../testing/test-providers';

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
    fixture.componentInstance.ngOnDestroy();
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush([]);
    });
    fixture.destroy();
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
      httpMock.match('/api/settings').forEach((req) => req.flush({ volume: 80 }));
      httpMock.match('/api/inclusion').forEach((req) => req.flush({ inclusion: 0 }));
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
 * The run is now scoped to the pair (`pairScope$`), so a switch tears it down.
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
    fixture.componentInstance.ngOnDestroy();
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush([]);
    });
    fixture.destroy();
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
      httpMock.match('/api/settings').forEach((req) => req.flush({ volume: 80 }));
      httpMock.match('/api/inclusion').forEach((req) => req.flush({ inclusion: 0 }));
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

  // The Inclusion POST is deferred until the slider settles (issue #2973), and
  // that settle window is inside the pair scope too: a slide the user abandons
  // by switching pair must never be written into the pair they switched *to*,
  // whose own inclusion the reload has just re-seeded.
  it('drops a pending inclusion POST when the pair switches first', async () => {
    await flushInit();
    // Land the first pair's ranking so the slider isn't disabled by sortBusy.
    httpMock
      .expectOne('/api/find-label')
      .flush({ results: [{ id: 1, score: 0.9 }], threshold: 0.5 });
    await flushInit();
    await settleZoneless(fixture);

    vi.useFakeTimers();
    try {
      fixture.componentInstance.onInclusionChange(5);
      // Still inside the settle window when the user switches pair.
      vi.advanceTimersByTime(50);
      activeContext.setActivePair('ds2', 'det2');
      vi.advanceTimersByTime(1000);

      httpMock.expectNone((req) => req.url === '/api/inclusion' && req.method === 'POST');
    } finally {
      vi.useRealTimers();
    }
  });
});

/**
 * Issue #2973: the Inclusion slider emits on every `input` event, so walking the
 * cutoff a few steps used to leave several `POST /api/inclusion` requests in
 * flight at once, each installing its own threshold on arrival. A slow response
 * for a value the user had already moved past could land *last* and overwrite
 * the newer threshold, snapping the green/red line (and the left/right split)
 * back to a cutoff that was no longer selected — with nothing to re-reconcile it
 * until the next slide. Slider changes now funnel through one debounced
 * `switchMap` pipeline, so only the settled value is sent and only its response
 * is applied.
 */
describe('FindViewComponent (inclusion supersession)', () => {
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
    fixture.componentInstance.ngOnDestroy();
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush([]);
    });
    fixture.destroy();
  });

  // Same drain as the canary above; no pair is active, so `runFindLabel` no-ops
  // and the only /api/inclusion traffic afterwards is the slider's.
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
      httpMock.match('/api/settings').forEach((req) => req.flush({ volume: 80 }));
      httpMock.match('/api/inclusion').forEach((req) => req.flush({ inclusion: 0 }));
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

  it('coalesces a rapid walk of the slider into one POST for the settled value', async () => {
    await flushInit();
    await settleZoneless(fixture);
    const sortState = seedRanking();
    const component = fixture.componentInstance;

    vi.useFakeTimers();
    // Three steps up in quick succession: the box tracks every one of them...
    component.onInclusionChange(1);
    vi.advanceTimersByTime(40);
    component.onInclusionChange(2);
    vi.advanceTimersByTime(40);
    component.onInclusionChange(3);
    expect(sortState.inclusion).toBe(3);
    // ...but nothing is sent until the slider settles.
    httpMock.expectNone('/api/inclusion');

    vi.advanceTimersByTime(200);
    const req = httpMock.expectOne('/api/inclusion');
    expect(req.request.body).toEqual({ inclusion: 3 });
    req.flush({ inclusion: 3, threshold: 0.7 });
    expect(sortState.threshold).toBe(0.7);
  });

  it('cancels a superseded POST so its stale threshold can never land', async () => {
    await flushInit();
    await settleZoneless(fixture);
    const sortState = seedRanking();
    const component = fixture.componentInstance;

    vi.useFakeTimers();
    component.onInclusionChange(3);
    vi.advanceTimersByTime(200);
    // The first POST is still in flight (a slow re-threshold server-side) when
    // the user moves the cutoff again.
    const stale = httpMock.expectOne('/api/inclusion');
    expect(stale.cancelled).toBe(false);

    component.onInclusionChange(7);
    vi.advanceTimersByTime(200);

    // switchMap aborted the superseded request, so its threshold (0.2) has no
    // subscriber left to install it however late it resolves.
    expect(stale.cancelled).toBe(true);
    const fresh = httpMock.expectOne('/api/inclusion');
    expect(fresh.request.body).toEqual({ inclusion: 7 });
    fresh.flush({ inclusion: 7, threshold: 0.9 });
    expect(sortState.threshold).toBe(0.9);
  });

  it('keeps posting after a failed slide', async () => {
    await flushInit();
    await settleZoneless(fixture);
    const sortState = seedRanking();
    const component = fixture.componentInstance;

    vi.useFakeTimers();
    component.onInclusionChange(2);
    vi.advanceTimersByTime(200);
    httpMock
      .expectOne('/api/inclusion')
      .flush({ message: 'boom' }, { status: 500, statusText: 'Server Error' });

    // The error is swallowed per-request, so the shared pipeline survives it.
    component.onInclusionChange(4);
    vi.advanceTimersByTime(200);
    const retry = httpMock.expectOne('/api/inclusion');
    expect(retry.request.body).toEqual({ inclusion: 4 });
    retry.flush({ inclusion: 4, threshold: 0.6 });
    expect(sortState.threshold).toBe(0.6);
  });
});
