import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { SortRunnerService } from './sort-runner.service';
import { PairScopeService } from './pair-scope.service';
import { SortStateService } from './sort-state.service';
import { MediaStateService } from './media-state.service';
import { VoteStateService } from './vote-state.service';
import { AutopilotStateService } from './autopilot-state.service';
import { configureZoneless } from '../testing/zoneless-testbed';
import { provideHttpTesting } from '../testing/test-providers';

/**
 * `SortRunnerService` in isolation.
 *
 * The point of the extraction (#3428) is that these paths no longer need a
 * `LabelViewComponent` to exercise: no `viewChild.required` layout element, no
 * left/right-panel HTTP to drain, no `ngOnInit` load storm. Every sort is one
 * call and one `flush`. label-view's own spec keeps the through-the-component
 * coverage (the template bindings and the pair-switch supersession); what lives
 * here is the machinery those tests could not reach cheaply — the "Load more"
 * re-entrancy guard, the window-metadata mapping, and `quiesce`.
 */
describe('SortRunnerService', () => {
  let runner: SortRunnerService;
  let sortState: SortStateService;
  let mediaState: MediaStateService;
  let voteState: VoteStateService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    configureZoneless({
      providers: [...provideHttpTesting(), PairScopeService, SortRunnerService],
    });
    runner = TestBed.inject(SortRunnerService);
    sortState = TestBed.inject(SortStateService);
    mediaState = TestBed.inject(MediaStateService);
    voteState = TestBed.inject(VoteStateService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    TestBed.inject(VoteStateService).stopPolling();
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush([]);
    });
  });

  // --- window metadata ------------------------------------------------------

  it('carries the window metadata a paged ranking needs', () => {
    runner.onTextSort('birds');
    httpMock.expectOne('/api/sort').flush({
      results: [
        { id: 1, similarity: 0.9 },
        { id: 2, similarity: 0.4 },
      ],
      threshold: 0.5,
      total: 900,
      above_threshold: 120,
      has_more_below: true,
      sort_token: 'tok-1',
    });

    expect(sortState.sortOrder?.map((i) => i.id)).toEqual([1, 2]);
    expect(sortState.sortTotal).toBe(900);
    expect(sortState.aboveThreshold).toBe(120);
    expect(sortState.sortHasMore).toBe(true);
    expect(sortState.sortToken).toBe('tok-1');
    expect(sortState.sortBusy).toBe(false);
  });

  it('derives the above-threshold count when the server omits it, and treats an unwindowed response as complete', () => {
    runner.onTextSort('birds');
    httpMock.expectOne('/api/sort').flush({
      results: [
        { id: 1, similarity: 0.9 },
        { id: 2, similarity: 0.4 },
      ],
      threshold: 0.5,
    });

    expect(sortState.aboveThreshold).toBe(1);
    expect(sortState.sortTotal).toBe(2);
    expect(sortState.sortHasMore).toBe(false);
    expect(sortState.sortToken).toBeNull();
  });

  it('reads `score` rows as well as `similarity` rows', () => {
    runner.onExampleSortStarted({
      results: [{ id: 7, score: 0.8, best_region: [0, 0, 1, 1] }],
      threshold: 0.1,
    });

    expect(sortState.sortOrder).toEqual([{ id: 7, score: 0.8, bestRegion: [0, 0, 1, 1] }]);
    expect(sortState.loadSortLabel).toBe('Example media');
    expect(sortState.sortMode).toBe('load');
  });

  it('reports a failed sort without leaving the panel busy', () => {
    runner.onTextSort('birds');
    httpMock.expectOne('/api/sort').flush(null, { status: 500, statusText: 'Server Error' });

    expect(sortState.sortBusy).toBe(false);
    expect(sortState.sortStatus).toBe('Sort failed');
  });

  // --- "Load more" ----------------------------------------------------------

  function seedWindow(): void {
    runner.onTextSort('birds');
    httpMock.expectOne('/api/sort').flush({
      results: [{ id: 1, similarity: 0.9 }],
      threshold: 0.5,
      total: 900,
      above_threshold: 1,
      has_more_below: true,
      sort_token: 'tok-1',
    });
  }

  it('appends a page and updates hasMore from the page response', () => {
    seedWindow();

    runner.onLoadMore();
    const page = httpMock.expectOne((req) => req.url.startsWith('/api/sort/page'));
    expect(runner.loadingMoreSort()).toBe(true);
    page.flush({ results: [{ id: 2, score: 0.3 }], has_more: false });

    expect(sortState.sortOrder?.map((i) => i.id)).toEqual([1, 2]);
    expect(sortState.sortHasMore).toBe(false);
    expect(runner.loadingMoreSort()).toBe(false);
  });

  it('does not issue a second page fetch while one is in flight', () => {
    seedWindow();

    runner.onLoadMore();
    runner.onLoadMore();

    // Re-entrancy guard: the second call must be a no-op, not a duplicate page
    // appended at the same offset.
    const pages = httpMock.match((req) => req.url.startsWith('/api/sort/page'));
    expect(pages.length).toBe(1);
    pages[0].flush({ results: [{ id: 2, score: 0.3 }], has_more: true });
    expect(sortState.sortOrder?.map((i) => i.id)).toEqual([1, 2]);
  });

  it('stops paging when the token expires, leaving the loaded window intact', () => {
    seedWindow();

    runner.onLoadMore();
    httpMock
      .expectOne((req) => req.url.startsWith('/api/sort/page'))
      .flush(null, { status: 404, statusText: 'Not Found' });

    expect(runner.loadingMoreSort()).toBe(false);
    expect(sortState.sortOrder?.map((i) => i.id)).toEqual([1]);
  });

  it('is a no-op with nothing left to page', () => {
    runner.onTextSort('birds');
    httpMock.expectOne('/api/sort').flush({ results: [{ id: 1, similarity: 0.9 }], threshold: 0.5 });

    runner.onLoadMore();

    httpMock.expectNone((req) => req.url.startsWith('/api/sort/page'));
  });

  // --- cancellation ---------------------------------------------------------

  /** Give the active detector one good and one bad label, which is what
   *  `learnedSortAvailable` gates on. */
  function enableLearnedSort(): void {
    voteState.loadVotes();
    httpMock
      .expectOne('/api/votes')
      .flush({ good: [1], bad: [2], click_times: {}, learned_scores: {} });
  }

  it('cancels the learned-sort job by id, and only once', () => {
    enableLearnedSort();
    runner.onLearnedSort();
    httpMock.expectOne('/api/learned-sort').flush({ status: 'running', job_id: 'job-7' });

    runner.onSortCancel();
    const cancel = httpMock.expectOne('/api/learned-sort/cancel/job-7');
    cancel.flush({ cancelled: true });

    // The id is consumed by the cancel, so a second press cannot re-target it.
    runner.onSortCancel();
    httpMock.expectNone((req) => req.url.startsWith('/api/learned-sort/cancel'));
  });

  it('cancels a detector sort through the find cancel flag', () => {
    sortState.setSortMode('load');

    runner.onSortCancel();

    httpMock.expectOne('/api/find/cancel').flush({ ok: true });
  });

  // --- quiesce --------------------------------------------------------------

  it('quiesce drops the busy flag a superseded sort left behind', () => {
    runner.onTextSort('birds');
    expect(sortState.sortBusy).toBe(true);

    runner.quiesce();

    expect(sortState.sortBusy).toBe(false);
  });

  it('quiesce forgets the in-flight job, so a later cancel cannot target the old pair’s run', () => {
    sortState.setSortMode('load');
    runner.quiesce();

    runner.onSortCancel();

    // Falls through to the load-sort branch (find cancel), never to a stale
    // `/api/learned-sort/cancel` for a job the previous pair started.
    httpMock.expectNone((req) => req.url.startsWith('/api/learned-sort/cancel'));
    httpMock.expectOne('/api/find/cancel').flush({ ok: true });
  });

  // --- selection advance ----------------------------------------------------

  it('probes the coverage atlas in `new` mode and records the level it reports', () => {
    sortState.setSelectMode('new');
    sortState.setSortResults([{ id: 1, score: 0.9 }], 0.5);

    runner.autoSelectNext();

    const probe = httpMock.expectOne((req) => req.url.startsWith('/api/coverage-atlas/next'));
    probe.flush({ id: 42, coverage_level: 3 });

    expect(mediaState.selectedId()).toBe(42);
    expect(TestBed.inject(AutopilotStateService).state.fracDiversity).toBe(3);
  });

  it('selects the top unlabeled item without any request in `top` mode', () => {
    sortState.setSelectMode('top');
    sortState.setSortResults(
      [
        { id: 1, score: 0.9 },
        { id: 2, score: 0.8 },
      ],
      0.5,
    );

    runner.autoSelectNext();

    expect(mediaState.selectedId()).toBe(1);
    httpMock.expectNone((req) => req.url.startsWith('/api/coverage-atlas/next'));
  });

  // --- inclusion ------------------------------------------------------------

  it('pushes the inclusion value and re-advances the selection', () => {
    sortState.setSelectMode('top');
    sortState.setSortResults([{ id: 5, score: 0.9 }], 0.5);

    runner.onInclusionChange(0.25);

    expect(sortState.inclusion).toBe(0.25);
    httpMock.expectOne('/api/inclusion').flush({ inclusion: 0.25 });
    expect(mediaState.selectedId()).toBe(5);
  });
});
