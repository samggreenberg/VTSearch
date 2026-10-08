import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { SortRunnerService } from './sort-runner.service';
import { PairScopeService } from './pair-scope.service';
import { SortStateService } from './sort-state.service';
import { MediaStateService } from './media-state.service';
import { VoteStateService } from './vote-state.service';
import { AutopilotStateService } from './autopilot-state.service';
import { ActiveContextService } from './active-context.service';
import { configureZoneless } from '../testing/zoneless-testbed';
import { provideHttpTesting } from '../testing/test-providers';
import { settleResource } from '../testing/settle-resource';
import { BALANCE_STATES, lineBalance, wireBalance } from '../testing/line-balance';

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

  // --- the balance on the line (#4247, #4272, #4413) ---------------------------

  it.each(BALANCE_STATES)(
    'installs a learned sort\'s line with its state when %s',
    (status) => {
      enableLearnedSort();
      runner.onLearnedSort(false);
      httpMock.expectOne('/api/learned-sort').flush({
        status: 'done',
        results: [
          { id: 1, score: 0.9 },
          { id: 2, score: 0.3 },
        ],
        threshold: 0.5,
        acq_threshold: 0.7,
        balance: wireBalance(status),
        total: 2,
        above_threshold: 1,
        has_more_below: false,
      });

      // The cut is a cut: the line, its count and Autopilot's acquisition cut all land.
      expect(sortState.threshold).toBe(0.5);
      expect(sortState.aboveThreshold).toBe(1);
      expect(sortState.acqThreshold).toBe(0.7);
      expect(sortState.balance?.status).toBe(status);
    },
  );

  it('a text sort carries no balance', () => {
    runner.onTextSort('birds');
    httpMock.expectOne('/api/sort').flush({ results: [{ id: 1, similarity: 0.9 }], threshold: 0.5 });
    expect(sortState.balance).toBeNull();
  });

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

  it('quiesce forgets the kind of the sort it superseded (#4326)', () => {
    runner.onTextSort('birds');
    runner.quiesce();

    expect(runner.newestSortKind()).toBeNull();
  });

  // --- a newer sort ends an older one (#4318) -----------------------------------

  describe('a newer sort ends an older one (#4318)', () => {
    it('drops the answer of a sort asked for before the one on its way', () => {
      runner.onTextSort('first');
      runner.onTextSort('second');
      const [first, second] = httpMock.match('/api/sort');

      expect(first.cancelled).toBe(true);
      second.flush({ results: [{ id: 2, similarity: 0.9 }], threshold: 0.5 });
      expect(sortState.sortOrder?.map((i) => i.id)).toEqual([2]);
      expect(sortState.sortBusy).toBe(false);
    });

    it('keeps the older sort from clearing the busy flag the newer one raised', () => {
      enableLearnedSort();
      runner.onTextSort('seed');
      runner.onLearnedSort(false);

      // The text request is gone, so nothing can land and report "not busy"
      // while the model is still training.
      expect(httpMock.expectOne('/api/sort').cancelled).toBe(true);
      expect(sortState.sortBusy).toBe(true);
      httpMock.expectOne('/api/learned-sort').flush({
        status: 'done',
        results: [{ id: 1, score: 0.9 }],
        threshold: 0.5,
      });
      expect(sortState.sortBusy).toBe(false);
    });

    it('stops polling a learned-sort job a newer sort replaced, and leaves the job alone', () => {
      enableLearnedSort();
      runner.onLearnedSort(false);
      httpMock.expectOne('/api/learned-sort').flush({ status: 'running', job_id: 'job-3' });

      runner.onTextSort('birds');
      httpMock.expectOne('/api/sort').flush({ results: [{ id: 2, similarity: 0.9 }], threshold: 0.5 });

      // The server coalesces a pair's learned sorts into one job, so cancelling
      // it there could cancel a newer request's; the Cancel button cannot reach
      // it either.
      runner.onSortCancel();
      httpMock.expectNone((req) => req.url.startsWith('/api/learned-sort/cancel'));
      expect(sortState.sortOrder?.map((i) => i.id)).toEqual([2]);
    });

    it('treats a ranking installed on the spot as the newest sort', () => {
      runner.onTextSort('birds');
      runner.onExampleSortStarted({ results: [{ id: 3, similarity: 0.8 }], threshold: 0.5 }, false);

      expect(httpMock.expectOne('/api/sort').cancelled).toBe(true);
      expect(sortState.sortOrder?.map((i) => i.id)).toEqual([3]);
    });

    it('never appends a page of the ranking a newer sort replaced', () => {
      seedWindow();
      runner.onLoadMore();
      const page = httpMock.expectOne((req) => req.url.startsWith('/api/sort/page'));

      runner.onTextSort('fish');
      expect(page.cancelled).toBe(true);
      // Paging is free again, rather than stuck behind the fetch that never answered.
      expect(runner.loadingMoreSort()).toBe(false);
    });

    it('stops a detector sort\'s progress feed and count when a newer sort starts', () => {
      runner.onModelSelected('det-1');
      sortState.setSortProgress(40, 100);

      runner.onTextSort('birds');

      expect(httpMock.expectOne('/api/find-label').cancelled).toBe(true);
      expect(sortState.sortProgress).toBe(0);
      expect(sortState.sortProgressTotal).toBe(0);
    });

    it('names the kind of the newest sort, in flight and once landed (#4326)', () => {
      enableLearnedSort();
      expect(runner.newestSortKind()).toBeNull();
      // A `learned` mode carried over from the last session says nothing about
      // the sort actually on its way.
      sortState.setSortMode('learned');
      runner.onTextSort('seed');
      expect(runner.newestSortKind()).toBe('text');

      runner.onLearnedSort(false);
      expect(runner.newestSortKind()).toBe('learned');
      httpMock.expectOne('/api/learned-sort').flush({
        status: 'done',
        results: [{ id: 1, score: 0.9 }],
        threshold: 0.5,
      });
      expect(runner.newestSortKind()).toBe('learned');

      runner.onExampleSortStarted({ results: [{ id: 3, similarity: 0.8 }], threshold: 0.5 }, false);
      expect(runner.newestSortKind()).toBe('example');
    });
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

  /**
   * The centre panel keeps the voted item swiped off-screen while this is true,
   * so it has to be true for exactly the length of the round-trip (#4307).
   */
  describe('advancePending', () => {
    const probes = () =>
      httpMock.match((req) => req.url.startsWith('/api/coverage-atlas/next'));

    beforeEach(() => {
      sortState.setSelectMode('new');
      sortState.setSortResults([{ id: 1, score: 0.9 }], 0.5);
    });

    it('is true while the New-mode probe is in the air, and false once it answers', () => {
      expect(runner.advancePending()).toBe(false);

      runner.autoSelectNext();
      expect(runner.advancePending()).toBe(true);

      probes()[0].flush({ id: 42, coverage_level: 3 });
      expect(mediaState.selectedId()).toBe(42);
      expect(runner.advancePending()).toBe(false);
    });

    it('ends when the probe comes back empty', () => {
      runner.autoSelectNext();
      probes()[0].flush({ id: null, coverage_level: 3 });

      expect(runner.queueExhausted()).toBe(true);
      expect(runner.advancePending()).toBe(false);
    });

    it('holds until the last of two overlapping probes answers', () => {
      runner.autoSelectNext();
      runner.autoSelectNext();
      const [first, second] = probes();

      first.flush({ id: 5, coverage_level: 1 });
      expect(runner.advancePending()).toBe(true);

      second.flush({ id: 6, coverage_level: 1 });
      expect(runner.advancePending()).toBe(false);
    });

    it('ends when a pair switch supersedes the probe', () => {
      runner.autoSelectNext();
      expect(runner.advancePending()).toBe(true);

      TestBed.inject(PairScopeService).resetForNewPair();

      expect(runner.advancePending()).toBe(false);
    });

    it('is never set by an advance that needs no request', () => {
      sortState.setSelectMode('top');

      runner.autoSelectNext();

      expect(mediaState.selectedId()).toBe(1);
      expect(runner.advancePending()).toBe(false);
    });
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

  it('peeks the same pick it would select, and selects nothing (#3896)', () => {
    sortState.setSelectMode('top');
    sortState.setSortResults(
      [
        { id: 1, score: 0.9 },
        { id: 2, score: 0.8 },
      ],
      0.5,
    );

    // The image prefetch runs off this while the reviewer is still looking at
    // item 1, so it must answer for the item *after* the one on screen without
    // moving anybody off it.
    expect(runner.peekNextMedia(1)).toEqual({ kind: 'media', id: 2 });
    expect(mediaState.selectedId()).toBeNull();

    runner.autoSelectNext(1);
    expect(mediaState.selectedId()).toBe(2);
  });

  it('peeks `diversity` without firing the atlas probe (#3896)', () => {
    sortState.setSelectMode('new');
    sortState.setSortResults([{ id: 1, score: 0.9 }], 0.5);

    // A `new`-mode pick is a server round-trip, which a prefetch must not fire:
    // the probe is what *chooses* the next item, so calling it speculatively
    // would consume a choice the reviewer has not arrived at yet.
    expect(runner.peekNextMedia(1)).toEqual({ kind: 'diversity' });
    httpMock.expectNone((req) => req.url.startsWith('/api/coverage-atlas/next'));
  });

  describe('peekUpcomingMedia (#3896)', () => {
    const ranking = [
      { id: 1, score: 0.9 },
      { id: 2, score: 0.8 },
      { id: 3, score: 0.6 },
      { id: 4, score: 0.4 },
      { id: 5, score: 0.2 },
    ];

    it('lists the next items the advance would show, skipping labeled ones', () => {
      sortState.setSelectMode('top');
      sortState.setSortResults(ranking, 0.5);
      voteState.applyOptimisticState(2, 'bad');

      expect(runner.peekUpcomingMedia(1, 2)).toEqual([3, 4]);
      expect(mediaState.selectedId()).toBeNull();
    });

    /**
     * The queue is only worth anything if each entry is what the advance will
     * actually pick once the reviewer gets there. Walk it for real: vote, let
     * `autoSelectNext` choose, and compare with the peek taken beforehand.
     */
    it.each(['top', 'hard'] as const)('matches what successive votes select in `%s` mode', (mode) => {
      sortState.setSelectMode(mode);
      sortState.setSortResults(ranking, 0.5);
      runner.autoSelectNext();
      const current = mediaState.selectedId()!;
      const predicted = runner.peekUpcomingMedia(current, 2);

      const walked: number[] = [];
      let on = current;
      for (let i = 0; i < 2; i++) {
        voteState.applyOptimisticState(on, 'good');
        runner.autoSelectNext(on);
        on = mediaState.selectedId()!;
        walked.push(on);
      }
      expect(predicted).toEqual(walked);
    });

    it('follows the ranking when a re-sort lands with the same item on screen', () => {
      sortState.setSelectMode('top');
      sortState.setSortResults(ranking, 0.5);
      expect(runner.peekUpcomingMedia(1, 2)).toEqual([2, 3]);

      sortState.setSortResults(
        [ranking[0], ranking[4], ranking[3], ranking[2], ranking[1]],
        0.5,
      );
      expect(runner.peekUpcomingMedia(1, 2)).toEqual([5, 4]);
    });

    it('stops short when the ranking runs out', () => {
      sortState.setSelectMode('top');
      sortState.setSortResults(ranking.slice(0, 2), 0.5);
      expect(runner.peekUpcomingMedia(1, 2)).toEqual([2]);
    });

    it('stops at a `new`-mode pick and fires no probe', () => {
      sortState.setSelectMode('new');
      sortState.setSortResults(ranking, 0.5);
      expect(runner.peekUpcomingMedia(1, 2)).toEqual([]);
      httpMock.expectNone((req) => req.url.startsWith('/api/coverage-atlas/next'));
    });

    it('is empty with nothing on screen', () => {
      sortState.setSelectMode('top');
      sortState.setSortResults(ranking, 0.5);
      expect(runner.peekUpcomingMedia(null, 2)).toEqual([]);
    });
  });

  // --- balance (#4413) --------------------------------------------------------

  describe('the line after a spot check (#4273)', () => {
    const balanceGet = (req: { url: string; method: string }) =>
      req.url === '/api/balance' && req.method === 'GET';

    it('moves a learned line to the set where the check peaked, without a re-sort', () => {
      vi.useFakeTimers();
      enableLearnedSort();
      sortState.setSortMode('learned');
      sortState.setSortResults(
        [
          { id: 5, score: 0.9 },
          { id: 6, score: 0.2 },
        ],
        0.5,
        lineBalance('unchecked', { count: 64 }),
      );

      runner.refreshLine();
      httpMock.expectOne(balanceGet).flush({ ...wireBalance('checked'), threshold: 0.8, n_returned: 7 });
      vi.advanceTimersByTime(1000);

      expect(sortState.threshold).toBe(0.8);
      expect(sortState.balance?.status).toBe('checked');
      expect(sortState.balance?.precision?.labelled).toBe(5);
      expect(sortState.balance?.recall?.labelled).toBe(5);
      // The server's count over the whole ranking, not the loaded head's.
      expect(sortState.aboveThreshold).toBe(7);
      expect(sortState.sortOrder?.map((i) => i.id)).toEqual([5, 6]);
      httpMock.expectNone('/api/learned-sort');
      vi.useRealTimers();
    });

    it('leaves a ranking the detector did not draw alone', () => {
      sortState.setSortMode('text');
      sortState.setSortResults([{ id: 5, score: 0.9 }], 0.3);
      runner.refreshLine();
      httpMock.expectOne(balanceGet).flush({ ...wireBalance('checked'), threshold: 0.8, n_returned: 1 });
      expect(sortState.threshold).toBe(0.3);
      expect(sortState.balance).toBeNull();
    });
  });

  describe('a balance change (#4413)', () => {
    const balancePost = (req: { url: string; method: string }) =>
      req.url === '/api/balance' && req.method === 'POST';

    /** A learned ranking on screen, with the balance's state on its line. */
    function learnedRanking(status: Parameters<typeof lineBalance>[0]): void {
      enableLearnedSort();
      sortState.setSortMode('learned');
      sortState.setSelectMode('top');
      sortState.setSortResults(
        [
          { id: 5, score: 0.9 },
          { id: 6, score: 0.2 },
        ],
        0.5,
        lineBalance(status),
      );
    }

    afterEach(() => vi.useRealTimers());

    it('posts the beta and moves the picker at once', () => {
      runner.onBetaChange(0.5);

      expect(sortState.beta).toBe(0.5);
      expect(httpMock.expectOne(balancePost).request.body).toEqual({ beta: 0.5 });
    });

    it('leaves a ranking the detector did not draw alone', () => {
      vi.useFakeTimers();
      sortState.setSortMode('text');
      sortState.setSortResults([{ id: 5, score: 0.9 }], 0.3);

      runner.onBetaChange(2);
      httpMock.expectOne(balancePost).flush({ ...wireBalance('checked', { beta: 2, count: 64 }), threshold: 0.1, n_returned: 9 });
      vi.advanceTimersByTime(1000);

      httpMock.expectNone('/api/learned-sort');
      expect(sortState.threshold).toBe(0.3);
      expect(sortState.balance).toBeNull();
    });

    it.each(BALANCE_STATES)(
      'keeps the line and swaps only the state when the new balance keeps the same count (%s)',
      (status) => {
        vi.useFakeTimers();
        learnedRanking('unchecked');

        runner.onBetaChange(0.5);
        httpMock.expectOne(balancePost).flush({ ...wireBalance(status, { beta: 0.5 }), threshold: 0.5, n_returned: 1 });
        vi.advanceTimersByTime(1000);

        // Both balances keep the top 32: the count decides the line, so it cannot move.
        httpMock.expectNone('/api/learned-sort');
        expect(sortState.threshold).toBe(0.5);
        expect(sortState.balance?.status).toBe(status);
        expect(sortState.balance?.beta).toBe(0.5);
      },
    );

    it('re-runs the learned sort only once the server has the new balance', () => {
      vi.useFakeTimers();
      learnedRanking('checked');

      runner.onBetaChange(2);
      const post = httpMock.expectOne(balancePost);
      // A re-sort that beat the POST would read the old balance server-side.
      vi.advanceTimersByTime(1000);
      httpMock.expectNone('/api/learned-sort');

      post.flush({ ...wireBalance('checked', { beta: 2, count: 64 }), threshold: 0.15, n_returned: 2 });
      vi.advanceTimersByTime(300);
      httpMock.expectOne('/api/learned-sort').flush({
        status: 'done',
        results: [
          { id: 6, score: 0.2 },
          { id: 5, score: 0.1 },
        ],
        threshold: 0.15,
        acq_threshold: 0.18,
        balance: wireBalance('checked', { beta: 2, count: 64 }),
        total: 2,
        above_threshold: 1,
        has_more_below: false,
      });

      // The re-sort brings the line, its state, the count and the acquisition cut back together...
      expect(sortState.threshold).toBe(0.15);
      expect(sortState.acqThreshold).toBe(0.18);
      expect(sortState.aboveThreshold).toBe(1);
      expect(sortState.balance?.beta).toBe(2);
      // ...and lands on the next pick from them.
      expect(mediaState.selectedId()).toBe(6);
    });

    it('re-sorts when the new balance keeps a different count', () => {
      vi.useFakeTimers();
      learnedRanking('unchecked');

      runner.onBetaChange(2);
      httpMock.expectOne(balancePost).flush({ ...wireBalance('checked', { beta: 2, count: 64 }), threshold: 0.15, n_returned: 2 });
      vi.advanceTimersByTime(300);

      httpMock.expectOne('/api/learned-sort');
    });

    it('drops a balance the user moved past', () => {
      learnedRanking('unchecked');

      runner.onBetaChange(0.5);
      const stale = httpMock.expectOne(balancePost);
      runner.onBetaChange(2);

      expect(stale.cancelled).toBe(true);
      const fresh = httpMock.expectOne(balancePost);
      expect(fresh.request.body).toEqual({ beta: 2 });
      fresh.flush({ ...wireBalance('unchecked', { beta: 2 }), threshold: 0.5, n_returned: 1 });
      expect(sortState.balance?.beta).toBe(2);
    });

    it('keeps posting after a failed change', () => {
      learnedRanking('unchecked');

      runner.onBetaChange(0.5);
      httpMock.expectOne(balancePost).flush({ message: 'boom' }, { status: 500, statusText: 'Server Error' });
      expect(sortState.balance?.beta).toBe(1);

      runner.onBetaChange(2);
      httpMock.expectOne(balancePost).flush({ ...wireBalance('checked', { beta: 2 }), threshold: 0.5, n_returned: 1 });
      expect(sortState.balance?.status).toBe('checked');
    });
  });

  // --- exhausted queue (#3887) ---------------------------------------------

  /**
   * The state `autoSelectNext` silently no-ops in. It needs a name because the
   * centre pane's vote-swipe animation pins the outgoing media off-screen until
   * a new item replaces the node — so an advance that finds nothing leaves a
   * blank pane with no message and the just-voted item still selected.
   */
  describe('queueExhausted', () => {
    it('is false before any sort has landed — that is the placeholder state', () => {
      expect(runner.queueExhausted()).toBe(false);
    });

    it('is false while the ranking still holds an unlabeled row', () => {
      sortState.setSelectMode('top');
      sortState.setSortResults([{ id: 1, score: 0.9 }, { id: 2, score: 0.8 }], 0.5);
      voteState.applyOptimisticState(1, 'good');

      expect(runner.queueExhausted()).toBe(false);
    });

    it('is true once every row in the ranking is labeled', () => {
      sortState.setSelectMode('top');
      sortState.setSortResults([{ id: 1, score: 0.9 }, { id: 2, score: 0.8 }], 0.5);
      voteState.applyOptimisticState(1, 'good');
      voteState.applyOptimisticState(2, 'bad');

      expect(runner.queueExhausted()).toBe(true);
      // ...and the advance it describes really does have nowhere to go.
      runner.autoSelectNext();
      expect(mediaState.selectedId()).toBeNull();
    });

    it('goes back to false when an undo un-votes a row', () => {
      sortState.setSelectMode('top');
      sortState.setSortResults([{ id: 1, score: 0.9 }], 0.5);
      voteState.applyOptimisticState(1, 'good');
      expect(runner.queueExhausted()).toBe(true);

      // Derived, not latched: nothing has to remember to clear it.
      voteState.applyOptimisticState(1, 'none');
      expect(runner.queueExhausted()).toBe(false);
    });

    it('defers to the coverage-atlas probe under the New select mode', () => {
      sortState.setSelectMode('new');
      sortState.setSortResults([{ id: 1, score: 0.9 }], 0.5);
      voteState.applyOptimisticState(1, 'good');
      // Every row is labeled, but `new` samples the whole dataset rather than
      // the loaded window, so the window says nothing about exhaustion.
      expect(runner.queueExhausted()).toBe(false);

      runner.autoSelectNext();
      httpMock
        .expectOne((req) => req.url.startsWith('/api/coverage-atlas/next'))
        .flush({ id: null, coverage_level: 3 });
      expect(runner.queueExhausted()).toBe(true);

      runner.autoSelectNext();
      httpMock
        .expectOne((req) => req.url.startsWith('/api/coverage-atlas/next'))
        .flush({ id: 7, coverage_level: 3 });
      expect(runner.queueExhausted()).toBe(false);
      expect(mediaState.selectedId()).toBe(7);
    });

    /**
     * #4312: under New the answer is the server's, so it cannot be derived from
     * the vote sets the way Top / Hard's is. It is kept with the labels it was
     * given, and lapses once one of those is un-voted.
     */
    describe('under the New select mode', () => {
      const isAtlasProbe = (req: { url: string }) => req.url.startsWith('/api/coverage-atlas/next');

      /** Vote item 1 good and let the advance come back empty. */
      function exhaustOnItemOne(): void {
        sortState.setSelectMode('new');
        sortState.setSortResults([{ id: 1, score: 0.9 }], 0.5);
        mediaState.selectMedia(1);
        voteState.recordVote(1, 'good', 'one.png');
        voteState.applyOptimisticState(1, 'good');
        runner.autoSelectNext(1);
        httpMock.expectOne(isAtlasProbe).flush({ id: null, coverage_level: 3 });
        expect(runner.queueExhausted()).toBe(true);
      }

      it('goes back to false when an undo un-votes a label the empty answer was given', () => {
        exhaustOnItemOne();

        voteState.undo();

        // Straight back to work, as under Top / Hard: the undone item is still
        // the selection, and nothing waits on the server to say so.
        expect(runner.queueExhausted()).toBe(false);
        expect(mediaState.selectedId()).toBe(1);
        httpMock.expectNone(isAtlasProbe);
        httpMock.expectOne('/api/medias/1/vote').flush({ ok: true, state: 'none', click_time: null });
        expect(runner.queueExhausted()).toBe(false);
      });

      it('holds through votes that only add labels or flip one', () => {
        exhaustOnItemOne();

        // The atlas runs dry when every node carries a label, so neither of
        // these can bring an unseen node back.
        voteState.applyOptimisticState(9, 'bad');
        voteState.applyOptimisticState(1, 'bad');

        expect(runner.queueExhausted()).toBe(true);
      });

      it('stays up across the probe a re-vote fires, rather than blinking off', () => {
        exhaustOnItemOne();
        voteState.applyOptimisticState(1, 'none');
        expect(runner.queueExhausted()).toBe(false);

        // Re-voting restores the labels the empty answer was given, so the
        // answer holds again, and the probe the vote fires can only confirm it.
        voteState.applyOptimisticState(1, 'good');
        expect(runner.queueExhausted()).toBe(true);
        runner.autoSelectNext(1);
        expect(runner.advancePending()).toBe(true);
        expect(runner.queueExhausted()).toBe(true);

        httpMock.expectOne(isAtlasProbe).flush({ id: null, coverage_level: 3 });
        expect(runner.queueExhausted()).toBe(true);
      });

      it('goes back to false for an undo that lands while the probe is in the air', () => {
        sortState.setSelectMode('new');
        sortState.setSortResults([{ id: 1, score: 0.9 }], 0.5);
        voteState.applyOptimisticState(1, 'good');
        runner.autoSelectNext(1);
        const probe = httpMock.expectOne(isAtlasProbe);

        // The server may well have answered before it saw the undo, so the
        // answer is about the labels the probe went out with.
        voteState.applyOptimisticState(1, 'none');
        probe.flush({ id: null, coverage_level: 3 });

        expect(runner.queueExhausted()).toBe(false);
      });
    });
  });

  // --- the dataset, and the advance, running out (#4028) -------------------

  /** Answer the dataset stub load with `ids`, and let the resource settle. */
  async function seedMedias(...ids: number[]): Promise<void> {
    mediaState.loadMedias();
    // The `rxResource` loader runs in an effect, so the GET is not issued until
    // the TestBed ticks (see `settle-resource.ts`).
    TestBed.tick();
    httpMock
      .expectOne('/api/medias/ids')
      .flush(ids.map((id) => ({ id, media_type: 'image' })));
    await settleResource();
  }

  /**
   * `queueExhausted` is about the loaded *ranking*, so it is false whenever no
   * sort has run — which is exactly the state manual labelling and a fresh
   * entry to a finished detector are both in.
   */
  describe('datasetExhausted', () => {
    it('is false before the dataset stubs have loaded', () => {
      expect(runner.datasetExhausted()).toBe(false);
    });

    it('is false while one item in the dataset is still unlabeled', async () => {
      await seedMedias(1, 2);
      voteState.applyOptimisticState(1, 'good');

      expect(runner.datasetExhausted()).toBe(false);
    });

    it('is true once every item is labeled, with no sort ever having run', async () => {
      await seedMedias(1, 2);
      voteState.applyOptimisticState(1, 'good');
      voteState.applyOptimisticState(2, 'bad');

      // The ranking is empty, so the #3887 flag cannot speak for this state.
      expect(runner.queueExhausted()).toBe(false);
      expect(runner.datasetExhausted()).toBe(true);
    });

    it('goes back to false when an undo un-votes a row', async () => {
      await seedMedias(1);
      voteState.applyOptimisticState(1, 'good');
      expect(runner.datasetExhausted()).toBe(true);

      voteState.applyOptimisticState(1, 'none');
      expect(runner.datasetExhausted()).toBe(false);
    });
  });
  // --- carrying a sort to a new pair (#4092) ---------------------------------

  /**
   * The rule for a Train entry or pair switch: the sort controls carry over and
   * are re-run against the new pair; a sort the new pair cannot run falls back
   * to Text, so the controls never claim a sort that is not the one on screen.
   */
  describe('rerunCarriedSort', () => {
    it('re-runs the carried text query', () => {
      sortState.setTextQuery('aaa');

      runner.rerunCarriedSort(true);

      expect(httpMock.expectOne('/api/sort').request.body).toEqual({ text: 'aaa' });
      expect(sortState.sortMode).toBe('text');
    });

    it('ranks nothing when the carried text box is empty', () => {
      runner.rerunCarriedSort(true);

      httpMock.expectNone('/api/sort');
    });

    it('does not fire a text sort the dataset\'s embedder cannot run', () => {
      sortState.setTextQuery('aaa');

      runner.rerunCarriedSort(false);

      httpMock.expectNone('/api/sort');
    });

    it('re-runs a learned sort when the new detector can train one', () => {
      enableLearnedSort();
      sortState.setSortMode('learned');

      runner.rerunCarriedSort(true);

      httpMock.expectOne('/api/learned-sort');
      expect(sortState.sortMode).toBe('learned');
    });

    it('falls back to Text when the new detector cannot train a learned sort', () => {
      sortState.setSortMode('learned');
      sortState.setTextQuery('aaa');

      runner.rerunCarriedSort(true);

      httpMock.expectNone('/api/learned-sort');
      expect(sortState.sortMode).toBe('text');
      expect(httpMock.expectOne('/api/sort').request.body).toEqual({ text: 'aaa' });
    });

    it('re-scores with the same detector for a detector Load sort', () => {
      runner.onModelSelected('det-x');
      httpMock.expectOne('/api/find-label').flush({ results: [], threshold: 0.5, detector_name: 'X' });
      expect(sortState.loadSortSource).toEqual({ kind: 'detector', detectorId: 'det-x' });

      runner.rerunCarriedSort(true);

      expect(httpMock.expectOne('/api/find-label').request.body).toEqual({ detector_id: 'det-x' });
      expect(sortState.sortMode).toBe('load');
    });

    it('re-runs server example files', () => {
      sortState.setSortMode('load');
      sortState.setLoadSortSource({ kind: 'files', filenames: ['a.wav', 'b.wav'] });

      runner.rerunCarriedSort(true);

      expect(httpMock.expectOne('/api/example-sort-server').request.body).toEqual({
        filenames: ['a.wav', 'b.wav'],
      });
    });

    it('re-uploads an uploaded example', () => {
      const file = new File(['x'], 'ex.wav', { type: 'audio/wav' });
      sortState.setSortMode('load');
      sortState.setLoadSortSource({ kind: 'upload', file });

      runner.rerunCarriedSort(true);

      httpMock.expectOne('/api/example-sort').flush({ results: [{ id: 3, similarity: 0.7 }], threshold: 0.5 });
      expect(sortState.sortOrder).toEqual([{ id: 3, score: 0.7, bestRegion: undefined }]);
      // The re-run keeps its recipe, so it can carry over again.
      expect(sortState.loadSortSource).toEqual({ kind: 'upload', file, cropParams: undefined });
    });

    it('re-runs "Sort by this" while its dataset is still the active one', () => {
      TestBed.inject(ActiveContextService).setActive('ds1', 'det1');
      runner.runExampleSortById(4, 'clip.wav');
      httpMock.expectOne('/api/example-sort-by-id').flush({ results: [], threshold: 0.5 });
      TestBed.inject(ActiveContextService).setActive('ds1', 'det2');

      runner.rerunCarriedSort(true);

      expect(httpMock.expectOne('/api/example-sort-by-id').request.body).toEqual({ media_id: 4 });
    });

    it('falls back to Text for "Sort by this" on another dataset, whose ids mean nothing here', () => {
      TestBed.inject(ActiveContextService).setActive('ds1', 'det1');
      runner.runExampleSortById(4, 'clip.wav');
      httpMock.expectOne('/api/example-sort-by-id').flush({ results: [], threshold: 0.5 });
      sortState.setTextQuery('aaa');
      TestBed.inject(ActiveContextService).setActive('ds2', 'det1');

      runner.rerunCarriedSort(true);

      httpMock.expectNone('/api/example-sort-by-id');
      expect(sortState.sortMode).toBe('text');
      expect(sortState.loadSortLabel).toBe('');
      expect(sortState.loadSortSource).toBeNull();
      expect(httpMock.expectOne('/api/sort').request.body).toEqual({ text: 'aaa' });
    });

    it('falls back to Text for a Load ranking with no recorded source', () => {
      sortState.setSortMode('load');
      sortState.setLoadSortLabel('Find detector');

      runner.rerunCarriedSort(true);

      expect(sortState.sortMode).toBe('text');
      expect(sortState.loadSortLabel).toBe('');
    });
  });
});
