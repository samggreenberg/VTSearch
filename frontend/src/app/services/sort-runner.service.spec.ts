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

  // --- inclusion ------------------------------------------------------------

  it('pushes the inclusion value and re-advances the selection', () => {
    sortState.setSelectMode('top');
    sortState.setSortResults([{ id: 5, score: 0.9 }], 0.5);

    runner.onInclusionChange(0.25);

    expect(sortState.inclusion).toBe(0.25);
    httpMock.expectOne('/api/inclusion').flush({ inclusion: 0.25 });
    expect(mediaState.selectedId()).toBe(5);
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
