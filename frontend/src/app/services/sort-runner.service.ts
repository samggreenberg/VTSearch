import { DestroyRef, Injectable, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { HttpErrorResponse } from '@angular/common/http';
import { of, throwError } from 'rxjs';
import { catchError, filter, take, tap } from 'rxjs/operators';

import { adaptivePoll } from './adaptive-poll';
import { ActiveContextService } from './active-context.service';
import { AutopilotStateService } from './autopilot-state.service';
import { DetectorsFindApiService } from './detectors-find-api.service';
import { MediaStateService } from './media-state.service';
import { PairScopeService } from './pair-scope.service';
import { SortStateService, SortMode, SelectMode, type LoadSortSource } from './sort-state.service';
import { SortingApiService } from './sorting-api.service';
import { ToastService } from './toast.service';
import { VoteStateService } from './vote-state.service';
import { allItemsLabeled } from '../utils/all-labeled';
import { autoSelectNext as pickNextMedia, type AutoSelectPick } from '../utils/auto-select-next';
import type { LearnedSortResponse } from '../generated/api-client/models/learned-sort-response';

/**
 * Runs sorts, and lands the user on the next thing to vote on.
 *
 * **Component-provided** (`providers: [SortRunnerService]` on `vt-label-view`),
 * never `providedIn: 'root'` — for the same reason {@link PairScopeService} is:
 * every request here is torn down by `pairScope.scoped()`, and a pair scope is
 * a *component's*. Hosting these calls on the root-singleton
 * {@link SortStateService} (as issue #3428 originally proposed) would mean
 * either reinventing pair-scoped cancellation inside a singleton or passing a
 * component's scope subject into one; `PairScopeService`'s header records the
 * same trap being declined for `seedInclusion` (#3448).
 *
 * ## Why these two things are one service
 *
 * Every sort here ends by advancing the selection — a ranking nobody is looking
 * at is not a finished sort — so `autoSelectNext` is the tail of the same
 * operation rather than a separate concern that happens to be called nearby.
 * Splitting them would put a callback across the seam at every one of the six
 * completion paths.
 *
 * The *rule* for which media to advance to stays a pure function in
 * `utils/auto-select-next` (digest-pinned against the eval harness by
 * `scripts/check-eval-app-sync.py`'s `autopilot.auto_select_next` mirror);
 * {@link autoSelectNext} below is only the side-effecting half.
 *
 * ## What stays with the view
 *
 * Everything that reads the *session* rather than the sort: Autopilot's phase
 * wiring and its seed-query lookups, the re-sort prompt's vote bookkeeping, and
 * the panel chrome. The view keeps one-line forwarders for the handlers its
 * template binds, so the template and the component's specs are unchanged by
 * this split.
 */
@Injectable()
export class SortRunnerService {
  private readonly sortingApi = inject(SortingApiService);
  private readonly detectorsFindApi = inject(DetectorsFindApiService);
  private readonly sortState = inject(SortStateService);
  private readonly voteState = inject(VoteStateService);
  private readonly mediaState = inject(MediaStateService);
  private readonly autopilotState = inject(AutopilotStateService);
  private readonly pairScope = inject(PairScopeService);
  private readonly activeContext = inject(ActiveContextService);
  private readonly toast = inject(ToastService);
  private readonly destroyRef = inject(DestroyRef);

  /** True while a windowed-sort "Load more" page fetch is in flight. */
  readonly loadingMoreSort = signal(false);

  /** Last coverage-atlas probe came back empty. Only meaningful under the `new`
   *  Select mode, whose pick is a server round-trip rather than a rule over the
   *  loaded window — see {@link fetchDiversityNext}. */
  private readonly diversityExhausted = signal(false);

  /**
   * True when the current Sort + Select has nothing left to advance to: every
   * row in the loaded ranking is labeled (`top` / `hard`), or the coverage
   * atlas has no unseen item left to offer (`new`).
   *
   * This is the state {@link autoSelectNext} silently no-ops in, and the reason
   * it needs a name: the vote-swipe animation pins the outgoing media node
   * off-screen with `forwards` until a *new* item replaces the node, so an
   * advance that finds nothing leaves the centre pane blank with no message and
   * no placeholder — `media()` is still the item just voted on, so the viewer's
   * own "Select a media item" empty state never fires (#3887). The centre panel
   * takes this as its `exhausted` input and says so instead.
   *
   * Derived rather than latched so an undo puts the user straight back to work:
   * un-voting a row makes it unlabeled again, which makes this false again with
   * nothing having to notice. Deliberately false before any sort has landed —
   * an unranked pair is the placeholder state, not an exhausted one.
   */
  readonly queueExhausted = computed(() => {
    const sortOrder = this.sortState.sortOrder;
    if (!sortOrder || sortOrder.length === 0) return false;
    if (this.sortState.selectMode === 'new') return this.diversityExhausted();
    const good = this.voteState.goodVotes;
    const bad = this.voteState.badVotes;
    return !sortOrder.some((s) => !good.has(s.id) && !bad.has(s.id));
  });

  /**
   * True when every item in the *dataset* is labeled — whatever the current
   * sort window holds, and whether or not a sort has ever run.
   *
   * {@link queueExhausted} is deliberately about the loaded ranking, so it is
   * false whenever `sortOrder` is empty. That leaves two ways to end up staring
   * at a pane with nothing in it and nothing saying why (#4028):
   *
   * - **Manual labelling with no sort.** Clicking items out of the left grid
   *   and voting never populates a ranking, so the last vote hits the same
   *   pinned-off-screen blank pane #3887 named, with `queueExhausted` false.
   * - **Coming back to a finished detector.** A fresh entry with no sort to
   *   carry over ranks nothing (see `seedRankingIfUnranked`), so the centre
   *   falls to its "Select a media item" placeholder — which asks the user to pick something when there is
   *   nothing left to pick.
   *
   * Both are the *dataset* being finished rather than the window, and they want
   * a different sentence from the window case: there is no "load more" or
   * "change the sort" that would produce another item.
   *
   * Derived rather than latched, for the same reason {@link queueExhausted} is:
   * an undo un-labels a row and puts the user straight back to work.
   */
  readonly datasetExhausted = computed(() => allItemsLabeled(
    this.mediaState.mediasSignal(),
    this.voteState.goodVotes,
    this.voteState.badVotes,
  ));

  private learnedSortPending = false;

  /** Active learned-sort job id while a training run is in flight. Set in
   *  {@link onLearnedSort} once the backend returns a job id, cleared in
   *  {@link applyLearnedSortResult} / the error/cancel paths. Used by the
   *  Cancel button on the sort progress bar to target the right job. */
  private currentLearnedSortJobId: string | null = null;

  /** Consecutive *transient* learned-sort result-poll failures tolerated before
   *  the run is declared failed. Roughly 10s–40s of unbroken failures at the
   *  poll's 500ms–2000ms cadence: long enough to ride out a backend blip,
   *  short enough that a genuinely unreachable server does not leave the panel
   *  spinning on 'Training…'. Terminal statuses (404/500) end the run at once
   *  and never consume this budget. */
  private readonly POLL_ERROR_LIMIT = 20;

  /**
   * Step 2 of the pair-change reset — see `PairScopeService.resetForNewPair`.
   *
   * The sorts below carry no `finalize`, so the busy flag, the job id and the
   * progress feed they own survive the supersede that kills their
   * subscriptions. This drops all three. It must run *after* the supersede, so
   * nothing can re-set them; passing it as `resetForNewPair`'s `quiesce` hook
   * is what guarantees that ordering.
   */
  quiesce(): void {
    this.sortState.stopFindProgressTracking();
    this.currentLearnedSortJobId = null;
    this.sortState.setSortBusy(false);
  }

  // --- Sort handlers ---

  onSortModeChange(mode: SortMode): void {
    this.sortState.setSortMode(mode);
    this.autoSelectNext();
  }

  /**
   * Install a (possibly windowed) sort response into the sort state. Handles
   * both `similarity` (text/example sort) and `score` (learned/detector sort)
   * result rows, and carries the window metadata (`total` / `has_more_below` /
   * `sort_token`) so the media-list can page deeper. Below the backend's window
   * threshold the whole ranking arrives and `has_more_below` is false —
   * behaviour is identical to the pre-windowing full-list path.
   */
  private applySortWindow(response: {
    results?: Array<Record<string, unknown>>;
    threshold?: number;
    acq_threshold?: number | null;
    total?: number;
    above_threshold?: number;
    has_more_below?: boolean;
    sort_token?: string;
  }): void {
    const threshold = response.threshold ?? 0;
    const items = (response.results ?? []).map((r) => ({
      id: r['id'] as number,
      score: (r['score'] ?? r['similarity'] ?? 0) as number,
      bestRegion: r['best_region'] as number[] | undefined,
    }));
    this.sortState.setSortWindow({
      items,
      threshold,
      acqThreshold: response.acq_threshold ?? null,
      total: response.total ?? items.length,
      hasMore: response.has_more_below ?? false,
      token: response.sort_token ?? null,
      aboveThreshold: response.above_threshold ?? items.filter((i) => i.score >= threshold).length,
    });
  }

  /**
   * Page in the next window of a windowed ranking (the media-list "Load more"
   * trigger). Fetches from the sort token at the current loaded offset and
   * appends. A failed/expired token just stops paging (the user can re-sort).
   */
  onLoadMore(): void {
    const token = this.sortState.sortToken;
    if (!token || !this.sortState.sortHasMore || this.loadingMoreSort()) return;
    this.loadingMoreSort.set(true);
    const offset = this.sortState.sortOrder?.length ?? 0;
    this.sortingApi
      .getSortPage(token, offset, 200)
      .pipe(this.pairScope.scoped())
      .subscribe({
        next: (page) => {
          const items = (page.results ?? []).map((r) => ({
            id: r['id'] as number,
            score: (r['score'] ?? r['similarity'] ?? 0) as number,
            bestRegion: r['best_region'] as number[] | undefined,
          }));
          this.sortState.appendSortItems(items, page.has_more);
          this.loadingMoreSort.set(false);
        },
        error: () => this.loadingMoreSort.set(false),
      });
  }

  /**
   * @param autoSelect Whether the finished ranking may move the centre viewer.
   *                   False on the pair-switch path, where the selection is the
   *                   view's pair-change seed effect to place (#3510).
   */
  onTextSort(text: string, autoSelect = true): void {
    this.sortState.setTextQuery(text);
    this.sortState.setSortBusy(true);
    this.sortState.setSortStatus('Sorting…');
    this.sortingApi.sort({ text }).pipe(this.pairScope.scoped()).subscribe({
      next: (response) => {
        this.applySortWindow(response);
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('');
        if (autoSelect) this.autoSelectNext();
      },
      error: () => {
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('Sort failed');
      },
    });
  }

  onLearnedSort(autoSelect = true): void {
    if (!this.voteState.learnedSortAvailable) return;
    this.sortState.setSortBusy(true);
    this.sortState.setSortStatus('Training…');
    this.sortingApi.learnedSort().pipe(this.pairScope.scoped()).subscribe({
      next: (response) => {
        if (response.status === 'done') {
          this.applyLearnedSortResult(response, autoSelect);
        } else if (response.status === 'running') {
          this.currentLearnedSortJobId = response.job_id;
          this.pollLearnedSortJob(response.job_id, autoSelect);
        } else {
          this.sortState.setSortBusy(false);
          this.sortState.setSortStatus(response.error || 'Training failed');
        }
      },
      error: () => {
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('Training failed');
      },
    });
  }

  /**
   * Poll a running learned-sort job until it settles.
   *
   * Uses {@link adaptivePoll}, not the `timer(200, 500)` + `switchMap` pattern
   * this once had: `switchMap` aborted the in-flight result GET on every tick,
   * so a backend that needed longer than the interval to answer — exactly the
   * situation while an MLP training job is hogging the process — had *every*
   * read cancelled, never saw a non-running status, and left the panel stuck
   * on 'Training…' with `sortBusy` true forever. That is the pathology
   * documented in `adaptive-poll.ts` (issue #2572) that the labeling-status
   * poll was already migrated off; this poll was left behind.
   *
   * Poll failures are no longer fatal either. The result endpoint reports two
   * genuine terminal states by HTTP status code — 404 (job evicted or unknown)
   * and 500 (the job itself errored) — so those still end the run, but any
   * other failure (a network blip, a proxy 502/503) is transient and costs
   * only that tick, until {@link POLL_ERROR_LIMIT} consecutive failures say
   * the backend is really gone. Previously a single transient error tore the
   * poll down and reported 'Training failed' for a job still running
   * server-side.
   */
  private pollLearnedSortJob(jobId: string, autoSelect: boolean): void {
    let consecutiveErrors = 0;
    const settledWith = (error: string): LearnedSortResponse => ({
      job_id: jobId,
      status: 'error',
      error,
    });

    adaptivePoll<LearnedSortResponse>(
      () =>
        this.sortingApi.getLearnedSortResult(jobId).pipe(
          tap(() => (consecutiveErrors = 0)),
          catchError((err: unknown) => {
            const status = err instanceof HttpErrorResponse ? err.status : 0;
            if (status === 404) return of(settledWith('Training job expired'));
            if (status === 500) return of(settledWith('Training failed'));
            consecutiveErrors += 1;
            if (consecutiveErrors >= this.POLL_ERROR_LIMIT) {
              return of(settledWith('Training failed'));
            }
            // Re-throw so adaptivePoll absorbs it: this tick is skipped and the
            // next one scheduled as usual, rather than the poll tearing down.
            return throwError(() => err);
          }),
        ),
      { fastMs: 500, slowMs: 2000 },
    )
      .pipe(
        // Pair-scoped: a training job can outlive the pair it was started for,
        // and its result must not be applied to whatever pair is active when it
        // finally settles (see `PairScopeService`).
        this.pairScope.scoped(),
        filter((res) => res.status !== 'running'),
        take(1),
      )
      // No `error` handler: adaptivePoll never errors — a request failure is
      // either absorbed above or converted into a terminal `error` status.
      .subscribe((res) => {
        if (res.status === 'done') {
          this.applyLearnedSortResult(res, autoSelect);
        } else if (res.status === 'cancelled') {
          this.currentLearnedSortJobId = null;
          this.sortState.setSortBusy(false);
          this.sortState.setSortStatus('Cancelled');
        } else {
          this.currentLearnedSortJobId = null;
          this.sortState.setSortBusy(false);
          this.sortState.setSortStatus(res.error || 'Training failed');
        }
      });
  }

  private applyLearnedSortResult(response: LearnedSortResponse, autoSelect: boolean): void {
    this.applySortWindow(response);
    this.currentLearnedSortJobId = null;
    this.sortState.setSortBusy(false);
    this.sortState.setSortStatus('');
    if (autoSelect) {
      this.autoSelectNext();
    }
  }

  /** Cancel whatever sort run is currently in flight.
   *
   *  - Learned sort: targets the active ``AsyncJob`` by id.
   *  - Load-sort (find-label): trips the shared ``find_progress`` cancel
   *    flag, which the scoring loop polls.
   *  - Text / example sort: no cancellation endpoint; those calls run
   *    synchronously and complete before the user can usefully cancel.
   */
  onSortCancel(): void {
    if (this.currentLearnedSortJobId) {
      const jobId = this.currentLearnedSortJobId;
      this.currentLearnedSortJobId = null;
      this.sortingApi.cancelLearnedSort(jobId).pipe(takeUntilDestroyed(this.destroyRef)).subscribe();
      return;
    }
    if (this.sortState.sortMode === 'load') {
      this.detectorsFindApi.cancelFind().pipe(takeUntilDestroyed(this.destroyRef)).subscribe();
    }
  }

  onModelSelected(modelId: string, autoSelect = true): void {
    if (!modelId) return;
    this.sortState.setSortMode('load');
    this.sortState.setLoadSortSource({ kind: 'detector', detectorId: modelId });
    this.sortState.setSortBusy(true);
    this.sortState.setSortStatus('Scoring with detector…');
    this.sortState.setSortProgress(0, 0);

    this.sortState.startFindProgressTracking();

    // Pair-scoped: scoring runs for minutes on a large dataset, so a pair switch
    // mid-run must kill this before it ranks the new pair with old scores.
    this.detectorsFindApi.findLabel({ detector_id: modelId }).pipe(this.pairScope.scoped()).subscribe({
      next: (raw) => {
        const response = raw as {
          results: { id: number; score: number; best_region?: number[] }[];
          threshold: number;
          detector_name?: string;
        };
        this.sortState.stopFindProgressTracking();
        this.applySortWindow(response);
        this.sortState.setLoadSortLabel(response.detector_name || 'Detector');
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('');
        this.sortState.setSortProgress(0, 0);
        if (autoSelect) this.autoSelectNext();
      },
      error: () => {
        this.sortState.stopFindProgressTracking();
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('Detector sort failed');
        this.sortState.setSortProgress(0, 0);
      },
    });
  }

  /**
   * Install an example sort the Load modal already ran. The modal attaches the
   * `source` it ranked against, so the sort can be re-run on another pair
   * (#4092); a response without one simply can't be.
   */
  onExampleSortStarted(data: unknown, autoSelect = true): void {
    const response = data as {
      results: { id: number; similarity: number; best_region?: number[] }[];
      threshold: number;
      source?: LoadSortSource;
    };
    this.sortState.setSortMode('load');
    this.applySortWindow(response);
    this.sortState.setLoadSortLabel('Example media');
    this.sortState.setLoadSortSource(response.source ?? null);
    this.sortState.setSortBusy(false);
    this.sortState.setSortStatus('');
    if (autoSelect) this.autoSelectNext();
  }

  /** Re-run an uploaded example sort (the Load modal's upload path) against
   *  the pair now active. */
  private uploadExampleSort(file: File, cropParams: Record<string, unknown> | undefined, autoSelect: boolean): void {
    this.sortState.setSortBusy(true);
    this.sortState.setSortStatus('Sorting by example…');
    this.sortingApi.exampleSort(file, cropParams).pipe(this.pairScope.scoped()).subscribe({
      next: (response) => this.onExampleSortStarted({ ...response, source: { kind: 'upload', file, cropParams } }, autoSelect),
      error: () => {
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('Example sort failed');
      },
    });
  }

  /**
   * Rank the haystack against one media item (the right-click "Sort by this"
   * action, and the crop overlay's confirm).
   *
   * @param label Display name for the sort bar. Passed in rather than derived
   *              here because the view already owns that lookup for its vote
   *              toasts, and it is not a sort concern.
   */
  runExampleSortById(
    mediaId: number,
    label: string,
    cropParams?: Record<string, unknown>,
    autoSelect = true,
  ): void {
    // Captured before the request: the source names the dataset the id belongs
    // to, which is the one active when the sort was asked for.
    const datasetId = this.activeContext.datasetId;
    this.sortState.setSortBusy(true);
    this.sortState.setSortStatus('Sorting by example…');
    this.sortingApi
      .exampleSortById({ media_id: mediaId, crop_params: cropParams })
      .pipe(this.pairScope.scoped())
      .subscribe({
        next: (response) => {
          this.sortState.setSortMode('load');
          this.applySortWindow(response);
          this.sortState.setLoadSortLabel(label);
          this.sortState.setLoadSortSource({ kind: 'media', datasetId, mediaId, cropParams });
          this.sortState.setSortBusy(false);
          this.sortState.setSortStatus('');
          if (autoSelect) this.autoSelectNext();
        },
        error: (err) => {
          this.sortState.setSortBusy(false);
          this.sortState.setSortStatus('Example sort failed');
          this.toast.error({ message: err?.error?.message || 'Example sort failed' });
        },
      });
  }

  /**
   * Rank the haystack against the detector's seed examples (Autopilot's "good"
   * phase on a media-seeded detector).
   *
   * Every example seeds the sort: plural examples rank against the centroid of
   * their embeddings, so the phase surfaces items resembling what the examples
   * have in common.
   */
  exampleSortByFilenames(filenames: string[], autoSelect = true): void {
    if (filenames.length === 0) return;
    this.sortState.setSortBusy(true);
    this.sortState.setSortStatus(filenames.length > 1 ? 'Sorting by examples…' : 'Sorting by example…');
    this.sortingApi.exampleSortServer({ filenames }).pipe(this.pairScope.scoped()).subscribe({
      next: (response) => {
        this.applySortWindow(response);
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('');
        this.sortState.setSortMode('load');
        this.sortState.setLoadSortSource({ kind: 'files', filenames });
        if (autoSelect) this.autoSelectNext();
      },
      error: () => {
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('Example sort failed');
      },
    });
  }

  // --- Carrying a sort to a new pair ---

  /**
   * Re-run the sort the controls show against the pair now on screen (#4092).
   *
   * The rule for a Train entry or a pair switch is "the sort carries over, the
   * ranking does not": Sort mode, Select mode and the text query are the user's
   * and survive the move, while the ranking — a list of per-dataset media ids —
   * is always dropped and re-derived here by running the same sort again. Before
   * this, the controls carried over but nothing re-ran them, so a Train window on
   * a new pair showed the last session's query and mode above an empty (or,
   * from the dashboard, foreign) ranking.
   *
   * A sort the new pair cannot run falls back to Text — the controls must never
   * claim a sort that is not the one on screen:
   *
   * - **Learned** needs a good and a bad label for the new detector; its radio
   *   is disabled without them.
   * - **Load** needs a {@link LoadSortSource} it can re-run. A "Sort by this"
   *   media id means nothing outside its own dataset, and a ranking installed
   *   with no recorded source (Find's scoring) has nothing to re-run.
   *
   * Text then re-runs the carried query, when there is one and the dataset's
   * embedder can embed text; otherwise the pair is left unranked, as a fresh
   * entry with an empty box would be.
   *
   * Every sort runs with `autoSelect: false`: the caller owns seeding the centre
   * viewer, and must not move someone who has already started labelling.
   *
   * @param textSupported Whether the active dataset's embedder embeds text.
   */
  rerunCarriedSort(textSupported: boolean): void {
    const mode = this.sortState.sortMode;
    if (mode === 'learned' && this.voteState.learnedSortAvailable) {
      this.onLearnedSort(false);
      return;
    }
    if (mode === 'load' && this.rerunLoadSource(this.sortState.loadSortSource)) return;
    if (mode !== 'text') {
      this.sortState.setSortMode('text');
      this.sortState.setLoadSortLabel('');
      this.sortState.setLoadSortSource(null);
    }
    const query = this.sortState.textQuery.trim();
    if (query && textSupported) this.onTextSort(query, false);
  }

  /** Re-run a Load sort's source; false when it can't be re-run here. */
  private rerunLoadSource(source: LoadSortSource | null): boolean {
    switch (source?.kind) {
      case 'detector':
        this.onModelSelected(source.detectorId, false);
        return true;
      case 'files':
        this.exampleSortByFilenames(source.filenames, false);
        return true;
      case 'upload':
        this.uploadExampleSort(source.file, source.cropParams, false);
        return true;
      case 'media':
        if (source.datasetId !== this.activeContext.datasetId) return false;
        this.runExampleSortById(source.mediaId, this.sortState.loadSortLabel, source.cropParams, false);
        return true;
      default:
        return false;
    }
  }

  // --- Select mode ---

  onSelectModeChange(mode: SelectMode): void {
    this.sortState.setSelectMode(mode);
    this.autoSelectNext();
  }

  private fetchDiversityNext(): void {
    const sortOrder = this.sortState.sortOrder;
    const scores = sortOrder
      ? Object.fromEntries(sortOrder.map((s) => [String(s.id), s.score]))
      : undefined;
    this.sortingApi
      // The New pick reads the threshold as a sampling position too (it steers
      // the atlas probe by a node's median score), so it takes the acquisition
      // cut alongside the Hard pick.
      .getCoverageAtlasNext(scores, this.sortState.acqThreshold ?? undefined)
      .pipe(this.pairScope.scoped())
      .subscribe({
        next: (response) => {
          this.diversityExhausted.set(response.id === null);
          if (response.id !== null) {
            this.mediaState.selectMedia(response.id);
          }
          if (typeof response.coverage_level === 'number') {
            this.autopilotState.updateDiversityLevel(response.coverage_level);
          }
        },
      });
  }

  // --- Inclusion ---

  onInclusionChange(value: number): void {
    this.sortState.setInclusion(value);
    this.sortingApi.setInclusion(value).pipe(this.pairScope.scoped()).subscribe();
    this.autoSelectNext();
    if (this.sortState.sortMode === 'learned' && this.voteState.learnedSortAvailable) {
      this.scheduleLearnedSort(false);
    }
  }

  /** Coalesce a flurry of re-rank triggers (a vote, an inclusion drag) into one
   *  learned sort 300ms after the last of them. */
  scheduleLearnedSort(autoSelect = true): void {
    if (this.learnedSortPending) return;
    this.learnedSortPending = true;
    setTimeout(() => {
      this.learnedSortPending = false;
      this.onLearnedSort(autoSelect);
    }, 300);
  }

  // --- Selection advance ---

  /**
   * Advance to the next media the current Sort + Select says to show.
   *
   * The rule itself lives in {@link pickNextMedia} (`utils/auto-select-next`)
   * as a pure function so it can be unit-tested and so the eval harness's copy
   * of it can be digest-pinned; this method is the side-effecting half —
   * applying the selection, or firing the coverage-atlas probe the `new` mode
   * asks for.
   */
  /**
   * What {@link autoSelectNext} *would* pick, with nothing applied.
   *
   * Exists so the image prefetch (#3896) can warm the next item during the
   * reviewer's think time through the same rule that will later select it —
   * a second copy of the rule would prefetch the wrong item every time the
   * two drifted, which is worse than not prefetching at all.
   */
  peekNextMedia(excludeId?: number): AutoSelectPick {
    return pickNextMedia({
      sortOrder: this.sortState.sortOrder,
      selectMode: this.sortState.selectMode,
      acqThreshold: this.sortState.acqThreshold,
      goodVotes: this.voteState.goodVotes,
      badVotes: this.voteState.badVotes,
      excludeId,
    });
  }

  /**
   * The next `depth` items the auto-advance would show if the reviewer voted on
   * `currentId` and then on each pick in turn, with the ranking unchanged —
   * the prefetch's queue (#3896).
   *
   * Each step is {@link peekNextMedia}'s rule with the earlier picks counted as
   * voted, so the first entry is exactly what the next vote selects. The later
   * ones are a forecast: a learned re-sort landing between votes can reorder
   * them, which is why the review views recompute this whenever the ranking or
   * the votes change rather than only when the selection does. Stops at the
   * first non-`media` pick: a `new`-mode advance is a server round-trip, and
   * nothing past it is knowable.
   *
   * Reads the sort and vote signals, so a caller inside an `effect` re-runs
   * when any of them moves.
   */
  peekUpcomingMedia(currentId: number | null, depth: number): number[] {
    if (currentId === null) return [];
    const sortOrder = this.sortState.sortOrder;
    const selectMode = this.sortState.selectMode;
    const acqThreshold = this.sortState.acqThreshold;
    const badVotes = this.voteState.badVotes;
    let goodVotes: ReadonlySet<number> = this.voteState.goodVotes;
    let excludeId = currentId;
    const upcoming: number[] = [];
    while (upcoming.length < depth) {
      const pick = pickNextMedia({ sortOrder, selectMode, acqThreshold, goodVotes, badVotes, excludeId });
      if (pick.kind !== 'media') break;
      upcoming.push(pick.id);
      // Count the item just stepped past as voted (its polarity is irrelevant
      // to every mode's rule) and step onto the pick.
      goodVotes = new Set(goodVotes).add(excludeId);
      excludeId = pick.id;
    }
    return upcoming;
  }

  autoSelectNext(excludeId?: number): void {
    const pick = this.peekNextMedia(excludeId);
    if (pick.kind === 'media') {
      this.mediaState.selectMedia(pick.id);
      this.diversityExhausted.set(false);
    } else if (pick.kind === 'diversity') {
      // The probe below is the only thing that can answer for the `new` mode, so
      // clear the previous answer rather than letting a stale "empty" latch
      // across the round-trip.
      this.diversityExhausted.set(false);
      this.fetchDiversityNext();
    }
  }
}
