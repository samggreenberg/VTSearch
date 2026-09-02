import { Injectable, inject, signal } from '@angular/core';
import { Subscription } from 'rxjs';
import { ProgressEventsService } from './progress-events.service';
import { formatProgressMessage } from '../utils/format-progress';
import type { ProgressEvent } from '../models/api.models';

export type SortMode = 'text' | 'learned' | 'load';
export type SelectMode = 'top' | 'hard' | 'new';

export interface SortedItem {
  id: number;
  score: number;
  /** Normalised [x0, y0, x1, y1] of the region the embedder/MLP matched best.
   *  Only set when the active dataset's embedder reports
   *  ``supports_patch_regions``. */
  bestRegion?: number[];
}

/**
 * Shared sort state. Backed by signals so a write from any context — an HTTP
 * subscribe, a poll timer, a learned-sort job continuation — notifies Angular's
 * scheduler and repaints the views that bind these getters, with no zone.js
 * involved.
 *
 * Each value is exposed as a *value-returning getter* over a private signal, so
 * existing `sortState.sortBusy` template bindings stay byte-for-byte the same
 * yet become reactive under zoneless: a signal read through a getter during
 * template evaluation is tracked as a dependency of that view (proven by
 * `testing/getter-signal-zoneless.spec.ts`). Imperative callers read the same
 * getters; mutation goes through the `set*` methods.
 */
@Injectable({ providedIn: 'root' })
export class SortStateService {
  private readonly progressEvents = inject(ProgressEventsService);

  private readonly _sortMode = signal<SortMode>('text');
  private readonly _selectMode = signal<SelectMode>('top');
  private readonly _sortOrder = signal<SortedItem[] | null>(null);
  private readonly _threshold = signal<number | null>(null);
  // The rank position Autopilot's Hard / New picks sample around. Since #2876
  // this is a *different* cut from `_threshold`: the decision line the user sees
  // stays where it is, while acquisition samples further up the ranking, which
  // buys 4.5x the positives per 100 votes at lower cost. Only the learned sort
  // carries one; every other sort leaves it null and the picks fall back to
  // `_threshold`, which is what they always used.
  private readonly _acqThreshold = signal<number | null>(null);
  private readonly _sortBusy = signal(false);
  private readonly _sortStatus = signal('');
  private readonly _sortProgress = signal(0);
  private readonly _sortProgressTotal = signal(0);
  // Whole-job fraction (0..1) and overall ETA for multi-step scoring jobs
  // (Find / detector sort), so the bar fills once across all phases instead of
  // resetting per phase. ``null`` for single-phase sorts (fall back to
  // current/total). See ProgressEvent.overall / ProgressTracker._compute_overall.
  private readonly _sortOverall = signal<number | null>(null);
  // Whole-job fraction at which the current step's slice ends; with
  // `_sortOverall` parked at the slice floor during a count-less step, the
  // pair bounds the pulsing zone. See ProgressEvent.overall_step_end.
  private readonly _sortStepEnd = signal<number | null>(null);
  private readonly _sortEtaSeconds = signal<number | null>(null);
  private readonly _inclusion = signal(0);
  private readonly _loadSortLabel = signal('');
  private readonly _textQuery = signal('');
  // Windowed-sort model (scalability.md S3/S17/S19). At scale the backend sends
  // only a head window of the ranking; `_sortOrder` holds the *loaded* window,
  // `_sortTotal` the full ranking length, `_sortHasMore` whether more rows can be
  // paged in via `_sortToken` (GET /api/sort/page). Below the backend's window
  // threshold the whole ranking arrives at once: total == sortOrder.length,
  // hasMore false, token null — identical to the pre-windowing behaviour.
  private readonly _sortTotal = signal(0);
  private readonly _sortHasMore = signal(false);
  private readonly _sortToken = signal<string | null>(null);
  private readonly _aboveThreshold = signal(0);

  get sortMode(): SortMode {
    return this._sortMode();
  }

  get selectMode(): SelectMode {
    return this._selectMode();
  }

  get sortOrder(): SortedItem[] | null {
    return this._sortOrder();
  }

  get threshold(): number | null {
    return this._threshold();
  }

  /**
   * The cut Autopilot's Hard / New picks sample around, falling back to the
   * reporting `threshold` when the sort carried no acquisition cut. Everything
   * the *user* is shown — the green/red line, the above-threshold count, the
   * Find verdicts — reads `threshold`, never this.
   */
  get acqThreshold(): number | null {
    return this._acqThreshold() ?? this._threshold();
  }

  get sortBusy(): boolean {
    return this._sortBusy();
  }

  get sortStatus(): string {
    return this._sortStatus();
  }

  get sortProgress(): number {
    return this._sortProgress();
  }

  get sortProgressTotal(): number {
    return this._sortProgressTotal();
  }

  get sortOverall(): number | null {
    return this._sortOverall();
  }

  get sortStepEnd(): number | null {
    return this._sortStepEnd();
  }

  get sortEtaSeconds(): number | null {
    return this._sortEtaSeconds();
  }

  get inclusion(): number {
    return this._inclusion();
  }

  get loadSortLabel(): string {
    return this._loadSortLabel();
  }

  get textQuery(): string {
    return this._textQuery();
  }

  /** Full ranking length (>= sortOrder.length once windowed). */
  get sortTotal(): number {
    return this._sortTotal();
  }

  /** True when more rows can be paged in beyond the loaded window. */
  get sortHasMore(): boolean {
    return this._sortHasMore();
  }

  /** Paging handle for GET /api/sort/page (the sort-generation token). */
  get sortToken(): string | null {
    return this._sortToken();
  }

  /** Count of rows scoring at or above the threshold across the whole ranking. */
  get aboveThreshold(): number {
    return this._aboveThreshold();
  }

  setSortMode(mode: SortMode): void {
    this._sortMode.set(mode);
  }

  setSelectMode(mode: SelectMode): void {
    this._selectMode.set(mode);
  }

  setSortResults(order: SortedItem[], threshold: number): void {
    this._sortOrder.set(order);
    this._threshold.set(threshold);
    // No acquisition cut on this path (load-sort restore, tests): the getter
    // falls back to the reporting threshold.
    this._acqThreshold.set(null);
    // A plain (non-windowed) result set: the whole ranking is present, so there
    // is nothing more to page. Keeps callers that don't carry window metadata
    // (load-sort, tests) behaving exactly as before.
    this._sortTotal.set(order.length);
    this._sortHasMore.set(false);
    this._sortToken.set(null);
    this._aboveThreshold.set(order.filter((i) => i.score >= threshold).length);
  }

  /**
   * Install the first window of a (possibly windowed) sort response. When the
   * backend windowed the ranking, `items` is the head window, `total` the full
   * length, and `hasMore` true with a `token` for paging the rest; otherwise
   * `items` is the whole ranking (`hasMore` false).
   */
  setSortWindow(win: {
    items: SortedItem[];
    threshold: number;
    acqThreshold?: number | null;
    total: number;
    hasMore: boolean;
    token: string | null;
    aboveThreshold: number;
  }): void {
    this._sortOrder.set(win.items);
    this._threshold.set(win.threshold);
    this._acqThreshold.set(win.acqThreshold ?? null);
    this._sortTotal.set(win.total);
    this._sortHasMore.set(win.hasMore);
    this._sortToken.set(win.token);
    this._aboveThreshold.set(win.aboveThreshold);
  }

  /**
   * Append a further page of rows to the loaded window (from GET
   * /api/sort/page). Updates `hasMore` from the page response; the threshold and
   * total are unchanged by paging.
   */
  appendSortItems(items: SortedItem[], hasMore: boolean): void {
    const current = this._sortOrder() ?? [];
    this._sortOrder.set([...current, ...items]);
    this._sortHasMore.set(hasMore);
  }

  setSortBusy(busy: boolean): void {
    this._sortBusy.set(busy);
  }

  setSortStatus(status: string): void {
    this._sortStatus.set(status);
  }

  setSortProgress(
    current: number,
    total: number,
    overall: number | null = null,
    etaSeconds: number | null = null,
    stepEnd: number | null = null,
  ): void {
    this._sortProgress.set(current);
    this._sortProgressTotal.set(total);
    this._sortOverall.set(overall);
    this._sortEtaSeconds.set(etaSeconds);
    this._sortStepEnd.set(stepEnd);
  }

  // --- detector-scoring progress -------------------------------------------

  private findProgressSub: Subscription | null = null;

  /**
   * Mirror the `find` SSE channel onto the sort status/progress fields for the
   * duration of a detector-scoring run, so a view only has to start and stop
   * the tracking rather than restate what a progress frame means.
   *
   * This lives here because every field a frame writes — `sortStatus`,
   * `sortProgress`, `sortProgressTotal`, `sortOverall`, `sortEtaSeconds`,
   * `sortStepEnd` — is owned by this service. Find and Label previously carried
   * byte-identical copies of the subscription; the dashboard's superficially
   * similar block is *not* one of them (it renders `/api/find` into
   * `DatasetStateService.progressMessage`, a different surface with a different
   * default message and no progress counts), so it deliberately stays where it
   * is.
   *
   * Restarting is idempotent: an already-running subscription is torn down
   * first, so a second scoring run never stacks two writers on the same fields.
   */
  startFindProgressTracking(): void {
    this.stopFindProgressTracking();
    this.findProgressSub = this.progressEvents.find$.subscribe((prog: ProgressEvent) => {
      if (prog.status !== 'running') return;
      this.setSortStatus(formatProgressMessage(prog, 'Scoring with detector…'));
      this.setSortProgress(
        prog.current ?? 0,
        prog.total ?? 0,
        prog.overall ?? null,
        prog.eta_seconds ?? null,
        prog.overall_step_end ?? null,
      );
    });
  }

  /**
   * Stop mirroring the `find` channel. Callers must reach here on the way out
   * (run finished, view destroyed): this service is a root singleton, so a
   * subscription left running would keep writing scoring progress into state
   * that a later, unrelated sort is displaying.
   */
  stopFindProgressTracking(): void {
    this.findProgressSub?.unsubscribe();
    this.findProgressSub = null;
  }

  setInclusion(value: number): void {
    this._inclusion.set(value);
  }

  setLoadSortLabel(label: string): void {
    this._loadSortLabel.set(label);
  }

  setTextQuery(query: string): void {
    this._textQuery.set(query);
  }

  clear(): void {
    this._sortMode.set('text');
    this._selectMode.set('top');
    this._sortOrder.set(null);
    this._threshold.set(null);
    this._acqThreshold.set(null);
    this._sortBusy.set(false);
    this._sortStatus.set('');
    this._sortProgress.set(0);
    this._sortProgressTotal.set(0);
    this._sortOverall.set(null);
    this._sortStepEnd.set(null);
    this._sortEtaSeconds.set(null);
    this._inclusion.set(0);
    this._loadSortLabel.set('');
    this._textQuery.set('');
    this._sortTotal.set(0);
    this._sortHasMore.set(false);
    this._sortToken.set(null);
    this._aboveThreshold.set(0);
  }
}
