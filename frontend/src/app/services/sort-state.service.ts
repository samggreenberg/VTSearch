import { Injectable, signal } from '@angular/core';

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
 * involved (docs/plans/zoneless-migration.md, Phase 2.5 / Recipe B).
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
  private readonly _sortMode = signal<SortMode>('text');
  private readonly _selectMode = signal<SelectMode>('top');
  private readonly _sortOrder = signal<SortedItem[] | null>(null);
  private readonly _threshold = signal<number | null>(null);
  private readonly _sortBusy = signal(false);
  private readonly _sortStatus = signal('');
  private readonly _sortProgress = signal(0);
  private readonly _sortProgressTotal = signal(0);
  // Whole-job fraction (0..1) and overall ETA for multi-step scoring jobs
  // (Find / detector sort), so the bar fills once across all phases instead of
  // resetting per phase. ``null`` for single-phase sorts (fall back to
  // current/total). See ProgressEvent.overall / ProgressTracker._compute_overall.
  private readonly _sortOverall = signal<number | null>(null);
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
    total: number;
    hasMore: boolean;
    token: string | null;
    aboveThreshold: number;
  }): void {
    this._sortOrder.set(win.items);
    this._threshold.set(win.threshold);
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
  ): void {
    this._sortProgress.set(current);
    this._sortProgressTotal.set(total);
    this._sortOverall.set(overall);
    this._sortEtaSeconds.set(etaSeconds);
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
    this._sortBusy.set(false);
    this._sortStatus.set('');
    this._sortProgress.set(0);
    this._sortProgressTotal.set(0);
    this._sortOverall.set(null);
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
