import {
  Component,
  Input,
  Output,
  EventEmitter,
  ElementRef,
  ViewChild,
  AfterViewChecked,
  ChangeDetectorRef,
  NgZone,
  OnChanges,
  OnDestroy,
  OnInit,
  SimpleChanges,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ScrollingModule, CdkVirtualScrollViewport } from '@angular/cdk/scrolling';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { MediaItemComponent } from '../media-item/media-item.component';
import { Media } from '../../../models/api.models';
import { MediaMetadataCacheService } from '../../../services/media-metadata-cache.service';
import { MediaStateService } from '../../../services/media-state.service';
import { SkeletonComponent } from '../../skeleton/skeleton.component';
import { SortedItem } from '../left-panel.component';
import { prefersReducedMotion } from '../../../utils/reduced-motion';

/**
 * Item count above which list mode switches from plain DOM to CDK virtual
 * scrolling.  Each list row still mounts a thumbnail ``<img>`` and its
 * component, so a few hundred rendered at once blocks the main thread for
 * ~1s.  Kept low enough that entering a view or switching to list never
 * freezes; small lists stay plain DOM so ``scrollIntoView`` stays exact.
 */
const VIRTUAL_SCROLL_THRESHOLD = 150;
/**
 * Item count above which grid mode switches to CDK virtual scrolling.  Grid
 * cards are even heavier than list rows, so this is lower still — a few
 * hundred thumbnails rendered at once froze the gallery for ~1.8s.
 */
const GRID_VIRTUAL_THRESHOLD = 80;
/** Approximate height of a single media-item row in list mode (px). */
const LIST_ITEM_HEIGHT = 28;
/** Horizontal/vertical gap between grid cards (px); mirrors ``--space-xs``. */
const GRID_GAP_PX = 4;
/** Fallback grid-row stride before the first real card is measured (px). */
const GRID_ROW_HEIGHT_FALLBACK = 100;
/**
 * Smallest plausible measured grid-row stride (px).  A card mid-relayout
 * (its ``<img>`` still loading after a zoom) can momentarily report a
 * near-zero ``getBoundingClientRect().height``.  Accepting that as the CDK
 * ``itemSize`` would make the viewport think each row is a few pixels tall and
 * mount nearly every item at once — defeating virtualization and decoding
 * hundreds of images simultaneously.  Heights below this floor are treated as
 * "not yet laid out" and ignored until a real card measures.
 */
const MIN_GRID_ROW_HEIGHT = 24;
/** Extra items to prefetch beyond the visible viewport edges. */
const PREFETCH_BUFFER = 50;

/** One ordered entry in the flat media list. */
interface OrderedItem {
  media: Media;
  score: number | null;
  showThreshold: boolean;
  bestRegion: number[] | null;
}

/** One virtualized grid row: a run of cards, or the full-width threshold marker. */
type GridRow = { kind: 'items'; items: OrderedItem[] } | { kind: 'threshold' };

@Component({
  selector: 'vt-media-list',
  standalone: true,
  imports: [CommonModule, ScrollingModule, MediaItemComponent, SkeletonComponent],
  templateUrl: './media-list.component.html',
  styleUrl: './media-list.component.scss',
})
export class MediaListComponent implements OnInit, AfterViewChecked, OnChanges, OnDestroy {
  @Input() medias: Media[] = [];
  @Input() sortOrder: SortedItem[] | null = null;
  @Input() threshold: number | null = null;
  @Input() selectedId: number | null = null;
  @Input() goodVotes: Set<number> = new Set();
  @Input() badVotes: Set<number> = new Set();
  @Input() viewMode: 'grid' | 'list' = 'list';
  @Input() gridGoalWidth: number = 80;
  @Input() focusMode: 'click' | 'hover' = 'click';
  @Input() showScores = true;

  @Output() mediaSelect = new EventEmitter<number>();
  @Output() mediaVote = new EventEmitter<{ id: number; vote: 'good' | 'bad' }>();
  @Output() mediaContextRequest = new EventEmitter<{ id: number; x: number; y: number }>();
  @ViewChild('listContainer') listContainer!: ElementRef<HTMLDivElement>;
  @ViewChild(CdkVirtualScrollViewport) virtualViewport?: CdkVirtualScrollViewport;

  /** Cached ordered items, rebuilt only when inputs change, not on every CD cycle. */
  cachedOrderedItems: OrderedItem[] = [];

  /** Grid mode only: ``cachedOrderedItems`` chunked into rows for virtual scrolling. */
  gridRows: GridRow[] = [];
  /** Number of cards per grid row, derived from the viewport width and goal width. */
  gridColumns = 1;
  /** Measured stride of one virtualized grid row (px); fed to CDK as ``itemSize``. */
  gridRowHeight = GRID_ROW_HEIGHT_FALLBACK;

  readonly listItemHeight = LIST_ITEM_HEIGHT;

  private pendingScrollToSelected = false;
  private pendingScrollPct: number | null = null;
  private readonly destroy$ = new Subject<void>();
  private scrollSubscribed = false;
  private resizeObserver?: ResizeObserver;
  private observedViewportEl?: HTMLElement;
  /** True once ``gridRowHeight`` has been measured from a real rendered card. */
  private gridHeightMeasured = false;

  /** Mirror of ``MediaStateService.loading``; drives the skeleton rows when the
   *  list is empty during the initial /api/medias/ids fetch. */
  loadingMedias = false;
  readonly skeletonPlaceholders = Array.from({ length: 12 });

  constructor(
    private metadataCache: MediaMetadataCacheService,
    private mediaState: MediaStateService,
    private cdr: ChangeDetectorRef,
    private zone: NgZone,
  ) {}

  ngOnInit(): void {
    // When the cache hydrates new items, re-enrich the displayed rows so
    // visible items show their filename / score / metadata.
    this.metadataCache.version$
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => this.rebuildOrderedItems());
    this.mediaState.loading$
      .pipe(takeUntil(this.destroy$))
      .subscribe((loading) => (this.loadingMedias = loading));
  }

  get showSkeletons(): boolean {
    return this.loadingMedias && this.cachedOrderedItems.length === 0;
  }

  ngOnDestroy(): void {
    this.resizeObserver?.disconnect();
    this.destroy$.next();
    this.destroy$.complete();
  }

  /** Whether list mode is virtualizing (many lightweight rows). */
  get useListVirtual(): boolean {
    return this.viewMode === 'list' && this.cachedOrderedItems.length > VIRTUAL_SCROLL_THRESHOLD;
  }

  /** Whether grid mode is virtualizing (many heavy thumbnail cards). */
  get useGridVirtual(): boolean {
    return this.viewMode === 'grid' && this.cachedOrderedItems.length > GRID_VIRTUAL_THRESHOLD;
  }

  /**
   * Whether either mode is virtualizing via a ``CdkVirtualScrollViewport``.
   * Shared plumbing (scroll restoration, ``scrollToIndex``, prefetch) keys off
   * this; the list/grid getters above pick the specific index math.
   */
  get useVirtualScroll(): boolean {
    return this.useListVirtual || this.useGridVirtual;
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['selectedId'] && !changes['selectedId'].firstChange) {
      this.pendingScrollToSelected = true;
    }

    const scrollContainer = this.listContainer?.nativeElement ?? this.virtualViewport?.elementRef.nativeElement;
    if (changes['viewMode'] && !changes['viewMode'].firstChange && scrollContainer) {
      const el = scrollContainer;
      const maxScroll = el.scrollHeight - el.clientHeight;
      this.pendingScrollPct = maxScroll > 0 ? el.scrollTop / maxScroll : 0;
    }

    // A wider/narrower goal width changes how many cards fit per grid row and
    // the height of each card, so recompute columns and re-measure the stride.
    if (changes['gridGoalWidth'] && !changes['gridGoalWidth'].firstChange) {
      this.gridHeightMeasured = false;
      this.recomputeGridColumns();
    }

    // Rebuild cache only when relevant inputs change.
    if (
      changes['medias'] ||
      changes['sortOrder'] ||
      changes['threshold'] ||
      changes['showScores'] ||
      changes['viewMode'] ||
      changes['gridGoalWidth']
    ) {
      this.rebuildOrderedItems();
    }
  }

  private rebuildOrderedItems(): void {
    // ``this.medias`` carries stubs (``{id, type, embedder?}``) from
    // ``/api/medias/ids``.  Look up the cached full metadata for visible
    // rows; fall back to the stub so the row renders immediately and gets
    // upgraded once the batch fetch lands.
    const stubMap = new Map(this.medias.map((m) => [m.id, m]));
    const enrich = (id: number): Media | undefined =>
      this.metadataCache.get(id) ?? stubMap.get(id);

    const items: OrderedItem[] = [];

    if (this.sortOrder && this.sortOrder.length > 0) {
      let thresholdInserted = false;
      for (const sorted of this.sortOrder) {
        const media = enrich(sorted.id);
        if (!media) continue;

        let showThreshold = false;
        if (!thresholdInserted && this.threshold !== null && sorted.score < this.threshold) {
          showThreshold = true;
          thresholdInserted = true;
        }

        items.push({
          media,
          score: this.showScores ? sorted.score : null,
          showThreshold,
          bestRegion: sorted.bestRegion ?? null,
        });
      }
    } else {
      for (const stub of this.medias) {
        const media = this.metadataCache.get(stub.id) ?? stub;
        items.push({ media, score: null, showThreshold: false, bestRegion: null });
      }
    }

    this.cachedOrderedItems = items;
    this.rebuildGridRows();

    // Prefetch metadata for the initial visible window.
    this.prefetchVisibleMetadata();
  }

  /**
   * Chunk ``cachedOrderedItems`` into fixed-width rows for the virtualized
   * grid.  The threshold marker becomes its own full-width row so it still
   * separates predicted-good from predicted-bad cards; every row keeps the
   * same stride so CDK's fixed-size strategy scrolls accurately.
   */
  private rebuildGridRows(): void {
    if (this.viewMode !== 'grid') {
      this.gridRows = [];
      return;
    }
    const cols = Math.max(1, this.gridColumns);
    const rows: GridRow[] = [];
    let current: OrderedItem[] = [];
    const flush = () => {
      if (current.length) {
        rows.push({ kind: 'items', items: current });
        current = [];
      }
    };
    for (const item of this.cachedOrderedItems) {
      if (item.showThreshold) {
        flush();
        rows.push({ kind: 'threshold' });
      }
      current.push(item);
      if (current.length === cols) flush();
    }
    flush();
    this.gridRows = rows;
  }

  /**
   * Recompute how many cards fit across the grid viewport.  Returns ``true``
   * when the column count changed (so callers know to rebuild the rows).
   */
  private recomputeGridColumns(): boolean {
    const el = this.virtualViewport?.elementRef.nativeElement ?? this.listContainer?.nativeElement;
    if (!el) return false;
    const inner = el.clientWidth - 2 * GRID_GAP_PX;
    if (inner <= 0) return false;
    const cols = Math.max(1, Math.floor((inner + GRID_GAP_PX) / (this.gridGoalWidth + GRID_GAP_PX)));
    if (cols === this.gridColumns) return false;
    this.gridColumns = cols;
    return true;
  }

  getVoteLabel(id: number): 'good' | 'bad' | null {
    if (this.goodVotes.has(id)) return 'good';
    if (this.badVotes.has(id)) return 'bad';
    return null;
  }

  onMediaSelect(id: number): void {
    this.mediaSelect.emit(id);
  }

  onMediaVote(event: { id: number; vote: 'good' | 'bad' }): void {
    this.mediaVote.emit(event);
  }

  onMediaContextRequest(event: { id: number; x: number; y: number }): void {
    this.mediaContextRequest.emit(event);
  }

  ngAfterViewChecked(): void {
    // Keep the grid viewport measured: a ResizeObserver tracks width (→ column
    // count) and the first rendered card's height (→ CDK row stride).  Tear it
    // down when we leave grid-virtual mode so the list viewport isn't probed
    // for grid cards that don't exist.
    if (this.useGridVirtual && this.virtualViewport) {
      this.setupGridViewport();
    } else if (this.observedViewportEl) {
      this.resizeObserver?.disconnect();
      this.resizeObserver = undefined;
      this.observedViewportEl = undefined;
    }

    const scrollEl =
      this.virtualViewport?.elementRef.nativeElement ?? this.listContainer?.nativeElement;

    if (this.pendingScrollPct !== null && scrollEl) {
      const pct = this.pendingScrollPct;
      this.pendingScrollPct = null;
      this.pendingScrollToSelected = false;
      const maxScroll = scrollEl.scrollHeight - scrollEl.clientHeight;
      scrollEl.scrollTop = pct * maxScroll;
    } else if (this.pendingScrollToSelected) {
      this.pendingScrollToSelected = false;
      const behavior: ScrollBehavior = prefersReducedMotion() ? 'auto' : 'smooth';
      if (this.useVirtualScroll && this.virtualViewport) {
        // In virtual-scroll mode, find the index and scroll to it.  Grid items
        // are chunked into rows, so map the item index to its row first.
        const idx = this.cachedOrderedItems.findIndex((i) => i.media.id === this.selectedId);
        if (idx >= 0) {
          const target = this.useGridVirtual ? this.itemIndexToRow(idx) : idx;
          this.virtualViewport.scrollToIndex(target, behavior);
        }
      } else if (this.listContainer) {
        const activeEl = this.listContainer.nativeElement.querySelector('.media-item.active');
        if (activeEl) {
          activeEl.scrollIntoView({ block: 'nearest', behavior });
        }
      }
    }

    // Subscribe to virtual viewport scroll events (once the viewport exists).
    if (this.virtualViewport && !this.scrollSubscribed) {
      this.scrollSubscribed = true;
      this.virtualViewport.scrolledIndexChange
        .pipe(takeUntil(this.destroy$))
        .subscribe(() => this.prefetchVisibleMetadata());
    }
  }

  /** Observe the grid viewport for width/height changes (runs outside Angular). */
  private setupGridViewport(): void {
    const vp = this.virtualViewport?.elementRef.nativeElement;
    if (!vp || this.observedViewportEl === vp) {
      // Already observing the right element.  Re-measure only until the row
      // height has been captured from a real card, so the steady state doesn't
      // force a reflow on every change-detection cycle.
      if (!this.gridHeightMeasured) this.measureGridLayout();
      return;
    }
    this.resizeObserver?.disconnect();
    this.observedViewportEl = vp;
    this.gridHeightMeasured = false;
    this.zone.runOutsideAngular(() => {
      this.resizeObserver = new ResizeObserver(() => this.measureGridLayout());
      this.resizeObserver.observe(vp);
    });
    this.measureGridLayout();
  }

  /**
   * Recompute the grid's column count (from viewport width) and row stride
   * (from a rendered card), applying changes in a fresh CD tick.  Guarded so
   * it converges instead of looping: it only re-renders when something moved.
   */
  private measureGridLayout(): void {
    let changed = this.recomputeGridColumns();
    if (changed) this.rebuildGridRows();

    const vp = this.virtualViewport?.elementRef.nativeElement;
    const card = vp?.querySelector('vt-media-item .media-item') as HTMLElement | null;
    if (card) {
      const measured = Math.round(card.getBoundingClientRect().height + GRID_GAP_PX);
      // Only trust a measurement once the card has actually laid out: a
      // transient near-zero height (image still loading after a zoom) would
      // otherwise be locked in as a tiny ``itemSize`` and collapse
      // virtualization.  Leave ``gridHeightMeasured`` false so the next
      // change-detection / resize pass re-measures against a real card.
      if (measured >= MIN_GRID_ROW_HEIGHT) {
        this.gridHeightMeasured = true;
        if (Math.abs(measured - this.gridRowHeight) > 1) {
          this.gridRowHeight = measured;
          changed = true;
        }
      }
    }

    if (changed) {
      this.zone.run(() => this.cdr.detectChanges());
    }
  }

  /** Map a flat item index to the grid row that contains it (best-effort). */
  private itemIndexToRow(itemIdx: number): number {
    const target = this.cachedOrderedItems[itemIdx];
    if (target) {
      for (let r = 0; r < this.gridRows.length; r++) {
        const row = this.gridRows[r];
        if (row.kind === 'items' && row.items.includes(target)) return r;
      }
    }
    return Math.floor(itemIdx / Math.max(1, this.gridColumns));
  }

  scrollToIndex(index: number): void {
    const behavior: ScrollBehavior = prefersReducedMotion() ? 'auto' : 'smooth';
    if (this.useVirtualScroll && this.virtualViewport) {
      const target = this.useGridVirtual ? this.itemIndexToRow(index) : index;
      this.virtualViewport.scrollToIndex(target, behavior);
      // After scrolling, select the item at this index.
      const item = this.cachedOrderedItems[index];
      if (item) {
        this.mediaSelect.emit(item.media.id);
      }
      return;
    }
    if (!this.listContainer) return;
    const container = this.listContainer.nativeElement;
    const items = container.querySelectorAll('vt-media-item');
    const target = items[index] as HTMLElement | undefined;
    if (!target) return;
    const containerRect = container.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    const offset = targetRect.top - containerRect.top + container.scrollTop;
    container.scrollTo({
      top: offset - containerRect.height / 2 + targetRect.height / 2,
      behavior,
    });
  }

  trackByMediaId(_index: number, item: { media: Media }): number {
    return item.media.id;
  }

  trackByGridRow(index: number, _row: GridRow): number {
    return index;
  }

  /**
   * Ask the metadata cache to prefetch items around the currently visible
   * viewport range.
   *
   * When the collection is small enough to skip virtual scrolling (plain DOM
   * list or grid), every item is rendered, so every item is "visible";
   * prefetch the lot. Without this, small datasets keep showing ``#284``
   * placeholders forever because the only other hydration trigger is an
   * explicit click via ``selectMedia()``.  When virtualized (list or grid),
   * only the rows around the viewport are prefetched.
   *
   * ``MediaMetadataCacheService.ensureLoaded()`` already de-dupes
   * already-cached and already-pending ids, so calling it on every
   * rebuild is cheap.
   */
  private prefetchVisibleMetadata(): void {
    const total = this.cachedOrderedItems.length;
    if (total === 0) return;

    if (!this.useVirtualScroll || !this.virtualViewport) {
      const ids = this.cachedOrderedItems.map((item) => item.media.id);
      this.metadataCache.ensureLoaded(ids);
      return;
    }

    const viewportEl = this.virtualViewport.elementRef.nativeElement;
    const viewportHeight = viewportEl.clientHeight || 600;

    if (this.useGridVirtual) {
      // Grid: the virtualized units are rows of ``gridColumns`` cards.
      const startRow = this.virtualViewport.measureScrollOffset('top') / this.gridRowHeight;
      const visibleRows = Math.ceil(viewportHeight / this.gridRowHeight);
      const cols = Math.max(1, this.gridColumns);
      const bufferRows = Math.ceil(PREFETCH_BUFFER / cols);
      const fromRow = Math.max(0, Math.floor(startRow) - bufferRows);
      const toRow = Math.min(this.gridRows.length, Math.ceil(startRow) + visibleRows + bufferRows);
      const ids: number[] = [];
      for (let r = fromRow; r < toRow; r++) {
        const row = this.gridRows[r];
        if (row.kind === 'items') {
          for (const item of row.items) ids.push(item.media.id);
        }
      }
      this.metadataCache.ensureLoaded(ids);
      return;
    }

    const startIndex = this.virtualViewport.measureScrollOffset('top') / this.listItemHeight;
    const visibleCount = Math.ceil(viewportHeight / this.listItemHeight);

    const from = Math.max(0, Math.floor(startIndex) - PREFETCH_BUFFER);
    const to = Math.min(total, Math.ceil(startIndex) + visibleCount + PREFETCH_BUFFER);

    const ids: number[] = [];
    for (let i = from; i < to; i++) {
      ids.push(this.cachedOrderedItems[i].media.id);
    }
    this.metadataCache.ensureLoaded(ids);
  }
}
