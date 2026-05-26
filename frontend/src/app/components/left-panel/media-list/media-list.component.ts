import {
  Component,
  Input,
  Output,
  EventEmitter,
  ElementRef,
  ViewChild,
  AfterViewChecked,
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

/** Threshold above which we switch from plain DOM to CDK virtual scrolling (list mode only). */
const VIRTUAL_SCROLL_THRESHOLD = 500;
/** Approximate height of a single media-item row in list mode (px). */
const LIST_ITEM_HEIGHT = 28;
/** Extra items to prefetch beyond the visible viewport edges. */
const PREFETCH_BUFFER = 50;

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

  /** Cached ordered items - rebuilt only when inputs change, not on every CD cycle. */
  cachedOrderedItems: { media: Media; score: number | null; showThreshold: boolean; bestRegion: number[] | null }[] = [];

  readonly listItemHeight = LIST_ITEM_HEIGHT;

  private pendingScrollToSelected = false;
  private pendingScrollPct: number | null = null;
  private readonly destroy$ = new Subject<void>();
  private scrollSubscribed = false;

  /** Mirror of ``MediaStateService.loading``; drives the skeleton rows when the
   *  list is empty during the initial /api/medias/ids fetch. */
  loadingMedias = false;
  readonly skeletonPlaceholders = Array.from({ length: 12 });

  constructor(
    private metadataCache: MediaMetadataCacheService,
    private mediaState: MediaStateService,
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
    this.destroy$.next();
    this.destroy$.complete();
  }

  /** Whether to use CDK virtual scrolling (list mode with many items). */
  get useVirtualScroll(): boolean {
    return this.viewMode === 'list' && this.cachedOrderedItems.length > VIRTUAL_SCROLL_THRESHOLD;
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

    // Rebuild cache only when relevant inputs change.
    if (
      changes['medias'] ||
      changes['sortOrder'] ||
      changes['threshold'] ||
      changes['showScores'] ||
      changes['viewMode']
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

    const items: { media: Media; score: number | null; showThreshold: boolean; bestRegion: number[] | null }[] = [];

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

    // Prefetch metadata for the initial visible window.
    this.prefetchVisibleMetadata();
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
        // In virtual-scroll mode, find the index and scroll to it.
        const idx = this.cachedOrderedItems.findIndex((i) => i.media.id === this.selectedId);
        if (idx >= 0) {
          this.virtualViewport.scrollToIndex(idx, behavior);
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

  scrollToIndex(index: number): void {
    const behavior: ScrollBehavior = prefersReducedMotion() ? 'auto' : 'smooth';
    if (this.useVirtualScroll && this.virtualViewport) {
      this.virtualViewport.scrollToIndex(index, behavior);
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

  /**
   * Ask the metadata cache to prefetch items around the currently visible
   * viewport range.
   *
   * When the list is short enough to skip virtual scrolling (or we're in
   * grid mode), every item is rendered into the DOM, so every item is
   * "visible" - prefetch the lot. Without this, small datasets keep
   * showing ``#284`` placeholders forever because the only other hydration
   * trigger is an explicit click via ``selectMedia()``.
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
