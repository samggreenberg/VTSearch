import {
  Component,
  Input,
  Output,
  EventEmitter,
  ElementRef,
  ViewChild,
  AfterViewChecked,
  OnChanges,
  SimpleChanges,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ScrollingModule, CdkVirtualScrollViewport } from '@angular/cdk/scrolling';
import { MediaItemComponent } from '../media-item/media-item.component';
import { MediaItem } from '../../../models/api.models';
import { SortedItem } from '../left-panel.component';

/** Threshold above which we switch from plain DOM to CDK virtual scrolling (list mode only). */
const VIRTUAL_SCROLL_THRESHOLD = 500;
/** Approximate height of a single media-item row in list mode (px). */
const LIST_ITEM_HEIGHT = 28;

@Component({
  selector: 'vt-media-list',
  standalone: true,
  imports: [CommonModule, ScrollingModule, MediaItemComponent],
  templateUrl: './media-list.component.html',
  styleUrl: './media-list.component.scss',
})
export class MediaListComponent implements AfterViewChecked, OnChanges {
  @Input() medias: MediaItem[] = [];
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
  @ViewChild('listContainer') listContainer!: ElementRef<HTMLDivElement>;
  @ViewChild(CdkVirtualScrollViewport) virtualViewport?: CdkVirtualScrollViewport;

  /** Cached ordered items — rebuilt only when inputs change, not on every CD cycle. */
  cachedOrderedItems: { media: MediaItem; score: number | null; showThreshold: boolean }[] = [];

  readonly listItemHeight = LIST_ITEM_HEIGHT;

  private pendingScrollToSelected = false;
  private pendingScrollPct: number | null = null;

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
    const mediaMap = new Map(this.medias.map((m) => [m.id, m]));
    const items: { media: MediaItem; score: number | null; showThreshold: boolean }[] = [];

    if (this.sortOrder && this.sortOrder.length > 0) {
      let thresholdInserted = false;
      for (const sorted of this.sortOrder) {
        const media = mediaMap.get(sorted.id);
        if (!media) continue;

        let showThreshold = false;
        if (!thresholdInserted && this.threshold !== null && sorted.score < this.threshold) {
          showThreshold = true;
          thresholdInserted = true;
        }

        items.push({ media, score: this.showScores ? sorted.score : null, showThreshold });
      }
    } else {
      for (const media of this.medias) {
        items.push({ media, score: null, showThreshold: false });
      }
    }

    this.cachedOrderedItems = items;
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
      if (this.useVirtualScroll && this.virtualViewport) {
        // In virtual-scroll mode, find the index and scroll to it.
        const idx = this.cachedOrderedItems.findIndex((i) => i.media.id === this.selectedId);
        if (idx >= 0) {
          this.virtualViewport.scrollToIndex(idx, 'smooth');
        }
      } else if (this.listContainer) {
        const activeEl = this.listContainer.nativeElement.querySelector('.media-item.active');
        if (activeEl) {
          activeEl.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
      }
    }
  }

  scrollToIndex(index: number): void {
    if (this.useVirtualScroll && this.virtualViewport) {
      this.virtualViewport.scrollToIndex(index, 'smooth');
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
      behavior: 'smooth',
    });
  }

  trackByMediaId(_index: number, item: { media: MediaItem }): number {
    return item.media.id;
  }
}
