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
import { MediaItemComponent } from '../media-item/media-item.component';
import { MediaItem } from '../../../models/api.models';
import { SortedItem } from '../left-panel.component';

@Component({
  selector: 'vt-media-list',
  standalone: true,
  imports: [CommonModule, MediaItemComponent],
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

  private pendingScrollToSelected = false;

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['selectedId'] && !changes['selectedId'].firstChange) {
      this.pendingScrollToSelected = true;
    }
  }

  get orderedItems(): { media: MediaItem; score: number | null; showThreshold: boolean }[] {
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

    return items;
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
    if (this.pendingScrollToSelected && this.listContainer) {
      this.pendingScrollToSelected = false;
      const activeEl = this.listContainer.nativeElement.querySelector('.media-item.active');
      if (activeEl) {
        activeEl.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }
  }

  scrollToIndex(index: number): void {
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
