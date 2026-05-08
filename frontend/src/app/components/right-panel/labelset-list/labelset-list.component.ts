import {
  AfterViewChecked,
  Component,
  ElementRef,
  EventEmitter,
  Input,
  OnChanges,
  Output,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { LabelElement } from '../../../models/api.models';
import { LabelSortMode } from '../label-sort/label-sort.component';
import { DetectorsApiService } from '../../../services/detectors-api.service';

interface SortedElement extends LabelElement {
  confidence: number;
}

@Component({
  selector: 'vt-labelset-list',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './labelset-list.component.html',
  styleUrls: ['../label-list/label-list.component.scss'],
})
export class LabelsetListComponent implements OnChanges, AfterViewChecked {
  @Input() label: 'good' | 'bad' = 'good';
  @Input() elements: LabelElement[] = [];
  @Input() modelName: string = '';
  @Input() sortMode: LabelSortMode = 'time-desc';
  @Input() viewMode: 'grid' | 'list' = 'grid';
  @Input() gridGoalWidth: number = 80;
  @Input() focusMode: 'click' | 'hover' = 'click';
  @Output() elementSelected = new EventEmitter<LabelElement>();
  @Output() elementVote = new EventEmitter<{ id: string; vote: 'good' | 'bad' }>();

  @ViewChild('voteListContainer') voteListContainer?: ElementRef<HTMLDivElement>;

  sortedEntries: SortedElement[] = [];
  private pendingScrollPct: number | null = null;
  private thumbnailFailedIds = new Set<string>();

  constructor(private detectorsApi: DetectorsApiService) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['viewMode'] && !changes['viewMode'].firstChange && this.voteListContainer) {
      const el = this.voteListContainer.nativeElement;
      const maxScroll = el.scrollHeight - el.clientHeight;
      this.pendingScrollPct = maxScroll > 0 ? el.scrollTop / maxScroll : 0;
    }
    this.sortedEntries = this.buildSorted();
  }

  ngAfterViewChecked(): void {
    if (this.pendingScrollPct !== null && this.voteListContainer) {
      const pct = this.pendingScrollPct;
      this.pendingScrollPct = null;
      const el = this.voteListContainer.nativeElement;
      const maxScroll = el.scrollHeight - el.clientHeight;
      el.scrollTop = pct * maxScroll;
    }
  }

  private buildSorted(): SortedElement[] {
    const entries: SortedElement[] = this.elements.map((e) => ({
      ...e,
      confidence: e.score >= 0 ? (this.label === 'good' ? e.score : 1 - e.score) : -1,
    }));
    return this.sortEntries(entries);
  }

  private sortEntries(entries: SortedElement[]): SortedElement[] {
    const sorted = [...entries];
    switch (this.sortMode) {
      case 'time-desc':
        sorted.sort((a, b) => b.time - a.time);
        break;
      case 'time-asc':
        sorted.sort((a, b) => a.time - b.time);
        break;
      case 'name-asc':
        sorted.sort((a, b) => a.name.localeCompare(b.name));
        break;
      case 'name-desc':
        sorted.sort((a, b) => b.name.localeCompare(a.name));
        break;
      case 'confidence-desc':
        sorted.sort((a, b) => b.confidence - a.confidence);
        break;
      case 'confidence-asc':
        sorted.sort((a, b) => a.confidence - b.confidence);
        break;
      case 'id-asc':
      default:
        sorted.sort((a, b) => a.id.localeCompare(b.id));
        break;
    }
    return sorted;
  }

  get isGrid(): boolean {
    return this.viewMode === 'grid';
  }

  hasThumbnailUrl(entry: LabelElement): boolean {
    if (this.thumbnailFailedIds.has(entry.id)) return false;
    return (
      entry.media_type === 'image' ||
      entry.media_type === 'video' ||
      entry.media_type === 'document' ||
      entry.media_type === 'audio'
    );
  }

  thumbnailUrl(entry: LabelElement): string {
    if (!this.modelName) return '';
    return this.detectorsApi.labelThumbnailUrl(this.modelName, entry.id);
  }

  onThumbnailError(id: string): void {
    this.thumbnailFailedIds.add(id);
  }

  placeholderIcon(entry: LabelElement): string | null {
    if (this.hasThumbnailUrl(entry)) return null;
    if (entry.media_type === 'audio') return '♫';
    if (entry.media_type === 'text') return '¶';
    return '□';
  }

  onEntryClick(entry: LabelElement): void {
    if (this.focusMode === 'hover') {
      this.elementVote.emit({ id: entry.id, vote: 'bad' });
    } else {
      this.elementSelected.emit(entry);
    }
  }

  onEntryContextMenu(event: MouseEvent, entry: LabelElement): void {
    if (this.focusMode === 'hover') {
      event.preventDefault();
      this.elementVote.emit({ id: entry.id, vote: 'good' });
    }
  }

  onEntryMouseEnter(entry: LabelElement): void {
    if (this.focusMode === 'hover') {
      this.elementSelected.emit(entry);
    }
  }

  onEntryKeydown(event: KeyboardEvent, entry: LabelElement): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      this.elementSelected.emit(entry);
    }
  }

  trackById(_index: number, entry: SortedElement): string {
    return entry.id;
  }
}
