import { Component, EventEmitter, Input, OnInit, Output, OnChanges, SimpleChanges, ViewChild, ElementRef, AfterViewChecked, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { Media } from '../../../models/api.models';
import { LabelSortMode } from '../label-sort/label-sort.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { MediaMetadataCacheService } from '../../../services/media-metadata-cache.service';

export interface LabelEntry {
  id: number;
  name: string;
  time: number;
  score: number;
  confidence: number;
}

@Component({
  selector: 'vt-label-list',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './label-list.component.html',
  styleUrl: './label-list.component.scss',
})
export class LabelListComponent implements OnInit, OnChanges, OnDestroy, AfterViewChecked {
  @Input() label: 'good' | 'bad' = 'good';
  @Input() ids: number[] = [];
  /**
   * Overrides the bucket heading word ("Goods"/"Bads"). Find mode passes
   * "Verified Good" / "Verified Bad" so the bucket reads as the confirmed pile.
   */
  @Input() headingLabel: string | null = null;
  /**
   * Find mode: a small "[N] Unverified Good/Bad" line folded under the heading,
   * counting the items still on the left work queue that the bucket's actions
   * (Browse / Export / To Dataset) act on alongside the shown verified items.
   */
  @Input() foldedNote: string | null = null;
  @Input() medias: Media[] = [];
  @Input() clickTimes: Record<string, number> = {};
  @Input() learnedScores: Record<string, number> = {};
  @Input() sortMode: LabelSortMode = 'time-desc';
  @Input() viewMode: 'grid' | 'list' = 'grid';
  @Input() gridGoalWidth: number = 80;
  @Input() focusMode: 'click' | 'hover' = 'click';
  @Output() mediaSelected = new EventEmitter<number>();
  @Output() mediaVote = new EventEmitter<{ id: number; vote: 'good' | 'bad' }>();

  @ViewChild('voteListContainer') voteListContainer?: ElementRef<HTMLDivElement>;

  sortedEntries: LabelEntry[] = [];
  private pendingScrollPct: number | null = null;
  /** Pre-built Map for O(1) media stub lookups by id (rebuilt when medias input changes). */
  private mediaMap = new Map<number, Media>();
  private readonly destroy$ = new Subject<void>();

  constructor(
    private activeContext: ActiveContextService,
    private metadataCache: MediaMetadataCacheService,
  ) {}

  ngOnInit(): void {
    this.mediaMap = new Map(this.medias.map(m => [m.id, m]));
    this.metadataCache.ensureLoaded(this.ids);
    this.sortedEntries = this.buildSortedEntries();
    // Re-render entry names when the cache hydrates voted items.
    this.metadataCache.version$
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => {
        this.sortedEntries = this.buildSortedEntries();
      });
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['viewMode'] && !changes['viewMode'].firstChange && this.voteListContainer) {
      const el = this.voteListContainer.nativeElement;
      const maxScroll = el.scrollHeight - el.clientHeight;
      this.pendingScrollPct = maxScroll > 0 ? el.scrollTop / maxScroll : 0;
    }
    if (changes['medias']) {
      this.mediaMap = new Map(this.medias.map(m => [m.id, m]));
    }
    if (changes['ids']) {
      this.metadataCache.ensureLoaded(this.ids);
    }
    this.sortedEntries = this.buildSortedEntries();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  private lookup(id: number): Media | undefined {
    return this.metadataCache.get(id) ?? this.mediaMap.get(id);
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

  private buildSortedEntries(): LabelEntry[] {
    const entries = this.ids.map(id => this.buildEntry(id));
    return this.sortEntries(entries);
  }

  private buildEntry(id: number): LabelEntry {
    const media = this.lookup(id);
    const name = media ? (media.filename || `Clip #${id}`) : `Clip #${id}`;
    const time = this.clickTimes[String(id)] ?? -1;
    const score = this.learnedScores[String(id)] ?? -1;
    let confidence = -1;
    if (score >= 0) {
      confidence = this.label === 'good' ? score : 1 - score;
    }
    return { id, name, time, score, confidence };
  }

  private sortEntries(entries: LabelEntry[]): LabelEntry[] {
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
        sorted.sort((a, b) => a.id - b.id);
        break;
    }
    return sorted;
  }

  get isGrid(): boolean {
    return this.viewMode === 'grid';
  }

  private thumbnailFailedUrls = new Set<string>();

  hasThumbnailUrl(id: number): boolean {
    const url = this.thumbnailUrl(id);
    if (url && this.thumbnailFailedUrls.has(url)) return false;
    const media = this.lookup(id);
    return !!media && (media.media_type === 'image' || media.media_type === 'video' || media.media_type === 'document' || media.media_type === 'audio');
  }

  thumbnailUrl(id: number): string {
    const media = this.lookup(id);
    if (!media) return '';
    // Downscaled tile, not the full-resolution ``/image``: the labeled set can
    // hold hundreds of items, and decoding every full-size bitmap at once
    // exhausts browser memory.
    return this.activeContext.mediaUrl(`/api/medias/${id}/thumbnail`);
  }

  onThumbnailError(url: string): void {
    if (url) this.thumbnailFailedUrls.add(url);
  }

  placeholderIcon(id: number): string | null {
    if (this.hasThumbnailUrl(id)) return null;
    const media = this.lookup(id);
    if (!media) return null;
    if (media.media_type === 'audio') return '\u266B';
    if (media.media_type === 'text') return '\u00B6';
    return '\u25A1';
  }

  isMissing(id: number): boolean {
    return !this.mediaMap.has(id);
  }

  onEntryClick(id: number): void {
    if (this.focusMode === 'hover') {
      this.mediaVote.emit({ id, vote: 'bad' });
    } else {
      this.mediaSelected.emit(id);
    }
  }

  onEntryContextMenu(event: MouseEvent, id: number): void {
    if (this.focusMode === 'hover') {
      event.preventDefault();
      this.mediaVote.emit({ id, vote: 'good' });
    }
  }

  onEntryMouseEnter(id: number): void {
    if (this.focusMode === 'hover') {
      this.mediaSelected.emit(id);
    }
  }

  onEntryKeydown(event: KeyboardEvent, id: number): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      this.mediaSelected.emit(id);
    }
  }

  trackById(_index: number, entry: LabelEntry): number {
    return entry.id;
  }
}
