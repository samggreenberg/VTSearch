import { ChangeDetectionStrategy, Component, inject, Input, input, OnChanges, OnDestroy, OnInit, output, signal, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { Media } from '../../../models/api.models';
import { LabelSortMode } from '../label-sort/label-sort.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { MediaMetadataCacheService } from '../../../services/media-metadata-cache.service';
import { THUMBNAIL_MEDIA_TYPES, VoteGridComponent, VoteGridEntry } from '../vote-grid/vote-grid.component';

export interface LabelEntry extends VoteGridEntry {
  id: number;
  time: number;
  score: number;
  confidence: number;
}

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-label-list',
  standalone: true,
  imports: [CommonModule, VoteGridComponent],
  templateUrl: './label-list.component.html',
  styleUrl: './label-list.component.scss',
})
export class LabelListComponent implements OnInit, OnChanges, OnDestroy {
  private activeContext = inject(ActiveContextService);
  private metadataCache = inject(MediaMetadataCacheService);

  @Input() label: 'good' | 'bad' = 'good';
  @Input() ids: number[] = [];
  /**
   * Overrides the bucket heading word ("Goods"/"Bads"). Find mode passes
   * "Verified Good" / "Verified Bad" so the bucket reads as the confirmed pile.
   */
  readonly headingLabel = input<string | null>(null);
  /**
   * Find mode: a small "[N] Unverified Good/Bad" line folded under the heading,
   * counting the items still on the left work queue that the bucket's actions
   * (Browse / Export / To Dataset) act on alongside the shown verified items.
   */
  @Input() foldedNote: string | null = null;
  @Input() medias: Media[] = [];
  @Input() clickTimes: Record<string, number> = {};
  @Input() learnedScores: Record<string, number> = {};
  /**
   * Normalised [x0, y0, x1, y1] region boxes keyed by media id. When an id has
   * a box (a region vote drawn on an image), its thumbnail is cropped to that
   * region rather than showing the whole frame.
   */
  readonly regionBoxes = input<Record<string, number[]>>({});
  @Input() sortMode: LabelSortMode = 'time-desc';
  readonly gridGoalWidth = input<number>(80);
  readonly focusMode = input<'click' | 'hover'>('click');
  readonly mediaSelected = output<number>();
  readonly mediaVote = output<{
    id: number;
    vote: 'good' | 'bad';
}>();

  // Signal: rebuilt from the metadataCache version$ subscribe (an unpatched
  // callback, not a zoneless CD trigger) as well as from ngOnChanges, and read in
  // the template, so it must repaint when the cache hydrates.
  readonly sortedEntries = signal<LabelEntry[]>([]);
  /** Pre-built Map for O(1) media stub lookups by id (rebuilt when medias input changes). */
  private mediaMap = new Map<number, Media>();
  private readonly destroy$ = new Subject<void>();

  ngOnInit(): void {
    this.mediaMap = new Map(this.medias.map(m => [m.id, m]));
    this.metadataCache.ensureLoaded(this.ids);
    this.sortedEntries.set(this.buildSortedEntries());
    // Re-render entry names when the cache hydrates voted items.
    this.metadataCache.version$
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => {
        this.sortedEntries.set(this.buildSortedEntries());
      });
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['medias']) {
      this.mediaMap = new Map(this.medias.map(m => [m.id, m]));
    }
    if (changes['ids']) {
      this.metadataCache.ensureLoaded(this.ids);
    }
    this.sortedEntries.set(this.buildSortedEntries());
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  private lookup(id: number): Media | undefined {
    return this.metadataCache.get(id) ?? this.mediaMap.get(id);
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
    return {
      id,
      key: String(id),
      name,
      time,
      score,
      confidence,
      thumbnailUrl: this.buildThumbnailUrl(media, id),
      fallbackIcon: this.buildFallbackIcon(media),
      missing: !this.mediaMap.has(id),
    };
  }

  private buildThumbnailUrl(media: Media | undefined, id: number): string {
    if (!media || !THUMBNAIL_MEDIA_TYPES.has(media.media_type)) return '';
    // Downscaled tile, not the full-resolution ``/image``: the labeled set can
    // hold hundreds of items, and decoding every full-size bitmap at once
    // exhausts browser memory.
    let url = this.activeContext.mediaUrl(`/api/medias/${id}/thumbnail`);
    // A region-voted item shows a thumbnail of just the voted crop. The box
    // coordinates ride in the query string so a re-vote with a different box
    // is a distinct URL (and cache entry) rather than a stale tile.
    const box = this.regionBoxes()[String(id)];
    if (box && box.length === 4) {
      const sep = url.includes('?') ? '&' : '?';
      url += `${sep}region=${box.map((v) => v.toFixed(4)).join(',')}`;
    }
    return url;
  }

  private buildFallbackIcon(media: Media | undefined): string | null {
    if (!media) return null;
    if (media.media_type === 'audio') return '♫';
    if (media.media_type === 'text') return '¶';
    return '□';
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

  onEntrySelected(entry: VoteGridEntry): void {
    this.mediaSelected.emit(Number(entry.key));
  }

  onEntryVote(event: { entry: VoteGridEntry; vote: 'good' | 'bad' }): void {
    this.mediaVote.emit({ id: Number(event.entry.key), vote: event.vote });
  }
}
