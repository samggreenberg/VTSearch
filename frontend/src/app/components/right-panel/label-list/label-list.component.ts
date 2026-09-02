import { ChangeDetectionStrategy, Component, computed, effect, inject, input, output, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { CommonModule } from '@angular/common';
import { Media } from '../../../models/api.models';
import { LabelSortMode } from '../label-sort/label-sort.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { MediaMetadataCacheService } from '../../../services/media-metadata-cache.service';
import { MediaTypeCapabilityService } from '../../../services/media-type-capability.service';
import { VoteGridComponent, VoteGridEntry } from '../vote-grid/vote-grid.component';
import { sortListEntries } from '../../../utils/sort-list-entries';

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
export class LabelListComponent {
  private activeContext = inject(ActiveContextService);
  private metadataCache = inject(MediaMetadataCacheService);
  private mediaTypeCaps = inject(MediaTypeCapabilityService);

  readonly label = input<'good' | 'bad'>('good');
  readonly ids = input<number[]>([]);
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
  readonly foldedNote = input<string | null>(null);
  readonly medias = input<Media[]>([]);
  readonly clickTimes = input<Record<string, number>>({});
  readonly learnedScores = input<Record<string, number>>({});
  /**
   * Normalised [x0, y0, x1, y1] region boxes keyed by media id. When an id has
   * a box (a region vote drawn on an image), its thumbnail is cropped to that
   * region rather than showing the whole frame.
   */
  readonly regionBoxes = input<Record<string, number[]>>({});
  readonly sortMode = input<LabelSortMode>('time-desc');
  readonly gridGoalWidth = input<number>(80);
  readonly focusMode = input<'click' | 'hover'>('click');
  readonly mediaSelected = output<number>();
  readonly mediaVote = output<{
    id: number;
    vote: 'good' | 'bad';
}>();

  /** Pre-built Map for O(1) media stub lookups by id (tracks the medias input). */
  private readonly mediaMap = computed(() => new Map(this.medias().map(m => [m.id, m])));
  // Bumped by the metadataCache version$ subscribe (an unpatched callback, not a
  // zoneless CD trigger) so the sortedEntries computed re-derives entry names as
  // the cache hydrates voted items.
  private readonly cacheVersion = signal(0);
  // Recomputes whenever the inputs it reads change (ids, medias, clickTimes,
  // learnedScores, label, regionBoxes, sortMode) or the metadata cache hydrates;
  // signal inputs don't fire ngOnChanges.
  readonly sortedEntries = computed<LabelEntry[]>(() => {
    this.cacheVersion();
    const entries = this.ids().map(id => this.buildEntry(id));
    return sortListEntries(entries, this.sortMode());
  });

  constructor() {
    // Prefetch metadata for the shown ids whenever they change.
    effect(() => {
      this.metadataCache.ensureLoaded(this.ids());
    });
    // Re-render entry names when the cache hydrates voted items.
    this.metadataCache.version$
      .pipe(takeUntilDestroyed())
      .subscribe(() => {
        this.cacheVersion.update(v => v + 1);
      });
  }

  private lookup(id: number): Media | undefined {
    return this.metadataCache.get(id) ?? this.mediaMap().get(id);
  }

  private buildEntry(id: number): LabelEntry {
    const media = this.lookup(id);
    const name = media ? (media.filename || `Clip #${id}`) : `Clip #${id}`;
    const time = this.clickTimes()[String(id)] ?? -1;
    const score = this.learnedScores()[String(id)] ?? -1;
    let confidence = -1;
    if (score >= 0) {
      confidence = this.label() === 'good' ? score : 1 - score;
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
      missing: !this.mediaMap().has(id),
      // Audio waveforms are theme-agnostic alpha masks (issue #2369); flag them
      // so vote-grid tints them via a CSS mask instead of a plain <img>.
      isAudio: media?.media_type === 'audio',
    };
  }

  private buildThumbnailUrl(media: Media | undefined, id: number): string {
    if (!media || !this.mediaTypeCaps.usesThumbnails(media.media_type)) return '';
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

  onEntrySelected(entry: VoteGridEntry): void {
    this.mediaSelected.emit(Number(entry.key));
  }

  onEntryVote(event: { entry: VoteGridEntry; vote: 'good' | 'bad' }): void {
    this.mediaVote.emit({ id: Number(event.entry.key), vote: event.vote });
  }
}
