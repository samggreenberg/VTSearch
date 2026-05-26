import { Component, Input, Output, EventEmitter, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Media } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';

@Component({
  selector: 'vt-media-item',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './media-item.component.html',
  styleUrl: './media-item.component.scss',
})
export class MediaItemComponent implements OnChanges {
  @Input({ required: true }) media!: Media;
  @Input() active = false;
  @Input() voteLabel: 'good' | 'bad' | null = null;
  @Input() score: number | null = null;
  /** Normalised [x0, y0, x1, y1] of the region that scored highest for this
   *  media. Only set when the dataset's embedder is patch-region-aware. */
  @Input() bestRegion: number[] | null = null;
  @Input() viewMode: 'grid' | 'list' = 'list';
  @Input() focusMode: 'click' | 'hover' = 'click';

  @Output() select = new EventEmitter<number>();
  @Output() vote = new EventEmitter<{ id: number; vote: 'good' | 'bad' }>();
  @Output() contextRequest = new EventEmitter<{ id: number; x: number; y: number }>();

  thumbnailFailed = false;
  private lastMediaId: number | null = null;

  constructor(private activeContext: ActiveContextService) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media'] && this.media.id !== this.lastMediaId) {
      this.thumbnailFailed = false;
      this.lastMediaId = this.media.id;
    }
  }

  get isGrid(): boolean {
    return this.viewMode === 'grid';
  }

  get thumbnailUrl(): string | null {
    if (this.thumbnailFailed) return null;
    if (this.media.media_type === 'image' || this.media.media_type === 'video' || this.media.media_type === 'document' || this.media.media_type === 'audio') {
      return this.activeContext.mediaUrl(`/api/medias/${this.media.id}/image`);
    }
    return null;
  }

  get placeholderIcon(): string | null {
    if (!this.isGrid) return null;
    if (this.thumbnailUrl) return null;
    if (this.media.media_type === 'audio') return '\u266B';
    if (this.media.media_type === 'text') return '\u00B6';
    return '\u25A1';
  }

  onThumbnailError(): void {
    this.thumbnailFailed = true;
  }

  /** Percent-position style for the patch-region outline.  Returns null when
   *  the box is missing, malformed, or covers (effectively) the whole image. */
  get bestRegionStyle(): { [key: string]: string } | null {
    const box = this.bestRegion;
    if (!box || box.length !== 4) return null;
    const [x0, y0, x1, y1] = box;
    if (![x0, y0, x1, y1].every((v) => Number.isFinite(v))) return null;
    const w = x1 - x0;
    const h = y1 - y0;
    if (w <= 0 || h <= 0) return null;
    // Hide near-full-image boxes - they're the legacy single-vector fallback
    // and would just outline the whole thumbnail.
    if (w >= 0.99 && h >= 0.99) return null;
    return {
      left: (x0 * 100).toFixed(2) + '%',
      top: (y0 * 100).toFixed(2) + '%',
      width: (w * 100).toFixed(2) + '%',
      height: (h * 100).toFixed(2) + '%',
    };
  }

  get displayName(): string {
    return this.media.filename || this.media.description || `#${this.media.id}`;
  }

  onClick(): void {
    if (this.focusMode === 'hover') {
      this.vote.emit({ id: this.media.id, vote: 'bad' });
    } else {
      this.select.emit(this.media.id);
    }
  }

  onContextMenu(event: MouseEvent): void {
    if (this.focusMode === 'hover') {
      // Hover mode keeps the existing right-click = vote-good shortcut; the
      // context menu is intentionally not available so speed-labeling stays
      // fast. Users can still seed a detector via the dashboard.
      event.preventDefault();
      this.vote.emit({ id: this.media.id, vote: 'good' });
      return;
    }
    event.preventDefault();
    this.contextRequest.emit({ id: this.media.id, x: event.clientX, y: event.clientY });
  }

  onMouseEnter(): void {
    if (this.focusMode === 'hover') {
      this.select.emit(this.media.id);
    }
  }
}
