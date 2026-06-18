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
      // Use the downscaled thumbnail endpoint, not the full-resolution
      // ``/image`` route: a grid of hundreds of high-res items would otherwise
      // force the browser to decode every full-size bitmap at once and exhaust
      // memory. The same thumbnail is reused at every zoom level.
      return this.activeContext.mediaUrl(`/api/medias/${this.media.id}/thumbnail`);
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
