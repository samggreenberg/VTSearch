import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MediaItem } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';

@Component({
  selector: 'vt-media-item',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './media-item.component.html',
  styleUrl: './media-item.component.scss',
})
export class MediaItemComponent {
  @Input({ required: true }) media!: MediaItem;
  @Input() active = false;
  @Input() voteLabel: 'good' | 'bad' | null = null;
  @Input() score: number | null = null;
  @Input() viewMode: 'grid' | 'list' = 'list';
  @Input() focusMode: 'click' | 'hover' = 'click';

  @Output() select = new EventEmitter<number>();
  @Output() vote = new EventEmitter<{ id: number; vote: 'good' | 'bad' }>();

  constructor(private activeContext: ActiveContextService) {}

  get isGrid(): boolean {
    return this.viewMode === 'grid';
  }

  get thumbnailUrl(): string | null {
    if (!this.isGrid) return null;
    if (this.media.type === 'image' || this.media.type === 'video' || this.media.type === 'document') {
      return this.activeContext.mediaUrl(`/api/medias/${this.media.id}/image`);
    }
    return null;
  }

  get placeholderIcon(): string | null {
    if (!this.isGrid) return null;
    if (this.thumbnailUrl) return null;
    if (this.media.type === 'audio') return '\u266B';
    if (this.media.type === 'text') return '\u00B6';
    return '\u25A1';
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
      event.preventDefault();
      this.vote.emit({ id: this.media.id, vote: 'good' });
    }
  }

  onMouseEnter(): void {
    if (this.focusMode === 'hover') {
      this.select.emit(this.media.id);
    }
  }
}
