import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MediaItem } from '../../../models/api.models';

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
  @Input() showThumbnail = true;

  @Output() select = new EventEmitter<number>();

  get thumbnailUrl(): string | null {
    if (!this.showThumbnail) return null;
    if (this.media.type === 'image' || this.media.type === 'video' || this.media.type === 'document') {
      return `/api/medias/${this.media.id}/image`;
    }
    return null;
  }

  get displayName(): string {
    return this.media.filename || this.media.description || `#${this.media.id}`;
  }

  onClick(): void {
    this.select.emit(this.media.id);
  }
}
