import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { MediaItem } from '../../../models/api.models';

@Component({
  selector: 'vt-document-viewer',
  standalone: true,
  templateUrl: './document-viewer.component.html',
  styleUrl: './document-viewer.component.scss',
})
export class DocumentViewerComponent implements OnChanges {
  @Input() media!: MediaItem;

  mediaSrc = '';

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media'] && this.media) {
      this.mediaSrc = `/api/medias/${this.media.id}/media`;
    }
  }
}
