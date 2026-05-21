import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { Media } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';

@Component({
  selector: 'vt-document-viewer',
  standalone: true,
  templateUrl: './document-viewer.component.html',
  styleUrl: './document-viewer.component.scss',
})
export class DocumentViewerComponent implements OnChanges {
  @Input() media!: Media;

  mediaSrc = '';
  // See ImageViewerComponent.lastMediaId — keep the iframe stable across
  // metadata-enrichment ngOnChanges cycles for the same id.
  private lastMediaId: number | null = null;

  constructor(private activeContext: ActiveContextService) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media'] && this.media && this.media.id !== this.lastMediaId) {
      this.lastMediaId = this.media.id;
      this.mediaSrc = this.activeContext.mediaUrl(`/api/medias/${this.media.id}/media`);
    }
  }
}
