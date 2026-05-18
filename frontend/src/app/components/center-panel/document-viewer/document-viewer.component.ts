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

  constructor(private activeContext: ActiveContextService) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media'] && this.media) {
      this.mediaSrc = this.activeContext.mediaUrl(`/api/medias/${this.media.id}/media`);
    }
  }
}
