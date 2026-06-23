import { ChangeDetectionStrategy, Component, inject, Input, OnChanges, SimpleChanges } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { Media } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-document-viewer',
  standalone: true,
  templateUrl: './document-viewer.component.html',
  styleUrl: './document-viewer.component.scss',
})
export class DocumentViewerComponent implements OnChanges {
  private activeContext = inject(ActiveContextService);
  private sanitizer = inject(DomSanitizer);

  @Input() media!: Media;

  // `<object [data]>` is a RESOURCE_URL sink, so Angular rejects a plain
  // string with NG0904. The URL is our own same-origin media endpoint, so
  // wrap it as a trusted resource URL. Null until the first media resolves.
  mediaSrc: SafeResourceUrl | null = null;
  // See ImageViewerComponent.lastMediaId; keep the iframe stable across
  // metadata-enrichment ngOnChanges cycles for the same id.
  private lastMediaId: number | null = null;

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media'] && this.media && this.media.id !== this.lastMediaId) {
      this.lastMediaId = this.media.id;
      this.mediaSrc = this.sanitizer.bypassSecurityTrustResourceUrl(
        this.activeContext.mediaUrl(`/api/medias/${this.media.id}/media`),
      );
    }
  }
}
