import { ChangeDetectionStrategy, Component, effect, inject, input, signal } from '@angular/core';
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
export class DocumentViewerComponent {
  private activeContext = inject(ActiveContextService);
  private sanitizer = inject(DomSanitizer);

  readonly media = input.required<Media>();

  // `<object [data]>` is a RESOURCE_URL sink, so Angular rejects a plain
  // string with NG0904. The URL is our own same-origin media endpoint, so
  // wrap it as a trusted resource URL. Null until the first media resolves.
  // Signal: written from the effect below and read in the template.
  readonly mediaSrc = signal<SafeResourceUrl | null>(null);
  // See ImageViewerComponent.lastMediaId; keep the iframe stable across
  // metadata-enrichment effect cycles for the same id.
  private lastMediaId: number | null = null;

  constructor() {
    effect(() => {
      const media = this.media();
      if (media.id === this.lastMediaId) return;
      this.lastMediaId = media.id;
      this.mediaSrc.set(
        this.sanitizer.bypassSecurityTrustResourceUrl(
          this.activeContext.mediaUrl(`/api/medias/${media.id}/media`),
        ),
      );
    });
  }
}
