import { ChangeDetectionStrategy, Component, effect, inject, input, OnDestroy, signal, untracked } from '@angular/core';
import { Subscription } from 'rxjs';
import { MediasApiService } from '../../../services/medias-api.service';
import { Media, PayloadVariant } from '../../../models/api.models';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-text-viewer',
  standalone: true,
  templateUrl: './text-viewer.component.html',
  styleUrl: './text-viewer.component.scss',
})
export class TextViewerComponent implements OnDestroy {
  private mediasApi = inject(MediasApiService);

  readonly media = input.required<Media>();
  /**
   * Which payload to show: `''` (canonical) or `'original'` (the pre-clean
   * snapshot of an item a cleaner rewrote at load time).  Driven by the
   * parent's Clean/Original toggle.
   */
  readonly variant = input<PayloadVariant>('');

  // Signal so the HTTP response (written from a `.subscribe()` callback, which is
  // not on the zoneless CD-notification path) repaints the view.
  readonly text = signal('Loading…');
  private sub: Subscription | null = null;
  // See ImageViewerComponent.lastMediaId; avoid re-fetching the text payload
  // every time the metadata cache hydrates a new reference for the same id.
  private lastMediaId: number | null = null;
  // Same guard for the payload variant: a Clean/Original flip keeps the media
  // id, so the effect has to notice the variant changed to refetch the text.
  private lastVariant: PayloadVariant = '';

  constructor() {
    effect(() => {
      const media = this.media();
      const variant = this.variant();
      if (media.id === this.lastMediaId && variant === this.lastVariant) return;
      this.lastMediaId = media.id;
      this.lastVariant = variant;
      untracked(() => this.loadText(media.id, variant));
    });
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
  }

  private loadText(mediaId: number, variant: PayloadVariant = ''): void {
    this.sub?.unsubscribe();
    this.text.set('Loading…');
    this.sub = this.mediasApi.getText(mediaId, variant).subscribe({
      next: (data) => {
        this.text.set(data.content || '');
      },
      error: () => {
        this.text.set('Error loading text content.');
      },
    });
  }
}
