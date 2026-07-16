import { ChangeDetectionStrategy, Component, effect, inject, input, OnDestroy, signal, untracked } from '@angular/core';
import { Subscription } from 'rxjs';
import { MediasApiService } from '../../../services/medias-api.service';
import { Media } from '../../../models/api.models';

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

  // Signal so the HTTP response (written from a `.subscribe()` callback, which is
  // not on the zoneless CD-notification path) repaints the view.
  readonly text = signal('Loading…');
  private sub: Subscription | null = null;
  // See ImageViewerComponent.lastMediaId; avoid re-fetching the text payload
  // every time the metadata cache hydrates a new reference for the same id.
  private lastMediaId: number | null = null;

  constructor() {
    effect(() => {
      const media = this.media();
      if (media.id === this.lastMediaId) return;
      this.lastMediaId = media.id;
      untracked(() => this.loadText(media.id));
    });
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
  }

  private loadText(mediaId: number): void {
    this.sub?.unsubscribe();
    this.text.set('Loading…');
    this.sub = this.mediasApi.getText(mediaId).subscribe({
      next: (data) => {
        this.text.set(data.content || '');
      },
      error: () => {
        this.text.set('Error loading text content.');
      },
    });
  }
}
