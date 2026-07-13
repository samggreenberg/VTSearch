import { ChangeDetectionStrategy, Component, inject, Input, OnChanges, OnDestroy, signal, SimpleChanges } from '@angular/core';
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
export class TextViewerComponent implements OnChanges, OnDestroy {
  private mediasApi = inject(MediasApiService);

  @Input() media!: Media;

  // Signal so the HTTP response (written from a `.subscribe()` callback, which is
  // not on the zoneless CD-notification path) repaints the view.
  readonly text = signal('Loading…');
  private sub: Subscription | null = null;
  // See ImageViewerComponent.lastMediaId; avoid re-fetching the text payload
  // every time the metadata cache hydrates a new reference for the same id.
  private lastMediaId: number | null = null;

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media'] && this.media && this.media.id !== this.lastMediaId) {
      this.lastMediaId = this.media.id;
      this.loadText();
    }
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
  }

  private loadText(): void {
    this.sub?.unsubscribe();
    this.text.set('Loading…');
    this.sub = this.mediasApi.getText(this.media.id).subscribe({
      next: (data) => {
        this.text.set(data.content || '');
      },
      error: () => {
        this.text.set('Error loading text content.');
      },
    });
  }
}
