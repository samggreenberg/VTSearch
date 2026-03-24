import { Component, Input, OnChanges, OnDestroy, SimpleChanges } from '@angular/core';
import { Subscription } from 'rxjs';
import { MediasApiService } from '../../../services/medias-api.service';
import { MediaItem } from '../../../models/api.models';

@Component({
  selector: 'vt-text-viewer',
  standalone: true,
  templateUrl: './text-viewer.component.html',
  styleUrl: './text-viewer.component.scss',
})
export class TextViewerComponent implements OnChanges, OnDestroy {
  @Input() media!: MediaItem;

  text = 'Loading...';
  private sub: Subscription | null = null;

  constructor(private mediasApi: MediasApiService) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media'] && this.media) {
      this.loadText();
    }
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
  }

  private loadText(): void {
    this.sub?.unsubscribe();
    this.text = 'Loading...';
    this.sub = this.mediasApi.getText(this.media.id).subscribe({
      next: (data) => {
        this.text = data.content || '';
      },
      error: () => {
        this.text = 'Error loading text content.';
      },
    });
  }
}
