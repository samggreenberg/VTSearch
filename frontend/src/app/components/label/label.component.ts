import { Component, AfterViewInit, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { Subscription } from 'rxjs';
import { MediaItem } from '../../models/api.models';
import { MediasApiService } from '../../services/medias-api.service';
import { CenterPanelComponent } from '../center-panel/center-panel.component';

@Component({
  selector: 'vt-label',
  standalone: true,
  imports: [CenterPanelComponent],
  templateUrl: './label.component.html',
  styleUrl: './label.component.scss',
})
export class LabelComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild(CenterPanelComponent) centerPanel!: CenterPanelComponent;

  medias: MediaItem[] = [];
  selectedMedia: MediaItem | null = null;

  private subs: Subscription[] = [];

  constructor(private mediasApi: MediasApiService) {}

  ngOnInit(): void {
    this.subs.push(
      this.mediasApi.getMedias().subscribe((items) => {
        this.medias = items;
        if (items.length > 0 && !this.selectedMedia) {
          this.selectMedia(items[0]);
        }
      }),
    );
  }

  ngAfterViewInit(): void {
    // Initialize center panel after view is ready
    setTimeout(() => this.centerPanel?.init());
  }

  ngOnDestroy(): void {
    this.subs.forEach((s) => s.unsubscribe());
  }

  selectMedia(media: MediaItem): void {
    this.selectedMedia = media;
  }

  onMediaVoted(event: { id: number; vote: 'good' | 'bad' }): void {
    // Auto-advance to next unvoted media
    const currentIndex = this.medias.findIndex((m) => m.id === event.id);
    if (currentIndex >= 0 && currentIndex < this.medias.length - 1) {
      this.selectMedia(this.medias[currentIndex + 1]);
    }
  }
}
