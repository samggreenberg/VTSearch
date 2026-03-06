import { Component, ElementRef, Input, OnChanges, SimpleChanges, ViewChild } from '@angular/core';
import { MediaItem } from '../../../models/api.models';

@Component({
  selector: 'vt-video-player',
  standalone: true,
  templateUrl: './video-player.component.html',
  styleUrl: './video-player.component.scss',
})
export class VideoPlayerComponent implements OnChanges {
  @Input() media!: MediaItem;
  @Input() volume = 1;

  @ViewChild('videoEl') videoRef!: ElementRef<HTMLVideoElement>;

  videoSrc = '';

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media'] && this.media) {
      this.videoSrc = `/api/medias/${this.media.id}/video`;
    }
    if (changes['volume'] && this.videoRef?.nativeElement) {
      this.videoRef.nativeElement.volume = this.volume;
    }
  }

  togglePlayback(): void {
    const video = this.videoRef?.nativeElement;
    if (!video) return;
    if (video.paused) {
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  }

  adjustVolume(delta: number): void {
    const video = this.videoRef?.nativeElement;
    if (!video) return;
    video.volume = Math.max(0, Math.min(1, video.volume + delta));
  }

  onLoadedMetadata(): void {
    if (this.videoRef?.nativeElement) {
      this.videoRef.nativeElement.volume = this.volume;
    }
  }
}
