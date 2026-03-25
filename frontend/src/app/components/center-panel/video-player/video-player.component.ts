import { Component, ElementRef, EventEmitter, Input, OnChanges, OnDestroy, Output, SimpleChanges, ViewChild } from '@angular/core';
import { MediaItem } from '../../../models/api.models';

@Component({
  selector: 'vt-video-player',
  standalone: true,
  templateUrl: './video-player.component.html',
  styleUrl: './video-player.component.scss',
})
export class VideoPlayerComponent implements OnChanges, OnDestroy {
  @Input() media!: MediaItem;
  @Input() volume = 1;
  @Input() audioPlaying = true;
  @Output() playingChanged = new EventEmitter<boolean>();

  @ViewChild('videoEl') videoRef!: ElementRef<HTMLVideoElement>;

  videoSrc = '';

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media'] && this.media) {
      this.videoSrc = `/api/medias/${this.media.id}/video`;
    }
    if (changes['volume'] && this.videoRef?.nativeElement) {
      this.videoRef.nativeElement.volume = this.volume;
    }
    if (changes['audioPlaying'] && !changes['media'] && this.videoRef?.nativeElement) {
      this.syncPlaybackState();
    }
  }

  ngOnDestroy(): void {
    const video = this.videoRef?.nativeElement;
    if (video) {
      video.pause();
      video.removeAttribute('src');
      video.load();
    }
  }

  onPlay(): void {
    if (!this.audioPlaying) {
      this.playingChanged.emit(true);
    }
  }

  onPause(): void {
    if (this.audioPlaying) {
      this.playingChanged.emit(false);
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
      this.syncPlaybackState();
    }
  }

  private syncPlaybackState(): void {
    const video = this.videoRef?.nativeElement;
    if (!video) return;
    if (this.audioPlaying && video.paused) {
      video.play().catch(() => {});
    } else if (!this.audioPlaying && !video.paused) {
      video.pause();
    }
  }
}
