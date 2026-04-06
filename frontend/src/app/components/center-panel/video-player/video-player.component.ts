import { Component, ElementRef, EventEmitter, Input, OnChanges, OnDestroy, Output, SimpleChanges, ViewChild } from '@angular/core';
import { MediaItem } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';

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

  private clipCheckInterval: ReturnType<typeof setInterval> | null = null;

  constructor(private activeContext: ActiveContextService) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media'] && this.media) {
      this.videoSrc = this.activeContext.mediaUrl(`/api/medias/${this.media.id}/video`);
    }
    if (changes['volume'] && this.videoRef?.nativeElement) {
      this.videoRef.nativeElement.volume = this.volume;
    }
    if (changes['audioPlaying'] && !changes['media'] && this.videoRef?.nativeElement) {
      this.syncPlaybackState();
    }
  }

  ngOnDestroy(): void {
    this.stopClipEnforcement();
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
    const video = this.videoRef?.nativeElement;
    if (!video) return;
    video.volume = this.volume;

    // For clipped videos, seek to clip_start and enforce clip boundaries.
    if (this.media?.clip_start != null) {
      video.currentTime = this.media.clip_start;
      this.startClipEnforcement();
    } else {
      this.stopClipEnforcement();
    }

    this.syncPlaybackState();
  }

  private startClipEnforcement(): void {
    this.stopClipEnforcement();
    if (this.media?.clip_start == null || this.media?.clip_end == null) return;

    const clipStart = this.media.clip_start;
    const clipEnd = this.media.clip_end;

    // Poll every 100ms to enforce clip boundaries. When the video
    // reaches clip_end, loop back to clip_start instead of continuing.
    this.clipCheckInterval = setInterval(() => {
      const video = this.videoRef?.nativeElement;
      if (!video || video.paused) return;
      if (video.currentTime >= clipEnd || video.currentTime < clipStart) {
        video.currentTime = clipStart;
      }
    }, 100);
  }

  private stopClipEnforcement(): void {
    if (this.clipCheckInterval != null) {
      clearInterval(this.clipCheckInterval);
      this.clipCheckInterval = null;
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
