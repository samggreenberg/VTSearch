import { Component, ElementRef, EventEmitter, Input, OnChanges, OnDestroy, Output, SimpleChanges, ViewChild } from '@angular/core';
import { NgIf } from '@angular/common';
import { Media } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';

@Component({
  selector: 'vt-video-player',
  standalone: true,
  imports: [NgIf],
  templateUrl: './video-player.component.html',
  styleUrl: './video-player.component.scss',
})
export class VideoPlayerComponent implements OnChanges, OnDestroy {
  @Input() media!: Media;
  @Input() volume = 1;
  @Input() audioPlaying = true;
  @Output() playingChanged = new EventEmitter<boolean>();

  @ViewChild('videoEl') videoRef!: ElementRef<HTMLVideoElement>;

  videoSrc = '';
  videoError = false;

  private clipCheckInterval: ReturnType<typeof setInterval> | null = null;
  // See ImageViewerComponent.lastMediaId; guards against metadata-enrichment
  // ngOnChanges cycles rebuilding videoSrc (and yanking playback) for the
  // same id.
  private lastMediaId: number | null = null;
  // Whether (loadedmetadata) has fired for the current videoSrc. Clip bounds
  // (clip_start/clip_end) often arrive via batch hydration *after* the video
  // has already loaded, on a later ngOnChanges with the same media id; in that
  // case (loadedmetadata) does not fire again, so we (re)apply the bounds here.
  private metadataLoaded = false;

  constructor(private activeContext: ActiveContextService) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media'] && this.media) {
      if (this.media.id !== this.lastMediaId) {
        this.lastMediaId = this.media.id;
        this.videoError = false;
        this.metadataLoaded = false;
        this.stopClipEnforcement();
        this.videoSrc = this.activeContext.mediaUrl(`/api/medias/${this.media.id}/video`);
      } else if (this.metadataLoaded) {
        // Same media id, but metadata (e.g. clip_start/clip_end) may have just
        // arrived via batch hydration. The video already loaded, so
        // (loadedmetadata) won't fire again; apply the clip window now.
        this.applyClipBounds();
      }
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

  onError(): void {
    this.videoError = true;
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
    this.metadataLoaded = true;
    video.volume = this.volume;
    this.applyClipBounds();
    this.syncPlaybackState();
  }

  // Seek into the clip window and (re)start boundary enforcement when the media
  // carries clip extents; otherwise tear enforcement down. Safe to call both on
  // (loadedmetadata) and on later metadata-enrichment ngOnChanges cycles.
  private applyClipBounds(): void {
    const video = this.videoRef?.nativeElement;
    if (!video) return;

    if (this.media?.clip_start != null) {
      const clipStart = this.media.clip_start;
      const clipEnd = this.media.clip_end;
      // Snap into the window only when currently outside it. This handles the
      // initial seek and the case where the full video already started playing
      // before clip extents arrived, without yanking a video already looping
      // correctly within its window.
      if (video.currentTime < clipStart || (clipEnd != null && video.currentTime >= clipEnd)) {
        video.currentTime = clipStart;
      }
      this.startClipEnforcement();
    } else {
      this.stopClipEnforcement();
    }
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
