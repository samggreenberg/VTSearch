import { ChangeDetectionStrategy, Component, ElementRef, inject, Input, input, OnChanges, OnDestroy, output, SimpleChanges, ViewChild } from '@angular/core';

import { Media } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-video-player',
  standalone: true,
  imports: [],
  templateUrl: './video-player.component.html',
  styleUrl: './video-player.component.scss',
})
export class VideoPlayerComponent implements OnChanges, OnDestroy {
  private activeContext = inject(ActiveContextService);

  @Input() media!: Media;
  readonly volume = input(1);
  readonly audioPlaying = input(true);
  readonly playingChanged = output<boolean>();

  @ViewChild('videoEl') videoRef!: ElementRef<HTMLVideoElement>;

  videoSrc = '';
  videoError = false;

  // Active clip window bounds while enforcement is on, else null. Enforcement is
  // driven by the <video> element's (timeupdate) event rather than a polling
  // timer, so it only runs while the clip is actually playing and progressing.
  private clipBounds: { start: number; end: number } | null = null;
  // See ImageViewerComponent.lastMediaId; guards against metadata-enrichment
  // ngOnChanges cycles rebuilding videoSrc (and yanking playback) for the
  // same id.
  private lastMediaId: number | null = null;
  // Whether (loadedmetadata) has fired for the current videoSrc. Clip bounds
  // (clip_start/clip_end) often arrive via batch hydration *after* the video
  // has already loaded, on a later ngOnChanges with the same media id; in that
  // case (loadedmetadata) does not fire again, so we (re)apply the bounds here.
  private metadataLoaded = false;

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
      this.videoRef.nativeElement.volume = this.volume();
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
    if (!this.audioPlaying()) {
      this.playingChanged.emit(true);
    }
  }

  onError(): void {
    this.videoError = true;
  }

  onPause(): void {
    if (this.audioPlaying()) {
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
    video.volume = this.volume();
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
    if (this.media?.clip_start == null || this.media?.clip_end == null) {
      this.clipBounds = null;
      return;
    }
    // Arm enforcement; the actual boundary check runs in (timeupdate), which the
    // <video> element fires as playback advances (no timer while paused).
    this.clipBounds = { start: this.media.clip_start, end: this.media.clip_end };
  }

  private stopClipEnforcement(): void {
    this.clipBounds = null;
  }

  // Enforce the clip window as playback advances: when the current time leaves
  // [start, end), loop back to the window start. Driven by the element's
  // (timeupdate) event instead of a 100ms polling interval.
  onTimeUpdate(): void {
    const bounds = this.clipBounds;
    if (!bounds) return;
    const video = this.videoRef?.nativeElement;
    if (!video || video.paused) return;
    if (video.currentTime >= bounds.end || video.currentTime < bounds.start) {
      video.currentTime = bounds.start;
    }
  }

  private syncPlaybackState(): void {
    const video = this.videoRef?.nativeElement;
    if (!video) return;
    const audioPlaying = this.audioPlaying();
    if (audioPlaying && video.paused) {
      video.play().catch(() => {});
    } else if (!audioPlaying && !video.paused) {
      video.pause();
    }
  }
}
